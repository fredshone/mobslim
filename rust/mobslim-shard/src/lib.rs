//! Compiled kernel for mobslim's `ShardSim`: the parallel design of
//! matsim-rust (<https://github.com/matsim-vsp/matsim-rust>), reimplemented
//! from its documentation and design (no code taken; that project is
//! GPL-3.0), on mobslim's plans and outputs.
//!
//! **Workers own their state.** The network's nodes are split into
//! partitions, one worker thread each. A worker owns its nodes, the *exit
//! side* of every link into them (queue, buffer, flow capacity, stuck
//! timer, and the transit stops on the link), and the *entry side* of every
//! link out of them (storage in use, and the waiting list of vehicles
//! departing onto it), plus the persons and transit trips on them. A link
//! whose two ends lie in different partitions is *split*. Its upstream half
//! collects the vehicles that enter it and sends them downstream; its
//! downstream half sends the space freed by vehicles leaving it back
//! upstream. Persons travel in messages too; a trip's state travels with
//! its vehicle (or, off the network, with its message to the next stop).
//!
//! **Each second (tick)** a worker runs, in this order (with junction
//! delays, matsim-rust's order; without them, steps 1, 5, 4, 2, 3, 6, so that
//! a vehicle reaching the end of a link crosses the node in the same tick and
//! an arrival starts its activity at once):
//!
//! 1. activities: persons whose activity starts or ends now, passengers
//!    reaching a stop or their destination, trips starting, by person (trips
//!    after persons);
//! 2. nodes: each active node picks its non-empty in-link buffers at random,
//!    weighted by their flow capacity, and empties each one into the next
//!    links while they have storage (or the front vehicle is stuck);
//! 3. departures: waiting lists enter their link while it has storage (or
//!    the front is stuck);
//! 4. links: the stops on each link run their due tasks (doors, boarding,
//!    alighting, departures); vehicles past their free-speed time move from
//!    queue to buffer while the link has flow capacity, except that a
//!    transit vehicle with a stop on the link serves it first (as in
//!    MATSim: at a blocking stop it holds the queue, at a non-blocking one
//!    it leaves it and rejoins at the front when it departs); vehicles at the
//!    end of their route leave without using any, and arrive; idle links
//!    sleep until due;
//! 5. teleports: teleported persons arrive;
//! 6. messages: it sends this tick's vehicles and freed space to each
//!    *neighbour* (a partition it shares a link with), and waits only for
//!    its neighbours' messages of this tick, not for all workers.
//!
//! Storage is only read in steps 2 and 3, and space freed (or taken by a
//! trip appearing on a link) in a tick is applied at its end, so locally and
//! across a split link it is reused from the next tick. A vehicle stays on a
//! link for at least one tick, and a vehicle sent over a split link enters
//! the downstream queue in the same tick, so a split link behaves like a
//! local one. Everything a link's stops do stays on that link, or goes into
//! a message for a later tick. With per-node random streams and
//! person-ordered activity steps, **trajectories do not depend on the
//! partitioning**; events are merged into a canonical order (by tick, then
//! by the person or trip acting, step).
//!
//! **Persons** move between workers only when an activity, teleport, walk
//! or ride ends elsewhere, always at least one tick ahead. Between
//! neighbours, channel order guarantees delivery in time; for other workers
//! each worker waits, at the start of a tick, until every worker is at most
//! `gate` ticks behind, where `gate` is the shortest such lookahead in the
//! run (one tick with transit).
//!
//! Model details as in matsim-rust's queue dynamics: flow capacity is
//! accumulated per tick (capped at one tick's worth less what waits in the
//! buffer) and consumed when a vehicle enters the buffer; free-speed times
//! are floored to whole ticks (at least one); a front vehicle is stuck once
//! it waited more than the stuck time; an arrival starts its activity one
//! tick later. One deviation, by default: a buffered vehicle keeps its
//! storage until it crosses the node, as in mobslim's other mobsims
//! (`buffer_in_storage`); matsim-rust frees it when the vehicle enters the
//! buffer (`buffer_in_storage=false`).

use std::cell::UnsafeCell;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};

const EPS: f64 = 1e-9;
const MARGIN: f64 = 1e-10; // flow capacity at or below this is none
const NONE: i64 = i64::MIN;
const NEVER: i64 = i64::MAX;
const NIL: u32 = u32::MAX;
const SPINS: u32 = 2000; // polls before blocking on the channel

// recorded event kinds (as HermesSim's: bit `1 << kind` of the event mask)
const EV_ACT_START: u32 = 0;
const EV_ACT_END: u32 = 1;
const EV_DEPARTURE: u32 = 2;
const EV_ARRIVAL: u32 = 3;
const EV_LINK_ENTER: u32 = 4;
const EV_LINK_LEAVE: u32 = 5;
const EV_TELEPORT: u32 = 6;
// transit: "person" holds the trip for vehicle events
const EV_DRIVER_START: u32 = 8;
const EV_VEH_ARRIVE: u32 = 9;
const EV_VEH_DEPART: u32 = 10;
const EV_WAITING: u32 = 11;
const EV_ENTER_VEH: u32 = 12;
const EV_LEAVE_VEH: u32 = 13;
const EV_VEH_LINK_ENTER: u32 = 14;
const EV_VEH_LINK_LEAVE: u32 = 15;
const N_EVENT_KINDS: usize = 16;
const KIND_SHIFT: u32 = 58;
const PERSON_SHIFT: u32 = 29;

// steps of a tick, the last part of an event's order key
const PH_ACT: u64 = 0;
const PH_NODE: u64 = 1;
const PH_DEPART: u64 = 2;
const PH_LINK: u64 = 3;
const PH_TELE: u64 = 4;
const PH_RECV: u64 = 5;

// agenda items (`x`: the activity, leg, ride or trip)
const START_ACT: u8 = 0;
const END_ACT: u8 = 1;
const TELEPORT: u8 = 2;
const PAX_AT_STOP: u8 = 3; // x: ride
const PAX_ARRIVE: u8 = 4; // x: leg
const TRIP_START: u8 = 5; // x: trip (p: n_plans + trip)
const TRIP_MOVE: u8 = 6; // a trip off the network reaching its next stop's link (x: trip)
const ACT_KINDS: [u8; 6] = [START_ACT, END_ACT, PAX_AT_STOP, PAX_ARRIVE, TRIP_START, TRIP_MOVE];

// trip tasks, run by the link the trip (or its stop) is on
const STOP_ARRIVE: u8 = 0;
const STOP_DEPART: u8 = 1;
const DOOR_OPEN: u8 = 2;
const ALIGHT_DONE: u8 = 3;
const BOARD_DONE: u8 = 4;
const DOOR_CLOSE: u8 = 5;

// doors of a trip in network mode
const OPENING: u8 = 1;
const OPEN: u8 = 2;
const CLOSING: u8 = 3;
const CLOSED: u8 = 4;

#[inline]
fn ceil_step(t: f64) -> i64 {
    (t - EPS).ceil() as i64
}

#[inline]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Inputs {
    queue_steps: Vec<i64>, // per link: ticks a vehicle stays in the queue (>= 1)
    cap: Vec<f64>,         // per link: flow capacity per tick [veh]
    storage: Vec<f64>,     // per link [veh]
    act_offsets: Vec<i64>,
    act_end_time: Vec<f64>,
    act_duration: Vec<f64>,
    leg_network: Vec<u8>,
    leg_teleport_steps: Vec<i64>,
    route_offsets: Vec<i64>,
    route_links: Vec<u32>, // the plans' route links, then (network mode) the trips'
    stuck_ticks: i64,      // < 0: never stuck
    // true: a buffered vehicle keeps its storage until it crosses the node;
    // false: it frees it on entering the buffer (matsim-rust)
    buffer_in_storage: bool,
    // true: a vehicle that reaches the end of a link crosses the node in the
    // next tick, and an activity starts one tick after the arrival
    // (matsim-rust); false: both in the same tick
    junction_delay: bool,
    end_tick: i64,
    event_mask: i64,
    link_from: Vec<u32>,
    link_to: Vec<u32>,
    act_owner: Vec<u32>, // per activity: the node whose worker runs it
    node_part: Vec<u32>,
    n_parts: usize,
    seed: u64,
    in_off: Vec<usize>, // CSR: links into each node
    in_links: Vec<u32>,
    // transit (as HermesSim's kernel takes it, in whole seconds)
    trip_network: Vec<u8>, // per trip: 1 on the network, 0 on the timetable
    rs_network: Vec<u8>,   // per route stop: its route's trips' mode
    leg_transit: Vec<u8>,
    ride_offsets: Vec<i64>,
    ride_board: Vec<u32>,
    ride_alight: Vec<u32>,
    walk_before: Vec<i64>,
    walk_after: Vec<i64>,
    rs_stop: Vec<u32>,
    rs_link_pos: Vec<i64>,
    rs_await: Vec<u8>,
    stop_blocking: Vec<u8>,
    rs_link: Vec<u32>, // per route stop: its stop's link, whose exit side runs it
    trip_k0: Vec<usize>,
    trip_k1: Vec<usize>,
    trip_start: Vec<i64>,
    trip_stop_offsets: Vec<usize>,
    trip_stop_arr: Vec<i64>,
    trip_stop_dep: Vec<i64>,
    trip_link_offsets: Vec<usize>,
    trip_link_steps: Vec<i64>,
    trip_capacity: Vec<f64>,
    trip_access: Vec<i64>,
    trip_egress: Vec<i64>,
    trip_parallel: Vec<u8>,
    door_open: i64,
    door_close: i64,
    // derived
    n_plan_links: usize,
    trip_pce: Vec<f64>,
    tr_rstart: Vec<usize>, // per trip: its first route position (network mode)
    ride_leg: Vec<u32>,    // per ride: its leg
}

impl Inputs {
    fn n_plans(&self) -> usize {
        self.act_offsets.len() - 1
    }
    fn n_links(&self) -> usize {
        self.cap.len()
    }
    fn n_trips(&self) -> usize {
        self.trip_k0.len()
    }
    #[inline]
    fn part(&self, node: u32) -> usize {
        self.node_part[node as usize] as usize
    }
    #[inline]
    fn act_part(&self, a: usize) -> usize {
        self.part(self.act_owner[a])
    }
    /// The worker that runs `link`'s exit side (and its stops).
    #[inline]
    fn link_part(&self, link: u32) -> usize {
        self.part(self.link_to[link as usize])
    }
    fn planned_end(&self, a: usize, start: i64) -> Option<i64> {
        let e = self.act_end_time[a];
        if !e.is_nan() {
            return Some(if start as f64 > e { start } else { ceil_step(e) });
        }
        let d = self.act_duration[a];
        if !d.is_nan() {
            return Some(start.max(ceil_step(start as f64 + d)));
        }
        None
    }
    fn last_act(&self, p: usize) -> usize {
        self.act_offsets[p + 1] as usize - 1
    }
    #[inline]
    fn next_link(&self, v: Veh) -> Option<u32> {
        let r = v.rpos + 1;
        (r < v.rend).then(|| self.route_links[r as usize])
    }
    #[inline]
    fn is_trip(&self, v: Veh) -> bool {
        v.p as usize >= self.n_plans()
    }
    #[inline]
    fn pce(&self, v: Veh) -> f64 {
        if self.is_trip(v) { self.trip_pce[v.p as usize - self.n_plans()] } else { 1.0 }
    }
    /// The link the trip `d` serves its stop `k` on (network mode), or where
    /// its stop is (schedule mode): the stop's link.
    #[inline]
    fn stop_link(&self, k: usize) -> u32 {
        self.rs_link[k]
    }
}

#[derive(Clone, Copy, Debug)]
struct Veh {
    p: u32,    // person, or n_plans + trip
    leg: u32,  // the person's leg (trips: NIL)
    rpos: u32, // index into route_links of the current link
    rend: u32, // end of the route (exclusive)
}

struct QEntry {
    v: Veh,
    earliest: i64,
}

/// A shared array whose elements are owned by one worker at a time: a
/// route stop's by its link's exit side, a trip's by the worker it is on
/// (or the receiver of the message carrying it, which orders the accesses);
/// access is unchecked.
struct Col<T>(UnsafeCell<Vec<T>>);

unsafe impl<T: Send> Sync for Col<T> {}

impl<T> Col<T> {
    fn new(v: Vec<T>) -> Self {
        Col(UnsafeCell::new(v))
    }
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize) -> &mut T {
        &mut *(*self.0.get()).as_mut_ptr().add(i)
    }
    fn into_inner(self) -> Vec<T> {
        self.0.into_inner()
    }
}

/// Transit state, indexed globally (see `Col`).
struct Shared {
    // route stops: waiting passengers (person, ride) and trips there
    waiting: Col<VecDeque<(u32, u32)>>,
    at_stop: Col<Vec<u32>>,
    // trips
    tr_next: Col<usize>,
    tr_load: Col<i64>,
    tr_visit: Col<i64>,
    tr_state: Col<u8>,
    tr_token: Col<u64>,
    tr_seq: Col<u64>,
    tr_alight_busy: Col<bool>,
    tr_board_busy: Col<bool>,
    tr_boarding: Col<(u32, u32)>,
    tr_pulled: Col<bool>,
    tr_veh: Col<Veh>, // network mode: the trip's vehicle while it is at a stop off the lane
    tr_onboard: Col<Vec<(u32, u32)>>,
    tr_alight_q: Col<VecDeque<(u32, u32)>>,
    // outputs
    ride_wait: Col<f64>,
    ride_board: Col<f64>,
    ride_alight: Col<f64>,
    ride_trip: Col<i64>,
    stop_arr: Col<f64>,
    stop_dep: Col<f64>,
}

impl Shared {
    fn new(inp: &Inputs) -> Self {
        let n_trips = inp.n_trips();
        let n_rs = inp.rs_stop.len();
        let n_rides = inp.ride_board.len();
        let n_ts = *inp.trip_stop_offsets.last().unwrap_or(&0);
        let nan = f64::NAN;
        let veh = Veh { p: NIL, leg: NIL, rpos: 0, rend: 0 };
        Shared {
            waiting: Col::new((0..n_rs).map(|_| VecDeque::new()).collect()),
            at_stop: Col::new((0..n_rs).map(|_| Vec::new()).collect()),
            tr_next: Col::new(inp.trip_k0.clone()),
            tr_load: Col::new(vec![0; n_trips]),
            tr_visit: Col::new(vec![-1; n_trips]),
            tr_state: Col::new(vec![0; n_trips]),
            tr_token: Col::new(vec![0; n_trips]),
            tr_seq: Col::new(vec![0; n_trips]),
            tr_alight_busy: Col::new(vec![false; n_trips]),
            tr_board_busy: Col::new(vec![false; n_trips]),
            tr_boarding: Col::new(vec![(NIL, NIL); n_trips]),
            tr_pulled: Col::new(vec![false; n_trips]),
            tr_veh: Col::new(vec![veh; n_trips]),
            tr_onboard: Col::new((0..n_trips).map(|_| Vec::new()).collect()),
            tr_alight_q: Col::new((0..n_trips).map(|_| VecDeque::new()).collect()),
            ride_wait: Col::new(vec![nan; n_rides]),
            ride_board: Col::new(vec![nan; n_rides]),
            ride_alight: Col::new(vec![nan; n_rides]),
            ride_trip: Col::new(vec![-1; n_rides]),
            stop_arr: Col::new(vec![nan; n_ts]),
            stop_dep: Col::new(vec![nan; n_ts]),
        }
    }
}

/// The exit side of a link, owned by its to-node's worker.
struct Link {
    global: u32,
    to_node: u32, // local node index
    q: VecDeque<QEntry>,
    buf: VecDeque<Veh>,
    buf_pce: f64,
    acc: f64, // flow capacity for the next vehicle into the buffer
    last: i64,
    cap: f64,
    stuck_start: i64,
    sleeping_until: i64,
    active: bool,
    from_part: u32, // NIL: the entry side is on this worker too
    released: f64,  // space freed this tick, for a split link's upstream half
}

/// The entry side of a link, owned by its from-node's worker.
struct Entry {
    global: u32,
    used: f64,
    max: f64,
    waiting: VecDeque<Veh>,
    wait_since: i64,
    departing: bool, // in the worker's list of entries with departures
    link: u32,       // local exit side, or NIL if split
    to_part: u32,
}

#[derive(Clone, Copy, Debug)]
struct Item {
    due: i64,
    p: u32,
    kind: u8,
    x: u32, // activity (START_ACT, END_ACT), leg (TELEPORT, PAX_ARRIVE), ride, or trip
}

enum Msg {
    Sync { from: u32, tick: i64, vehs: Vec<(u32, Veh)>, freed: Vec<(u32, f64)> },
    Persons(Vec<Item>),
    Stop { from: u32 },
}

struct Ctx {
    done: Vec<AtomicI64>, // last tick each worker completed (NEVER once stopped)
    remaining: AtomicUsize, // persons and trips not finished
    scheduled: AtomicI64, // agenda items and trip tasks, including those in messages
    last_progress: AtomicI64,
    start: i64,
    gate: i64,  // lookahead of person messages between non-neighbours
    quiet: i64, // ticks without progress after which nothing can happen
}

type Event = (i64, u64, i64); // (tick, actor << 3 | step, code)
type SyncMsg = (u32, i64, Vec<(u32, Veh)>, Vec<(u32, f64)>); // (from, tick, vehicles, freed)
/// A trip task's key: (tick, link, late, trip, seq), in run order.
type TaskKey = (i64, u32, u8, u32, u64);

/// What a worker hands back.
#[derive(Default)]
struct Report {
    act_start: Vec<(u32, i64)>,
    act_end: Vec<(u32, i64)>,
    dep: Vec<(u32, i64)>,
    arr: Vec<(u32, i64)>,
    enter: Vec<(u32, i64)>,
    done: Vec<u32>,
    held: Vec<(u32, i64, i64)>, // persons still out there: (person, leg, link)
    events: Vec<Event>,
}

struct Worker<'a> {
    me: usize,
    inp: &'a Inputs,
    ctx: &'a Ctx,
    sh: &'a Shared,
    rx: Receiver<Msg>,
    tx: Vec<Sender<Msg>>,
    neighbours: Vec<u32>,
    alive: Vec<bool>,
    lloc: Vec<u32>, // global link -> local exit side
    links: Vec<Link>,
    eloc: Vec<u32>, // global link -> local entry side
    entries: Vec<Entry>,
    nodes: Vec<u32>,
    node_rng: Vec<u64>,
    node_active: Vec<bool>,
    active_nodes: Vec<u32>,
    active_links: Vec<u32>,
    sleeping: BTreeMap<i64, Vec<u32>>,
    departing: Vec<u32>,
    agenda: BTreeMap<i64, Vec<Item>>,
    tasks: BTreeMap<TaskKey, (u8, u64)>, // trip tasks of this worker's links: (kind, token)
    task_wake: BTreeMap<i64, Vec<u32>>,  // local links with tasks due then
    stash: Vec<SyncMsg>,
    out_vehs: Vec<Vec<(u32, Veh)>>,
    out_freed: Vec<Vec<(u32, f64)>>,
    out_persons: Vec<Vec<Item>>,
    freed_links: Vec<u32>,
    // storage freed (or taken) this tick on local entries, applied after the
    // last step reading it
    deferred: Vec<(u32, f64)>,
    starts: Vec<(u32, u32)>, // activities starting this tick, at the arrival: (person, activity)
    rank: [u64; 6], // each step's position in the tick, for the event order
    cand: Vec<(u32, f64)>,
    rec: [bool; N_EVENT_KINDS],
    now: i64,
    phase: u64,
    actor: u64, // NIL as u64: events are keyed by their person; else by this trip
    progressed: bool,
    processed: i64, // agenda items and tasks run this tick
    out: Report,
}

const NO_ACTOR: u64 = u64::MAX;

impl<'a> Worker<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(me: usize, inp: &'a Inputs, ctx: &'a Ctx, sh: &'a Shared, rx: Receiver<Msg>,
           tx: Vec<Sender<Msg>>, neighbours: Vec<u32>) -> Self {
        let n_links = inp.n_links();
        let mut nloc = vec![NIL; inp.node_part.len()];
        let mut nodes = Vec::new();
        for (n, &part) in inp.node_part.iter().enumerate() {
            if part as usize == me {
                nloc[n] = nodes.len() as u32;
                nodes.push(n as u32);
            }
        }
        let (mut lloc, mut eloc) = (vec![NIL; n_links], vec![NIL; n_links]);
        let (mut links, mut entries) = (Vec::new(), Vec::new());
        for l in 0..n_links {
            let (from, to) = (inp.link_from[l], inp.link_to[l]);
            let (pf, pt) = (inp.part(from), inp.part(to));
            if pt == me {
                lloc[l] = links.len() as u32;
                links.push(Link {
                    global: l as u32,
                    to_node: nloc[to as usize],
                    q: VecDeque::new(),
                    buf: VecDeque::new(),
                    buf_pce: 0.0,
                    acc: inp.cap[l],
                    last: 0,
                    cap: inp.cap[l],
                    stuck_start: NONE,
                    sleeping_until: NONE,
                    active: false,
                    from_part: if pf == me { NIL } else { pf as u32 },
                    released: 0.0,
                });
            }
            if pf == me {
                eloc[l] = entries.len() as u32;
                entries.push(Entry {
                    global: l as u32,
                    used: 0.0,
                    max: inp.storage[l],
                    waiting: VecDeque::new(),
                    wait_since: NONE,
                    departing: false,
                    link: lloc[l], // set above if the exit side is here too
                    to_part: pt as u32,
                });
            }
        }
        let node_rng = nodes
            .iter()
            .map(|&n| {
                let mut s = inp.seed ^ (n as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                splitmix(&mut s);
                s
            })
            .collect();
        let k = inp.n_parts;
        let mut rec = [false; N_EVENT_KINDS];
        for (kind, r) in rec.iter_mut().enumerate() {
            *r = inp.event_mask & (1 << kind) != 0;
        }
        Worker {
            me,
            inp,
            ctx,
            sh,
            rx,
            tx,
            alive: (0..k).map(|p| neighbours.contains(&(p as u32))).collect(),
            neighbours,
            lloc,
            links,
            eloc,
            entries,
            node_active: vec![false; nodes.len()],
            nodes,
            node_rng,
            active_nodes: Vec::new(),
            active_links: Vec::new(),
            sleeping: BTreeMap::new(),
            departing: Vec::new(),
            agenda: BTreeMap::new(),
            tasks: BTreeMap::new(),
            task_wake: BTreeMap::new(),
            stash: Vec::new(),
            out_vehs: (0..k).map(|_| Vec::new()).collect(),
            out_freed: (0..k).map(|_| Vec::new()).collect(),
            out_persons: (0..k).map(|_| Vec::new()).collect(),
            freed_links: Vec::new(),
            deferred: Vec::new(),
            starts: Vec::new(),
            // without junction delays a tick runs activities, teleports,
            // links, nodes, departures; with them, activities, nodes,
            // departures, links, teleports (as matsim-rust)
            rank: if inp.junction_delay { [0, 1, 2, 3, 4, 5] } else { [0, 3, 4, 2, 1, 5] },
            cand: Vec::new(),
            rec,
            now: 0,
            phase: PH_ACT,
            actor: NO_ACTOR,
            progressed: false,
            processed: 0,
            out: Report::default(),
        }
    }

    /// Records an event about person (or trip) `p`, ordered under the trip
    /// acting, if any, else under `p`.
    #[inline]
    fn record(&mut self, kind: u32, p: u32, r: usize) {
        if self.rec[kind as usize] {
            let code = (kind as i64) << KIND_SHIFT | (p as i64) << PERSON_SHIFT | r as i64;
            let actor = if self.actor == NO_ACTOR { p as u64 } else { self.actor };
            self.out.events.push((self.now, actor << 3 | self.phase, code));
        }
    }

    /// Records an event of trip `d`, ordered under the trip.
    #[inline]
    fn record_trip(&mut self, kind: u32, d: usize, r: usize) {
        if self.rec[kind as usize] {
            let code = (kind as i64) << KIND_SHIFT | (d as i64) << PERSON_SHIFT | r as i64;
            let actor = (self.inp.n_plans() + d) as u64;
            self.out.events.push((self.now, actor << 3 | self.phase, code));
        }
    }

    fn schedule(&mut self, item: Item, to: usize) {
        self.ctx.scheduled.fetch_add(1, Ordering::Relaxed);
        if to == self.me {
            self.agenda.entry(item.due).or_default().push(item);
        } else {
            self.out_persons[to].push(item);
        }
    }

    // ---------------------------------------------------------- persons

    fn start_act(&mut self, p: u32, a: usize) {
        let now = self.now;
        self.out.act_start.push((a as u32, now));
        self.record(EV_ACT_START, p, a);
        self.progressed = true;
        if a == self.inp.last_act(p as usize) {
            self.out.done.push(p);
            self.ctx.remaining.fetch_sub(1, Ordering::Relaxed);
            return;
        }
        match self.inp.planned_end(a, now) {
            None => {} // stays until the end
            Some(t) if t <= now => self.end_act(p, a),
            Some(t) => {
                self.schedule(Item { due: t, p, kind: END_ACT, x: a as u32 }, self.me);
            }
        }
    }

    fn end_act(&mut self, p: u32, a: usize) {
        let (inp, now) = (self.inp, self.now);
        self.out.act_end.push((a as u32, now));
        self.record(EV_ACT_END, p, a);
        self.progressed = true;
        let leg = a - p as usize;
        self.out.dep.push((leg as u32, now));
        self.record(EV_DEPARTURE, p, leg);
        if inp.leg_network[leg] != 0 {
            let (r0, r1) = (inp.route_offsets[leg] as usize, inp.route_offsets[leg + 1] as usize);
            if r0 == r1 {
                self.arrive(p, leg, Some(inp.act_owner[a]));
                return;
            }
            let e = self.eloc[inp.route_links[r0] as usize];
            assert!(e != NIL, "a departure is run by the worker of its first link's from-node");
            let entry = &mut self.entries[e as usize];
            if entry.waiting.is_empty() {
                entry.wait_since = now;
            }
            entry.waiting.push_back(Veh { p, leg: leg as u32, rpos: r0 as u32, rend: r1 as u32 });
            if !entry.departing {
                entry.departing = true;
                self.departing.push(e);
            }
        } else if inp.leg_transit[leg] != 0 {
            let (r0, r1) = (inp.ride_offsets[leg] as usize, inp.ride_offsets[leg + 1] as usize);
            if r0 == r1 {
                // walk only
                self.walk_to_destination(p, leg, None);
            } else {
                self.walk_to_stop(p, r0, inp.act_owner[a], NIL);
            }
        } else {
            let due = now + inp.leg_teleport_steps[leg].max(1);
            let item = Item { due, p, kind: TELEPORT, x: leg as u32 };
            self.schedule(item, inp.act_part(a + 1));
        }
    }

    /// Arrives at `node` (`None`: a teleport or walk, at the next activity's worker).
    fn arrive(&mut self, p: u32, leg: usize, node: Option<u32>) {
        let inp = self.inp;
        self.out.arr.push((leg as u32, self.now));
        self.record(EV_ARRIVAL, p, leg);
        self.progressed = true;
        let a = leg + p as usize + 1;
        if !inp.junction_delay && node.is_none_or(|n| inp.act_owner[a] == n) {
            // starts now, here (after this step, in person order)
            self.starts.push((p, a as u32));
            return;
        }
        let item = Item { due: self.now + 1, p, kind: START_ACT, x: a as u32 };
        self.schedule(item, inp.act_part(a));
    }

    /// Starts the activities of this step's arrivals, in person order.
    fn flush_starts(&mut self) {
        while !self.starts.is_empty() {
            let mut starts = std::mem::take(&mut self.starts);
            starts.sort_unstable();
            for (p, a) in starts {
                self.start_act(p, a as usize);
            }
        }
    }

    fn run_agenda(&mut self, kinds: &[u8]) {
        let Some(mut items) = self.agenda.remove(&self.now) else { return };
        if let Some((&t, _)) = self.agenda.iter().next() {
            assert!(t > self.now, "a person message arrived late");
        }
        let (mut run, rest): (Vec<Item>, Vec<Item>) =
            items.drain(..).partition(|it| kinds.contains(&it.kind));
        if !rest.is_empty() {
            self.agenda.insert(self.now, rest);
        }
        self.processed += run.len() as i64; // released at the end of the tick
        run.sort_unstable_by_key(|it| (it.p, it.kind));
        for it in run {
            match it.kind {
                START_ACT => self.start_act(it.p, it.x as usize),
                END_ACT => self.end_act(it.p, it.x as usize),
                TELEPORT => {
                    self.record(EV_TELEPORT, it.p, it.x as usize);
                    self.arrive(it.p, it.x as usize, None);
                }
                PAX_AT_STOP => unsafe { self.pax_at_stop(it.p, it.x) },
                PAX_ARRIVE => self.arrive(it.p, it.x as usize, None),
                TRIP_START => unsafe { self.trip_start(it.x as usize) },
                _ => {
                    // a trip off the network reaches the link of its next stop
                    let d = it.x as usize;
                    let k = unsafe { *self.sh.tr_next.at(d) };
                    let t = self.inp.trip_stop_arr[self.trip_stop(d, k)].max(self.now);
                    unsafe { self.task(t, 0, d, STOP_ARRIVE, self.inp.stop_link(k), 0) };
                }
            }
        }
        self.flush_starts();
    }

    // ---------------------------------------------------------- passengers
    // Moves elsewhere, or into a later tick, are agenda items; a walk of 0 s
    // on the spot happens at once.

    /// Walk to ride `r`'s board stop, from `node` (in the activity step) or
    /// from the stop on `link` (in the links step; else NIL).
    fn walk_to_stop(&mut self, p: u32, r: usize, node: u32, link: u32) {
        let inp = self.inp;
        let t = self.now + inp.walk_before[r];
        let stop = inp.stop_link(inp.ride_board[r] as usize);
        let here = if link == NIL { inp.link_to[stop as usize] == node } else { stop == link };
        if t == self.now && here {
            unsafe { self.pax_at_stop(p, r as u32) };
        } else {
            let item = Item { due: t.max(self.now + 1), p, kind: PAX_AT_STOP, x: r as u32 };
            self.schedule(item, inp.link_part(stop));
        }
    }

    fn walk_to_destination(&mut self, p: u32, leg: usize, at: Option<u32>) {
        let inp = self.inp;
        let t = self.now + inp.walk_after[leg];
        let a = leg + p as usize + 1;
        if t == self.now && at.is_some() {
            // after a ride: arrives at the stop's node
            self.arrive(p, leg, at.map(|l| inp.link_to[l as usize]));
        } else if t == self.now {
            self.arrive(p, leg, Some(inp.act_owner[a - 1])); // a walk of 0 s on the spot
        } else {
            self.schedule(Item { due: t, p, kind: PAX_ARRIVE, x: leg as u32 }, inp.act_part(a));
        }
    }

    unsafe fn pax_at_stop(&mut self, p: u32, r: u32) {
        let (inp, sh) = (self.inp, self.sh);
        let k = inp.ride_board[r as usize] as usize;
        *sh.ride_wait.at(r as usize) = self.now as f64;
        self.record(EV_WAITING, p, k);
        self.progressed = true;
        let n_here = sh.at_stop.at(k).len();
        if inp.rs_network[k] == 0 {
            for i in 0..n_here {
                let d = sh.at_stop.at(k)[i] as usize;
                if (*sh.tr_load.at(d) as f64) < inp.trip_capacity[d] {
                    *sh.tr_load.at(d) += 1;
                    self.board(d, p, r);
                    return;
                }
            }
            sh.waiting.at(k).push_back((p, r));
            return;
        }
        sh.waiting.at(k).push_back((p, r));
        for i in 0..n_here {
            let d = sh.at_stop.at(k)[i] as usize;
            if self.serve_new_boarder(d) {
                return;
            }
        }
    }

    unsafe fn board(&mut self, d: usize, p: u32, r: u32) {
        let sh = self.sh;
        *sh.ride_board.at(r as usize) = self.now as f64;
        *sh.ride_trip.at(r as usize) = d as i64;
        sh.tr_onboard.at(d).push((p, r));
        self.record(EV_ENTER_VEH, p, d);
        self.progressed = true;
    }

    unsafe fn alight(&mut self, d: usize, p: u32, r: u32, link: u32) {
        let (inp, sh) = (self.inp, self.sh);
        *sh.ride_alight.at(r as usize) = self.now as f64;
        let onboard = sh.tr_onboard.at(d);
        let i = onboard.iter().position(|&(x, _)| x == p).expect("passenger on board");
        onboard.remove(i);
        *sh.tr_load.at(d) -= 1;
        self.record(EV_LEAVE_VEH, p, d);
        self.progressed = true;
        let leg = inp.ride_leg[r as usize] as usize;
        if (r as i64) + 1 < inp.ride_offsets[leg + 1] {
            self.walk_to_stop(p, r as usize + 1, NIL, link);
        } else {
            self.walk_to_destination(p, leg, Some(link));
        }
    }

    unsafe fn alighters(&self, d: usize, k: usize) -> VecDeque<(u32, u32)> {
        let inp = self.inp;
        self.sh
            .tr_onboard
            .at(d)
            .iter()
            .copied()
            .filter(|&(_, r)| inp.ride_alight[r as usize] as usize == k)
            .collect()
    }

    // ---------------------------------------------------------- trips

    #[inline]
    fn trip_stop(&self, d: usize, k: usize) -> usize {
        self.inp.trip_stop_offsets[d] + k - self.inp.trip_k0[d]
    }

    /// A task of trip `d` on `link` (where the trip or its stop is), at tick `t`.
    unsafe fn task(&mut self, t: i64, late: u8, d: usize, kind: u8, link: u32, token: u64) {
        let seq = *self.sh.tr_seq.at(d) + 1;
        *self.sh.tr_seq.at(d) = seq;
        self.ctx.scheduled.fetch_add(1, Ordering::Relaxed);
        self.tasks.insert((t, link, late, d as u32, seq), (kind, token));
        let ll = self.lloc[link as usize];
        debug_assert!(ll != NIL, "a trip task runs on its link's worker");
        self.task_wake.entry(t).or_default().push(ll);
    }

    unsafe fn trip_start(&mut self, d: usize) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        self.record_trip(EV_DRIVER_START, d, 0);
        self.progressed = true;
        if inp.trip_network[d] == 0 {
            let k = inp.trip_k0[d];
            let t = now.max(inp.trip_stop_arr[inp.trip_stop_offsets[d]]);
            self.task(t, 0, d, STOP_ARRIVE, inp.stop_link(k), 0);
            return;
        }
        // appears at the end of its first link, taking its storage from the
        // next tick
        let rpos = inp.tr_rstart[d];
        let rend = rpos + inp.trip_link_offsets[d + 1] - inp.trip_link_offsets[d];
        let v = Veh { p: (inp.n_plans() + d) as u32, leg: NIL, rpos: rpos as u32, rend: rend as u32 };
        let link = inp.route_links[rpos];
        let pce = inp.trip_pce[d];
        let ll = self.lloc[link as usize];
        let lk = &mut self.links[ll as usize];
        if lk.from_part == NIL {
            self.deferred.push((self.eloc[link as usize], -pce));
        } else {
            if lk.released == 0.0 {
                self.freed_links.push(ll);
            }
            lk.released -= pce;
        }
        let _ = sh;
        self.push_queue_at(ll, v, now);
    }

    unsafe fn has_stop_here(&self, d: usize, v: Veh) -> bool {
        let inp = self.inp;
        let k = *self.sh.tr_next.at(d);
        k < inp.trip_k1[d] && inp.rs_link_pos[k] == (v.rpos as usize - inp.tr_rstart[d]) as i64
    }

    unsafe fn record_arrival(&mut self, d: usize, k: usize) {
        let j = self.trip_stop(d, k);
        *self.sh.stop_arr.at(j) = self.now as f64;
        self.record_trip(EV_VEH_ARRIVE, d, k);
        self.sh.at_stop.at(k).push(d as u32);
        self.progressed = true;
    }

    unsafe fn stop_depart(&mut self, d: usize, token: u64, ll: u32) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        if token != *sh.tr_token.at(d) {
            return;
        }
        let net = inp.trip_network[d] != 0;
        if net && *sh.tr_state.at(d) != CLOSED {
            return;
        }
        let k = *sh.tr_next.at(d);
        let j = self.trip_stop(d, k);
        *sh.stop_dep.at(j) = now as f64;
        self.record_trip(EV_VEH_DEPART, d, k);
        self.progressed = true;
        let trips = sh.at_stop.at(k);
        let i = trips.iter().position(|&x| x as usize == d).expect("trip at stop");
        trips.remove(i);
        *sh.tr_next.at(d) = k + 1;
        *sh.tr_visit.at(d) = -1;
        *sh.tr_state.at(d) = 0;
        let link = self.links[ll as usize].global;
        if !net {
            if k + 1 < inp.trip_k1[d] {
                // on to the next stop: on this link, or an agenda item
                let t = now.max(inp.trip_stop_arr[j + 1]);
                let next = inp.stop_link(k + 1);
                if next == link {
                    self.task(t, 0, d, STOP_ARRIVE, link, 0);
                } else {
                    let item = Item { due: t.max(now + 1), p: (inp.n_plans() + d) as u32,
                                      kind: TRIP_MOVE, x: d as u32 };
                    self.schedule(item, inp.link_part(next));
                }
            } else {
                self.ctx.remaining.fetch_sub(1, Ordering::Relaxed); // the trip is done
            }
            return;
        }
        let v = *sh.tr_veh.at(d);
        if self.has_stop_here(d, v) {
            // several stops on one link
            self.reach_stop(d, ll);
            return;
        }
        if *sh.tr_pulled.at(d) {
            // back from the stop into the lane, at its front
            *sh.tr_pulled.at(d) = false;
            self.links[ll as usize].q.push_front(QEntry { v, earliest: now });
        }
    }

    unsafe fn sched_arrive(&mut self, d: usize, ll: u32) {
        let (inp, sh) = (self.inp, self.sh);
        let k = *sh.tr_next.at(d);
        self.record_arrival(d, k);
        let link = self.links[ll as usize].global;
        for (p, r) in self.alighters(d, k) {
            self.alight(d, p, r, link);
        }
        while !sh.waiting.at(k).is_empty() && (*sh.tr_load.at(d) as f64) < inp.trip_capacity[d] {
            *sh.tr_load.at(d) += 1;
            let (p, r) = sh.waiting.at(k).pop_front().unwrap();
            self.board(d, p, r);
        }
        *sh.tr_token.at(d) += 1;
        let token = *sh.tr_token.at(d);
        let t = self.now.max(inp.trip_stop_dep[self.trip_stop(d, k)]);
        self.task(t, 1, d, STOP_DEPART, link, token);
    }

    unsafe fn reach_stop(&mut self, d: usize, ll: u32) {
        let sh = self.sh;
        let k = *sh.tr_next.at(d);
        *sh.tr_visit.at(d) = k as i64;
        self.record_arrival(d, k);
        *sh.tr_alight_q.at(d) = self.alighters(d, k);
        if !sh.tr_alight_q.at(d).is_empty() || self.boarders(d) {
            self.open_doors(d, ll);
        } else {
            self.ready(d, ll);
        }
    }

    unsafe fn boarders(&self, d: usize) -> bool {
        let sh = self.sh;
        let q = sh.waiting.at(*sh.tr_visit.at(d) as usize);
        !q.is_empty()
            && ((*sh.tr_load.at(d) - sh.tr_alight_q.at(d).len() as i64) as f64)
                < self.inp.trip_capacity[d]
    }

    #[inline]
    fn link_of(&self, ll: u32) -> u32 {
        self.links[ll as usize].global
    }

    unsafe fn open_doors(&mut self, d: usize, ll: u32) {
        let sh = self.sh;
        *sh.tr_state.at(d) = OPENING;
        *sh.tr_token.at(d) += 1;
        let token = *sh.tr_token.at(d);
        self.task(self.now + self.inp.door_open, 0, d, DOOR_OPEN, self.link_of(ll), token);
    }

    unsafe fn door_open(&mut self, d: usize, token: u64, ll: u32) {
        let sh = self.sh;
        if token != *sh.tr_token.at(d) || *sh.tr_state.at(d) != OPENING {
            return;
        }
        *sh.tr_state.at(d) = OPEN;
        self.next_alight(d, ll);
        if self.inp.trip_parallel[d] != 0 || !*sh.tr_alight_busy.at(d) {
            self.next_board(d, ll);
        }
        self.maybe_close(d, ll);
    }

    unsafe fn next_alight(&mut self, d: usize, ll: u32) {
        let busy = !self.sh.tr_alight_q.at(d).is_empty();
        *self.sh.tr_alight_busy.at(d) = busy;
        if busy {
            self.task(self.now + self.inp.trip_egress[d], 0, d, ALIGHT_DONE, self.link_of(ll), 0);
        }
    }

    unsafe fn alight_done(&mut self, d: usize, ll: u32) {
        let sh = self.sh;
        let (p, r) = sh.tr_alight_q.at(d).pop_front().expect("an alighting passenger");
        self.alight(d, p, r, self.link_of(ll));
        self.next_alight(d, ll);
        if self.inp.trip_parallel[d] == 0 && !*sh.tr_alight_busy.at(d) && !*sh.tr_board_busy.at(d) {
            self.next_board(d, ll);
        }
        self.maybe_close(d, ll);
    }

    unsafe fn next_board(&mut self, d: usize, ll: u32) {
        let sh = self.sh;
        if self.boarders(d) {
            let k = *sh.tr_visit.at(d) as usize;
            let pr = sh.waiting.at(k).pop_front().unwrap();
            *sh.tr_load.at(d) += 1;
            *sh.tr_board_busy.at(d) = true;
            *sh.tr_boarding.at(d) = pr;
            self.task(self.now + self.inp.trip_access[d], 0, d, BOARD_DONE, self.link_of(ll), 0);
        } else {
            *sh.tr_board_busy.at(d) = false;
        }
    }

    unsafe fn board_done(&mut self, d: usize, ll: u32) {
        let sh = self.sh;
        let (p, r) = *sh.tr_boarding.at(d);
        *sh.tr_boarding.at(d) = (NIL, NIL);
        self.board(d, p, r);
        self.next_board(d, ll);
        self.maybe_close(d, ll);
    }

    unsafe fn maybe_close(&mut self, d: usize, ll: u32) {
        let sh = self.sh;
        if *sh.tr_state.at(d) == OPEN && !*sh.tr_alight_busy.at(d) && !*sh.tr_board_busy.at(d) {
            *sh.tr_state.at(d) = CLOSING;
            self.task(self.now + self.inp.door_close, 0, d, DOOR_CLOSE, self.link_of(ll), 0);
        }
    }

    unsafe fn door_close(&mut self, d: usize, ll: u32) {
        if *self.sh.tr_state.at(d) == CLOSING {
            self.ready(d, ll);
        }
    }

    /// Doors closed and nobody to serve: depart (or await the departure).
    unsafe fn ready(&mut self, d: usize, ll: u32) {
        let (inp, sh) = (self.inp, self.sh);
        if self.boarders(d) {
            self.open_doors(d, ll);
            return;
        }
        *sh.tr_state.at(d) = CLOSED;
        let k = *sh.tr_visit.at(d) as usize;
        let mut t = self.now;
        if inp.rs_await[k] != 0 {
            t = t.max(inp.trip_stop_dep[self.trip_stop(d, k)]);
        }
        *sh.tr_token.at(d) += 1;
        let token = *sh.tr_token.at(d);
        self.task(t, 1, d, STOP_DEPART, self.link_of(ll), token);
    }

    /// A passenger joined the line at trip `d`'s stop. True if it takes them.
    unsafe fn serve_new_boarder(&mut self, d: usize) -> bool {
        let (inp, sh) = (self.inp, self.sh);
        if !self.boarders(d) {
            return false;
        }
        let ll = self.lloc[inp.stop_link(*sh.tr_visit.at(d) as usize) as usize];
        match *sh.tr_state.at(d) {
            OPEN => {
                if !*sh.tr_board_busy.at(d)
                    && (inp.trip_parallel[d] != 0 || !*sh.tr_alight_busy.at(d))
                {
                    self.next_board(d, ll);
                }
                true
            }
            CLOSED => {
                self.open_doors(d, ll);
                true
            }
            s => s == OPENING || s == CLOSING,
        }
    }

    fn due_range(&self, link: u32) -> std::ops::RangeInclusive<TaskKey> {
        (self.now, link, 0, 0, 0)..=(self.now, link, u8::MAX, u32::MAX, u64::MAX)
    }

    /// The first of `link`'s tasks due now, if any.
    fn next_task(&self, link: u32) -> Option<(TaskKey, (u8, u64))> {
        self.tasks.range(self.due_range(link)).next().map(|(&k, &v)| (k, v))
    }

    /// The link's trip tasks due now, in order (ordinary, then late; by trip).
    unsafe fn run_tasks(&mut self, ll: u32) {
        let link = self.link_of(ll);
        while let Some((key, (kind, token))) = self.next_task(link) {
            self.tasks.remove(&key);
            self.processed += 1;
            let d = key.3 as usize;
            self.actor = (self.inp.n_plans() + d) as u64;
            match kind {
                STOP_ARRIVE => self.sched_arrive(d, ll),
                STOP_DEPART => self.stop_depart(d, token, ll),
                DOOR_OPEN => self.door_open(d, token, ll),
                ALIGHT_DONE => self.alight_done(d, ll),
                BOARD_DONE => self.board_done(d, ll),
                _ => self.door_close(d, ll),
            }
            self.actor = NO_ACTOR;
        }
    }

    // ---------------------------------------------------------- network

    /// Puts `v` on link `link` (storage is checked by the caller).
    fn enter(&mut self, v: Veh, link: u32) {
        let e = self.eloc[link as usize] as usize;
        self.entries[e].used += self.inp.pce(v);
        let local = self.entries[e].link;
        if local == NIL {
            let to = self.entries[e].to_part as usize;
            self.out_vehs[to].push((link, v));
        } else {
            self.push_queue(local, v);
        }
        self.progressed = true;
    }

    fn push_queue(&mut self, l: u32, v: Veh) {
        let inp = self.inp;
        let global = self.links[l as usize].global as usize;
        let steps = if inp.is_trip(v) {
            inp.trip_link_steps[v.rpos as usize - inp.n_plan_links].max(1)
        } else {
            inp.queue_steps[global]
        };
        self.push_queue_at(l, v, self.now + steps);
    }

    fn push_queue_at(&mut self, l: u32, v: Veh, earliest: i64) {
        let (inp, now) = (self.inp, self.now);
        let lk = &mut self.links[l as usize];
        lk.q.push_back(QEntry { v, earliest });
        let global = lk.global as usize;
        // a sleeping link wakes for its first vehicle anyway
        if !lk.active && lk.sleeping_until == NONE {
            lk.active = true;
            self.active_links.push(l);
        }
        self.out.enter.push((v.rpos, now));
        if inp.is_trip(v) {
            self.record_trip(EV_VEH_LINK_ENTER, v.p as usize - inp.n_plans(), global);
        } else {
            self.record(EV_LINK_ENTER, v.p, global);
        }
    }

    fn move_nodes(&mut self) {
        self.phase = self.rank[PH_NODE as usize];
        let mut list = std::mem::take(&mut self.active_nodes);
        list.retain(|&n| {
            let keep = self.move_node(n);
            self.node_active[n as usize] = keep;
            keep
        });
        self.active_nodes = list;
    }

    /// Returns whether an in-link still offers a vehicle.
    fn move_node(&mut self, n: u32) -> bool {
        let inp = self.inp;
        let g = self.nodes[n as usize] as usize;
        let mut cand = std::mem::take(&mut self.cand);
        cand.clear();
        let mut total = 0.0;
        for &l in &inp.in_links[inp.in_off[g]..inp.in_off[g + 1]] {
            let ll = self.lloc[l as usize];
            let lk = &self.links[ll as usize];
            if !lk.buf.is_empty() {
                cand.push((ll, lk.cap));
                total += lk.cap;
            }
        }
        while !cand.is_empty() && total > MARGIN {
            let r = (splitmix(&mut self.node_rng[n as usize]) >> 11) as f64
                * (1.0 / (1u64 << 53) as f64)
                * total;
            let mut acc = 0.0;
            let mut i = cand.len() - 1;
            for (j, c) in cand.iter().enumerate() {
                acc += c.1;
                if acc >= r {
                    i = j;
                    break;
                }
            }
            let (ll, w) = cand.swap_remove(i);
            total -= w;
            self.drain(ll);
        }
        self.cand = cand;
        inp.in_links[inp.in_off[g]..inp.in_off[g + 1]]
            .iter()
            .any(|&l| !self.links[self.lloc[l as usize] as usize].buf.is_empty())
    }

    fn drain(&mut self, ll: u32) {
        let (inp, now) = (self.inp, self.now);
        loop {
            let lk = &self.links[ll as usize];
            let Some(&v) = lk.buf.front() else { break };
            let next = inp.next_link(v).expect("only through traffic waits in a buffer");
            let stuck = inp.stuck_ticks >= 0
                && lk.stuck_start != NONE
                && now - lk.stuck_start > inp.stuck_ticks;
            let entry = &self.entries[self.eloc[next as usize] as usize];
            if !(entry.max - entry.used > 0.0 || stuck) {
                break;
            }
            let pce = inp.pce(v);
            let lk = &mut self.links[ll as usize];
            lk.buf.pop_front();
            lk.buf_pce = if lk.buf.is_empty() { 0.0 } else { lk.buf_pce - pce };
            lk.stuck_start = now;
            let global = lk.global as usize;
            if inp.buffer_in_storage {
                self.release(ll as usize, pce);
            }
            if inp.is_trip(v) {
                self.record_trip(EV_VEH_LINK_LEAVE, v.p as usize - inp.n_plans(), global);
            } else {
                self.record(EV_LINK_LEAVE, v.p, global);
            }
            self.enter(Veh { rpos: v.rpos + 1, ..v }, next);
        }
    }

    fn departures(&mut self) {
        self.phase = self.rank[PH_DEPART as usize];
        let (inp, now) = (self.inp, self.now);
        let mut list = std::mem::take(&mut self.departing);
        list.retain(|&e| {
            loop {
                let entry = &self.entries[e as usize];
                let Some(&v) = entry.waiting.front() else { break };
                let stuck = inp.stuck_ticks >= 0 && now - entry.wait_since > inp.stuck_ticks;
                if !(entry.max - entry.used > 0.0 || stuck) {
                    break;
                }
                let entry = &mut self.entries[e as usize];
                entry.waiting.pop_front();
                entry.wait_since = now;
                let global = entry.global;
                self.enter(v, global);
            }
            let entry = &mut self.entries[e as usize];
            entry.departing = !entry.waiting.is_empty();
            entry.departing
        });
        self.departing = list;
    }

    /// Frees `pce` of storage on link `ll`. Locally, it is applied at the end
    /// of the tick, so that, as across a split link, it is only reused from
    /// the next tick.
    fn release(&mut self, ll: usize, pce: f64) {
        let lk = &mut self.links[ll];
        if lk.from_part == NIL {
            self.deferred.push((self.eloc[lk.global as usize], pce));
        } else {
            if lk.released == 0.0 {
                self.freed_links.push(ll as u32);
            }
            lk.released += pce;
        }
    }

    fn move_links(&mut self) {
        self.phase = self.rank[PH_LINK as usize];
        let now = self.now;
        while let Some(entry) = self.sleeping.first_entry() {
            if *entry.key() > now {
                break;
            }
            let (t, list) = entry.remove_entry();
            for l in list {
                let lk = &mut self.links[l as usize];
                if lk.sleeping_until == t {
                    lk.sleeping_until = NONE;
                    lk.active = true;
                    self.active_links.push(l);
                }
            }
        }
        // links whose stops have tasks due
        while let Some(entry) = self.task_wake.first_entry() {
            if *entry.key() > now {
                break;
            }
            for l in entry.remove() {
                let lk = &mut self.links[l as usize];
                if !lk.active {
                    lk.sleeping_until = NONE;
                    lk.active = true;
                    self.active_links.push(l);
                }
            }
        }
        let mut list = std::mem::take(&mut self.active_links);
        list.retain(|&l| self.move_link(l as usize));
        // links activated meanwhile (none: this step only empties queues)
        debug_assert!(self.active_links.is_empty());
        self.active_links = list;
        self.flush_starts();
    }

    /// Returns whether the link stays active.
    fn move_link(&mut self, ll: usize) -> bool {
        let now = self.now;
        let buffer_was_empty = self.links[ll].buf.is_empty();
        {
            let lk = &mut self.links[ll];
            if lk.last < now {
                lk.acc = (lk.acc + (now - lk.last) as f64 * lk.cap).min(lk.cap - lk.buf_pce);
                lk.last = now;
            }
        }
        let link = self.links[ll].global;
        loop {
            unsafe { self.run_tasks(ll as u32) };
            self.move_queue(ll);
            if self.tasks.range(self.due_range(link)).next().is_none() {
                break;
            }
        }
        let held = self.held_by_stop(ll);
        let lk = &mut self.links[ll];
        if buffer_was_empty && !lk.buf.is_empty() {
            lk.stuck_start = now;
        }
        if !lk.buf.is_empty() {
            let n = lk.to_node as usize;
            if !self.node_active[n] {
                self.node_active[n] = true;
                self.active_nodes.push(n as u32);
            }
        }
        if (lk.q.is_empty() || held) && lk.buf.is_empty() {
            // nothing to move until a vehicle comes or the stop's tasks wake it
            lk.active = false;
            return false;
        }
        if lk.buf.is_empty() && lk.acc >= lk.cap {
            let due = lk.q.front().map_or(NEVER, |f| f.earliest);
            if due > now {
                lk.active = false;
                lk.sleeping_until = due;
                self.sleeping.entry(due).or_default().push(ll as u32);
                return false;
            }
        }
        true
    }

    /// Whether the front of `ll`'s queue is a trip serving a (blocking) stop.
    fn held_by_stop(&self, ll: usize) -> bool {
        let inp = self.inp;
        match self.links[ll].q.front() {
            Some(f) if inp.is_trip(f.v) => unsafe {
                *self.sh.tr_visit.at(f.v.p as usize - inp.n_plans()) >= 0
            },
            _ => false,
        }
    }

    /// Vehicles past their free-speed time move from queue to buffer (trips
    /// serve their stops first), or arrive.
    fn move_queue(&mut self, ll: usize) {
        let (inp, now) = (self.inp, self.now);
        loop {
            let lk = &self.links[ll];
            let Some(front) = lk.q.front() else { break };
            if front.earliest > now {
                break;
            }
            let v = front.v;
            if inp.is_trip(v) {
                let d = v.p as usize - inp.n_plans();
                unsafe {
                    if *self.sh.tr_visit.at(d) >= 0 {
                        break; // serving a blocking stop: it holds the queue
                    }
                    if self.has_stop_here(d, v) {
                        let stop = inp.rs_stop[*self.sh.tr_next.at(d)] as usize;
                        if inp.stop_blocking[stop] == 0 {
                            self.links[ll].q.pop_front(); // into the stop, still in storage
                            *self.sh.tr_pulled.at(d) = true;
                        }
                        *self.sh.tr_veh.at(d) = v;
                        self.actor = v.p as u64;
                        self.reach_stop(d, ll as u32);
                        self.actor = NO_ACTOR;
                        continue;
                    }
                }
            }
            let pce = inp.pce(v);
            match inp.next_link(v) {
                None => {
                    let global = lk.global as usize;
                    self.links[ll].q.pop_front();
                    self.release(ll, pce);
                    if inp.is_trip(v) {
                        // a trip leaves the network
                        self.record_trip(EV_VEH_LINK_LEAVE, v.p as usize - inp.n_plans(), global);
                        self.ctx.remaining.fetch_sub(1, Ordering::Relaxed);
                        self.progressed = true;
                    } else {
                        self.record(EV_LINK_LEAVE, v.p, global);
                        self.arrive(v.p, v.leg as usize, Some(inp.link_to[global]));
                    }
                }
                Some(_) if lk.acc > MARGIN => {
                    let lk = &mut self.links[ll];
                    lk.q.pop_front();
                    lk.acc -= pce;
                    lk.buf_pce += pce;
                    lk.buf.push_back(v);
                    if !inp.buffer_in_storage {
                        self.release(ll, pce);
                    }
                    self.progressed = true;
                }
                Some(_) => break,
            }
        }
    }

    // ---------------------------------------------------------- messages

    fn take(&mut self, msg: Msg, in_tick: bool) -> Option<SyncMsg> {
        match msg {
            Msg::Persons(items) => {
                for it in items {
                    assert!(it.due > self.now || (!in_tick && it.due == self.now),
                            "a person message arrived late");
                    self.agenda.entry(it.due).or_default().push(it);
                }
                None
            }
            Msg::Sync { from, tick, vehs, freed } => Some((from, tick, vehs, freed)),
            Msg::Stop { from } => {
                self.alive[from as usize] = false;
                None
            }
        }
    }

    /// Before a tick: wait for the gate, then collect what has arrived.
    fn collect(&mut self) {
        let target = self.now.saturating_sub(self.ctx.gate);
        if self.ctx.gate < NEVER {
            for d in &self.ctx.done {
                while d.load(Ordering::Acquire) < target {
                    std::hint::spin_loop();
                    std::thread::yield_now();
                }
            }
        }
        while let Ok(msg) = self.rx.try_recv() {
            if let Some(s) = self.take(msg, false) {
                self.stash.push(s);
            }
        }
    }

    fn send(&mut self) {
        for d in 0..self.inp.n_parts {
            if !self.out_persons[d].is_empty() {
                let items = std::mem::take(&mut self.out_persons[d]);
                let _ = self.tx[d].send(Msg::Persons(items));
            }
        }
        for l in std::mem::take(&mut self.freed_links) {
            let lk = &mut self.links[l as usize];
            self.out_freed[lk.from_part as usize].push((lk.global, lk.released));
            lk.released = 0.0;
        }
        for i in 0..self.neighbours.len() {
            let nb = self.neighbours[i] as usize;
            let msg = Msg::Sync {
                from: self.me as u32,
                tick: self.now,
                vehs: std::mem::take(&mut self.out_vehs[nb]),
                freed: std::mem::take(&mut self.out_freed[nb]),
            };
            let _ = self.tx[nb].send(msg);
        }
    }

    fn apply(&mut self, vehs: Vec<(u32, Veh)>, freed: Vec<(u32, f64)>) {
        for (link, r) in freed {
            let e = self.eloc[link as usize] as usize;
            self.entries[e].used -= r;
        }
        for (link, v) in vehs {
            let l = self.lloc[link as usize];
            self.push_queue(l, v);
        }
    }

    /// Waits for this tick's message from every live neighbour.
    fn receive(&mut self) {
        self.phase = self.rank[PH_RECV as usize];
        let now = self.now;
        let mut got = vec![false; self.inp.n_parts];
        let mut early = Vec::new();
        for (from, tick, vehs, freed) in std::mem::take(&mut self.stash) {
            if tick == now {
                got[from as usize] = true;
                self.apply(vehs, freed);
            } else {
                assert!(tick > now, "a neighbour message arrived late");
                early.push((from, tick, vehs, freed));
            }
        }
        self.stash = early;
        let missing = |w: &Self, got: &[bool]| {
            w.neighbours.iter().any(|&nb| w.alive[nb as usize] && !got[nb as usize])
        };
        while missing(self, &got) {
            let mut msg = None;
            for _ in 0..SPINS {
                match self.rx.try_recv() {
                    Ok(m) => {
                        msg = Some(m);
                        break;
                    }
                    Err(_) => std::hint::spin_loop(),
                }
            }
            let msg = match msg {
                Some(m) => m,
                None => self.rx.recv().expect("workers hold every sender"),
            };
            if let Some((from, tick, vehs, freed)) = self.take(msg, true) {
                if tick == now {
                    got[from as usize] = true;
                    self.apply(vehs, freed);
                } else {
                    assert!(tick > now, "a neighbour message arrived late");
                    self.stash.push((from, tick, vehs, freed));
                }
            }
        }
    }

    // ---------------------------------------------------------- run

    fn run(mut self) -> (Report, Receiver<Msg>) {
        let (inp, ctx) = (self.inp, self.ctx);
        // first activities, and trip starts
        for p in 0..inp.n_plans() {
            let a = inp.act_offsets[p] as usize;
            if a == inp.last_act(p) || inp.act_part(a) != self.me {
                continue;
            }
            if let Some(t) = inp.planned_end(a, 0) {
                self.agenda.entry(t.max(0)).or_default().push(Item {
                    due: t.max(0),
                    p: p as u32,
                    kind: END_ACT,
                    x: a as u32,
                });
            }
        }
        for d in 0..inp.n_trips() {
            if trip_part(inp, d) == self.me {
                let due = inp.trip_start[d].max(0);
                let p = (inp.n_plans() + d) as u32;
                self.agenda.entry(due).or_default().push(Item { due, p, kind: TRIP_START,
                                                                 x: d as u32 });
            }
        }
        let mut now = ctx.start;
        let mut running = ctx.start < NEVER && ctx.start <= inp.end_tick;
        while running {
            self.now = now;
            self.progressed = false;
            self.collect();
            self.phase = self.rank[PH_ACT as usize];
            self.run_agenda(&ACT_KINDS);
            if inp.junction_delay {
                self.move_nodes();
                self.departures();
                self.move_links();
                self.phase = self.rank[PH_TELE as usize];
                self.run_agenda(&[TELEPORT]);
            } else {
                self.phase = self.rank[PH_TELE as usize];
                self.run_agenda(&[TELEPORT]);
                self.move_links();
                self.move_nodes();
                self.departures();
            }
            for (e, r) in std::mem::take(&mut self.deferred) {
                self.entries[e as usize].used -= r;
            }
            self.send();
            self.receive();
            // Publish this tick's progress before releasing the agenda items
            // it ran, and those before the tick itself: a worker that sees
            // the items gone or the tick done also sees the progress.
            if self.progressed {
                ctx.last_progress.fetch_max(now, Ordering::AcqRel);
            }
            if self.processed > 0 {
                ctx.scheduled.fetch_sub(std::mem::take(&mut self.processed), Ordering::AcqRel);
            }
            ctx.done[self.me].store(now, Ordering::Release);
            // Nothing can happen any more once nothing is scheduled and no
            // worker has made progress for longer than any timer runs,
            // counted up to the tick every worker has completed (a worker
            // can run ahead of the others when it has no live neighbours).
            let idle = ctx.scheduled.load(Ordering::Acquire) == 0;
            let slowest = ctx.done.iter().map(|d| d.load(Ordering::Acquire)).min().unwrap_or(now);
            let quiet = idle && slowest - ctx.last_progress.load(Ordering::Acquire) > ctx.quiet;
            running = now < inp.end_tick && ctx.remaining.load(Ordering::Acquire) > 0 && !quiet;
            now += 1;
        }
        ctx.done[self.me].store(NEVER, Ordering::Release);
        for &nb in &self.neighbours {
            let _ = self.tx[nb as usize].send(Msg::Stop { from: self.me as u32 });
        }
        // persons still out there
        let n_plans = inp.n_plans() as u32;
        let mut held = Vec::new();
        for entry in &self.entries {
            held.extend(entry.waiting.iter().map(|v| (v.p, v.leg as i64, -1)));
        }
        for lk in &self.links {
            let g = lk.global as i64;
            held.extend(lk.q.iter().filter(|e| e.v.p < n_plans).map(|e| (e.v.p, e.v.leg as i64, g)));
            held.extend(lk.buf.iter().filter(|v| v.p < n_plans).map(|v| (v.p, v.leg as i64, g)));
        }
        for items in self.agenda.values() {
            held.extend(items.iter().filter_map(|it| held_item(inp, it)));
        }
        self.out.held = held;
        let rx = self.rx;
        (self.out, rx)
    }
}

/// The worker that starts trip `d`: at the end of its first link (network
/// mode), else at its first stop's link.
fn trip_part(inp: &Inputs, d: usize) -> usize {
    if inp.trip_network[d] != 0 {
        inp.link_part(inp.route_links[inp.tr_rstart[d]])
    } else {
        inp.link_part(inp.stop_link(inp.trip_k0[d]))
    }
}

/// A person's (leg, link) from an agenda item it is in, if it is a person's.
fn held_item(inp: &Inputs, it: &Item) -> Option<(u32, i64, i64)> {
    match it.kind {
        TELEPORT | PAX_ARRIVE => Some((it.p, it.x as i64, -1)),
        PAX_AT_STOP => Some((it.p, inp.ride_leg[it.x as usize] as i64, -1)),
        TRIP_START | TRIP_MOVE => None,
        _ => Some((it.p, -1, -1)),
    }
}

struct Outputs {
    act_start: Vec<f64>,
    act_end: Vec<f64>,
    dep: Vec<f64>,
    arr: Vec<f64>,
    enter: Vec<f64>,
    stuck: Vec<bool>,
    ev_time: Vec<f64>,
    ev_code: Vec<i64>,
    stuck_leg: Vec<i64>,
    stuck_link: Vec<i64>,
    last_time: f64,
    ride_wait: Vec<f64>,
    ride_board: Vec<f64>,
    ride_alight: Vec<f64>,
    ride_trip: Vec<i64>,
    stop_arr: Vec<f64>,
    stop_dep: Vec<f64>,
}

/// Partitions sharing a link, per partition.
fn neighbours(inp: &Inputs) -> Vec<Vec<u32>> {
    let k = inp.n_parts;
    let mut adj = vec![vec![false; k]; k];
    for l in 0..inp.n_links() {
        let (a, b) = (inp.part(inp.link_from[l]), inp.part(inp.link_to[l]));
        if a != b {
            adj[a][b] = true;
            adj[b][a] = true;
        }
    }
    adj.iter()
        .map(|row| (0..k as u32).filter(|&j| row[j as usize]).collect())
        .collect()
}

/// The shortest lookahead of a person message between partitions that are
/// not neighbours (NEVER if there is none). With transit, walks and rides
/// can end anywhere one tick ahead: one.
fn gate(inp: &Inputs, adj: &[Vec<u32>]) -> i64 {
    if inp.n_trips() > 0 && inp.n_parts > 1 {
        return 1;
    }
    let mut gate = NEVER;
    for p in 0..inp.n_plans() {
        let (a0, a1) = (inp.act_offsets[p] as usize, inp.act_offsets[p + 1] as usize);
        for a in a0..a1.saturating_sub(1) {
            let leg = a - p;
            let (from, look) = if inp.leg_network[leg] != 0 {
                let (r0, r1) = (inp.route_offsets[leg] as usize, inp.route_offsets[leg + 1] as usize);
                let at = if r1 > r0 {
                    inp.part(inp.link_to[inp.route_links[r1 - 1] as usize])
                } else {
                    inp.act_part(a)
                };
                (at, 1)
            } else {
                (inp.act_part(a), inp.leg_teleport_steps[leg].max(1))
            };
            let to = inp.act_part(a + 1);
            if from != to && !adj[from].contains(&(to as u32)) {
                gate = gate.min(look);
            }
        }
    }
    gate
}

fn simulate(inp: &Inputs) -> Outputs {
    let k = inp.n_parts;
    let n_plans = inp.n_plans();
    let adj = neighbours(inp);
    let mut start = NEVER;
    let mut remaining = 0;
    let mut scheduled = 0;
    for p in 0..n_plans {
        let a = inp.act_offsets[p] as usize;
        if a == inp.last_act(p) {
            continue;
        }
        remaining += 1;
        if let Some(t) = inp.planned_end(a, 0) {
            start = start.min(t.max(0));
            scheduled += 1;
        }
    }
    for d in 0..inp.n_trips() {
        remaining += 1;
        scheduled += 1;
        start = start.min(inp.trip_start[d].max(0));
    }
    let max_queue = inp.queue_steps.iter().copied().max().unwrap_or(1);
    let max_headway = inp
        .cap
        .iter()
        .filter(|&&c| c > 0.0)
        .map(|&c| (1.0 / c).ceil() as i64)
        .max()
        .unwrap_or(1);
    let max_trip_link = inp.trip_link_steps.iter().copied().max().unwrap_or(1);
    let ctx = Ctx {
        done: (0..k).map(|_| AtomicI64::new(start.saturating_sub(1))).collect(),
        remaining: AtomicUsize::new(remaining),
        scheduled: AtomicI64::new(scheduled),
        last_progress: AtomicI64::new(start),
        start,
        gate: gate(inp, &adj),
        quiet: max_queue.max(max_trip_link) + max_headway + inp.stuck_ticks.max(0) + k as i64 + 2,
    };
    let sh = Shared::new(inp);
    let (txs, rxs): (Vec<_>, Vec<_>) = (0..k).map(|_| channel::<Msg>()).unzip();
    let results: Vec<(Report, Receiver<Msg>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = rxs
            .into_iter()
            .zip(adj)
            .enumerate()
            .map(|(me, (rx, nb))| {
                let (ctx, sh, tx) = (&ctx, &sh, txs.clone());
                scope.spawn(move || Worker::new(me, inp, ctx, sh, rx, tx, nb).run())
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect()
    });
    drop(txs);

    let nan = f64::NAN;
    let mut act_start = vec![nan; inp.act_end_time.len()];
    let mut act_end = vec![nan; inp.act_end_time.len()];
    let mut dep = vec![nan; inp.leg_network.len()];
    let mut arr = vec![nan; inp.leg_network.len()];
    let mut enter = vec![nan; inp.route_links.len()];
    // a plan of one activity is done from the start
    let mut done: Vec<bool> =
        (0..n_plans).map(|p| inp.act_offsets[p] as usize == inp.last_act(p)).collect();
    let mut held = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    for (rep, rx) in results {
        for (col, log) in [(&mut act_start, &rep.act_start), (&mut act_end, &rep.act_end),
                           (&mut dep, &rep.dep), (&mut arr, &rep.arr), (&mut enter, &rep.enter)] {
            for &(i, t) in log {
                col[i as usize] = t as f64;
            }
        }
        for p in rep.done {
            done[p as usize] = true;
        }
        held.extend(rep.held);
        while let Ok(msg) = rx.try_recv() {
            if let Msg::Persons(items) = msg {
                held.extend(items.iter().filter_map(|it| held_item(inp, it)));
            }
        }
        events.extend(rep.events);
    }
    // passengers waiting at stops or riding when the run ended
    unsafe {
        for k in 0..inp.rs_stop.len() {
            for &(p, r) in sh.waiting.at(k).iter() {
                held.push((p, inp.ride_leg[r as usize] as i64, -1));
            }
        }
        for d in 0..inp.n_trips() {
            for &(p, r) in sh.tr_onboard.at(d).iter() {
                held.push((p, inp.ride_leg[r as usize] as i64, -1));
            }
            for &(p, r) in sh.tr_alight_q.at(d).iter() {
                held.push((p, inp.ride_leg[r as usize] as i64, -1));
            }
            let (p, r) = *sh.tr_boarding.at(d);
            if p != NIL {
                held.push((p, inp.ride_leg[r as usize] as i64, -1));
            }
        }
    }
    // canonical order: by tick, actor, step; one actor's events in one step
    // all come from one worker, in order (stable sort)
    events.sort_by_key(|&(t, key, _)| (t, key));

    let mut where_ = vec![(-1i64, -1i64); n_plans];
    for (p, leg, link) in held {
        where_[p as usize] = (leg, link);
    }
    let stuck: Vec<bool> = done.iter().map(|&d| !d).collect();
    let stuck_leg = (0..n_plans).filter(|&p| stuck[p]).map(|p| where_[p].0).collect();
    let stuck_link = (0..n_plans).filter(|&p| stuck[p]).map(|p| where_[p].1).collect();
    let ran = start < NEVER && start <= inp.end_tick;
    let last = if ran { ctx.last_progress.load(Ordering::Acquire) } else { 0 };
    let mut last_time = last as f64;
    if inp.end_tick < NEVER {
        last_time = last_time.max(inp.end_tick as f64);
    }
    Outputs {
        act_start,
        act_end,
        dep,
        arr,
        enter,
        stuck,
        ev_time: events.iter().map(|e| e.0 as f64).collect(),
        ev_code: events.iter().map(|e| e.2).collect(),
        stuck_leg,
        stuck_link,
        last_time,
        ride_wait: sh.ride_wait.into_inner(),
        ride_board: sh.ride_board.into_inner(),
        ride_alight: sh.ride_alight.into_inner(),
        ride_trip: sh.ride_trip.into_inner(),
        stop_arr: sh.stop_arr.into_inner(),
        stop_dep: sh.stop_dep.into_inner(),
    }
}

fn err(msg: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.to_string())
}

fn csr(n: usize, key: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let mut off = vec![0usize; n + 1];
    for &x in key {
        off[x as usize + 1] += 1;
    }
    for i in 0..n {
        off[i + 1] += off[i];
    }
    let mut pos = off.clone();
    let mut items = vec![0u32; key.len()];
    for (i, &x) in key.iter().enumerate() {
        items[pos[x as usize]] = i as u32;
        pos[x as usize] += 1;
    }
    (off, items)
}

/// Run the ShardSim kernel on `n_parts` worker threads, one per partition
/// of the nodes (`node_part`). Returns `(act_start, act_end, leg_departure,
/// leg_arrival, link_enter, stuck, ev_time, ev_code, stuck_leg, stuck_link,
/// last_time, ride_wait_start, ride_board, ride_alight, ride_trip,
/// stop_arrival, stop_departure)`, as the other mobsim kernels do.
#[pyfunction]
#[pyo3(signature = (*, queue_steps, cap, storage, act_offsets, act_end_time, act_duration,
                    leg_network, leg_teleport_steps, route_offsets, route_links, stuck_time,
                    end_time, event_mask, link_from, link_to, n_nodes, act_owner, node_part,
                    n_parts, seed, network_vehicles, leg_transit, ride_offsets, ride_board,
                    ride_alight, walk_before_steps, walk_after_steps, rs_stop, rs_link_pos,
                    rs_await, stop_blocking, trip_k0, trip_k1, trip_start_step,
                    trip_stop_offsets, trip_stop_arr_step, trip_stop_dep_step,
                    trip_link_offsets, trip_links, trip_link_steps, trip_capacity, trip_pce,
                    trip_access_steps, trip_egress_steps, trip_parallel, door_open_steps,
                    door_close_steps, trip_network, rs_network, rs_link, buffer_in_storage=true, junction_delay=1))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    queue_steps: PyReadonlyArray1<'py, i64>,
    cap: PyReadonlyArray1<'py, f64>,
    storage: PyReadonlyArray1<'py, f64>,
    act_offsets: PyReadonlyArray1<'py, i64>,
    act_end_time: PyReadonlyArray1<'py, f64>,
    act_duration: PyReadonlyArray1<'py, f64>,
    leg_network: PyReadonlyArray1<'py, u8>,
    leg_teleport_steps: PyReadonlyArray1<'py, i64>,
    route_offsets: PyReadonlyArray1<'py, i64>,
    route_links: PyReadonlyArray1<'py, i64>,
    stuck_time: f64,
    end_time: f64,
    event_mask: i64,
    link_from: PyReadonlyArray1<'py, i64>,
    link_to: PyReadonlyArray1<'py, i64>,
    n_nodes: usize,
    act_owner: PyReadonlyArray1<'py, i64>,
    node_part: PyReadonlyArray1<'py, i64>,
    n_parts: usize,
    seed: u64,
    network_vehicles: i64,
    leg_transit: PyReadonlyArray1<'py, u8>,
    ride_offsets: PyReadonlyArray1<'py, i64>,
    ride_board: PyReadonlyArray1<'py, i64>,
    ride_alight: PyReadonlyArray1<'py, i64>,
    walk_before_steps: PyReadonlyArray1<'py, i64>,
    walk_after_steps: PyReadonlyArray1<'py, i64>,
    rs_stop: PyReadonlyArray1<'py, i64>,
    rs_link_pos: PyReadonlyArray1<'py, i64>,
    rs_await: PyReadonlyArray1<'py, u8>,
    stop_blocking: PyReadonlyArray1<'py, u8>,
    trip_k0: PyReadonlyArray1<'py, i64>,
    trip_k1: PyReadonlyArray1<'py, i64>,
    trip_start_step: PyReadonlyArray1<'py, i64>,
    trip_stop_offsets: PyReadonlyArray1<'py, i64>,
    trip_stop_arr_step: PyReadonlyArray1<'py, i64>,
    trip_stop_dep_step: PyReadonlyArray1<'py, i64>,
    trip_link_offsets: PyReadonlyArray1<'py, i64>,
    trip_links: PyReadonlyArray1<'py, i64>,
    trip_link_steps: PyReadonlyArray1<'py, i64>,
    trip_capacity: PyReadonlyArray1<'py, f64>,
    trip_pce: PyReadonlyArray1<'py, f64>,
    trip_access_steps: PyReadonlyArray1<'py, i64>,
    trip_egress_steps: PyReadonlyArray1<'py, i64>,
    trip_parallel: PyReadonlyArray1<'py, u8>,
    door_open_steps: i64,
    door_close_steps: i64,
    trip_network: PyReadonlyArray1<'py, u8>,
    rs_network: PyReadonlyArray1<'py, u8>,
    rs_link: PyReadonlyArray1<'py, i64>,
    buffer_in_storage: bool,
    junction_delay: i64,
) -> PyResult<Bound<'py, PyTuple>> {
    let n_links = cap.as_array().len();
    let act_offsets = act_offsets.as_array().to_vec();
    if act_offsets.is_empty() {
        return Err(err("act_offsets must not be empty"));
    }
    let n_plans = act_offsets.len() - 1;
    let n_trips = trip_k0.as_array().len();
    if n_plans + n_trips >= (1 << 29) || n_links >= (1 << 29) || n_nodes >= (1 << 31) {
        return Err(err("too many persons, trips, links or nodes"));
    }
    if n_parts == 0 {
        return Err(err("n_parts must be at least 1"));
    }
    let idx = |a: &PyReadonlyArray1<'py, i64>, n: usize, what: &str| -> PyResult<Vec<u32>> {
        let v = a.as_array();
        if v.iter().any(|&x| x < 0 || x as usize >= n) {
            return Err(err(&format!("{what} index out of range")));
        }
        Ok(v.iter().map(|&x| x as u32).collect())
    };
    let usizes = |a: &PyReadonlyArray1<'py, i64>| -> Vec<usize> {
        a.as_array().iter().map(|&x| x.max(0) as usize).collect()
    };
    let mut route_links = idx(&route_links, n_links, "route link")?;
    let link_from = idx(&link_from, n_nodes, "link from-node")?;
    let link_to = idx(&link_to, n_nodes, "link to-node")?;
    let act_owner = idx(&act_owner, n_nodes, "activity owner node")?;
    let node_part = idx(&node_part, n_parts, "partition")?;
    if link_from.len() != n_links || link_to.len() != n_links || node_part.len() != n_nodes
        || queue_steps.as_array().len() != n_links || storage.as_array().len() != n_links
    {
        return Err(err("link and node arrays must match the network"));
    }
    if act_owner.len() != act_end_time.as_array().len() {
        return Err(err("act_owner must have one entry per activity"));
    }
    // transit
    let n_rs = rs_stop.as_array().len();
    let rs_link = idx(&rs_link, n_links, "route stop link")?;
    let ride_board = idx(&ride_board, n_rs.max(1), "board stop")?;
    let ride_alight = idx(&ride_alight, n_rs.max(1), "alight stop")?;
    let trip_links = idx(&trip_links, n_links, "trip link")?;
    let trip_k0 = usizes(&trip_k0);
    let trip_k1 = usizes(&trip_k1);
    let trip_stop_offsets = usizes(&trip_stop_offsets);
    let trip_link_offsets = usizes(&trip_link_offsets);
    let ride_offsets = ride_offsets.as_array().to_vec();
    if rs_link.len() != n_rs
        || trip_k1.len() != n_trips
        || trip_stop_offsets.len() != n_trips + 1
        || trip_link_offsets.len() != n_trips + 1
        || trip_k0.iter().zip(&trip_k1).any(|(&a, &b)| a > b || b > n_rs)
        || ride_offsets.len() != leg_network.as_array().len() + 1
    {
        return Err(err("inconsistent transit arrays"));
    }
    let net_veh = network_vehicles != 0;
    let trip_network = trip_network.as_array().to_vec();
    if trip_network.len() != n_trips {
        return Err(err("trip_network must have one entry per trip"));
    }
    let n_plan_links = route_links.len();
    let mut tr_rstart = vec![0usize; n_trips];
    if net_veh {
        for d in 0..n_trips {
            if trip_network[d] != 0 && trip_link_offsets[d + 1] == trip_link_offsets[d] {
                return Err(err("a trip in network mode needs links"));
            }
            tr_rstart[d] = n_plan_links + trip_link_offsets[d];
        }
        route_links.extend_from_slice(&trip_links);
    }
    let mut ride_leg = vec![0u32; ride_board.len()];
    for leg in 0..ride_offsets.len() - 1 {
        for r in ride_offsets[leg]..ride_offsets[leg + 1] {
            ride_leg[r as usize] = leg as u32;
        }
    }
    let queue_steps: Vec<i64> = queue_steps.as_array().iter().map(|&s| s.max(1)).collect();
    let (in_off, in_links) = csr(n_nodes, &link_to);
    let inp = Inputs {
        queue_steps,
        cap: cap.as_array().to_vec(),
        storage: storage.as_array().to_vec(),
        act_offsets,
        act_end_time: act_end_time.as_array().to_vec(),
        act_duration: act_duration.as_array().to_vec(),
        leg_network: leg_network.as_array().to_vec(),
        leg_teleport_steps: leg_teleport_steps.as_array().to_vec(),
        route_offsets: route_offsets.as_array().to_vec(),
        route_links,
        buffer_in_storage,
        junction_delay: junction_delay != 0,
        stuck_ticks: if stuck_time.is_finite() { stuck_time.floor() as i64 } else { -1 },
        end_tick: if end_time.is_finite() { end_time.floor() as i64 } else { NEVER },
        event_mask,
        link_from,
        link_to,
        act_owner,
        node_part,
        n_parts,
        seed,
        in_off,
        in_links,
        trip_network,
        rs_network: rs_network.as_array().to_vec(),
        leg_transit: leg_transit.as_array().to_vec(),
        ride_offsets,
        ride_board,
        ride_alight,
        walk_before: walk_before_steps.as_array().to_vec(),
        walk_after: walk_after_steps.as_array().to_vec(),
        rs_stop: rs_stop.as_array().iter().map(|&x| x as u32).collect(),
        rs_link_pos: rs_link_pos.as_array().to_vec(),
        rs_await: rs_await.as_array().to_vec(),
        stop_blocking: stop_blocking.as_array().to_vec(),
        rs_link,
        trip_k0,
        trip_k1,
        trip_start: trip_start_step.as_array().to_vec(),
        trip_stop_offsets,
        trip_stop_arr: trip_stop_arr_step.as_array().to_vec(),
        trip_stop_dep: trip_stop_dep_step.as_array().to_vec(),
        trip_link_offsets,
        trip_link_steps: trip_link_steps.as_array().to_vec(),
        trip_capacity: trip_capacity.as_array().to_vec(),
        trip_access: trip_access_steps.as_array().to_vec(),
        trip_egress: trip_egress_steps.as_array().to_vec(),
        trip_parallel: trip_parallel.as_array().to_vec(),
        door_open: door_open_steps,
        door_close: door_close_steps,
        n_plan_links,
        trip_pce: trip_pce.as_array().to_vec(),
        tr_rstart,
        ride_leg,
    };
    let o = py.allow_threads(|| simulate(&inp));
    let items: Vec<Bound<'py, PyAny>> = vec![
        o.act_start.into_pyarray(py).into_any(),
        o.act_end.into_pyarray(py).into_any(),
        o.dep.into_pyarray(py).into_any(),
        o.arr.into_pyarray(py).into_any(),
        o.enter.into_pyarray(py).into_any(),
        o.stuck.into_pyarray(py).into_any(),
        o.ev_time.into_pyarray(py).into_any(),
        o.ev_code.into_pyarray(py).into_any(),
        o.stuck_leg.into_pyarray(py).into_any(),
        o.stuck_link.into_pyarray(py).into_any(),
        PyFloat::new(py, o.last_time).into_any(),
        o.ride_wait.into_pyarray(py).into_any(),
        o.ride_board.into_pyarray(py).into_any(),
        o.ride_alight.into_pyarray(py).into_any(),
        o.ride_trip.into_pyarray(py).into_any(),
        o.stop_arr.into_pyarray(py).into_any(),
        o.stop_dep.into_pyarray(py).into_any(),
    ];
    PyTuple::new(py, items)
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
