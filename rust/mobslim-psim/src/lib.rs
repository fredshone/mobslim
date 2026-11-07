//! Compiled kernel for mobslim's `PSim`.
//!
//! The rules are those of `run_kernel` in `src/mobslim/mobsim/psim.py`
//! (the reference, with the model documented there); both must give
//! identical results, and the results must not depend on the partitioning.
//! `tests/test_psim.py` checks both.
//!
//! The nodes are split into partitions (`node_part`), one thread each. Every
//! piece of state has one owner at a time:
//!
//! * a node, and the exit side of each link into it (its queue, flow slot,
//!   and the source state of that queue): the node's partition;
//! * the entry side of each link (storage in use, the line of persons
//!   waiting to depart onto it, how many sources it refused): the partition
//!   of its from-node;
//! * a route stop (its waiting passengers and the trips there): the
//!   partition of its node (`rs_node`), which also runs the trips' tasks
//!   (doors, boarding, departures) while they are there;
//! * a person or trip: the partition holding the queue, line, stop or agenda
//!   entry it is in; persons and trips move between partitions only in
//!   messages.
//!
//! Each simulated second is one step: every partition runs its persons (in
//! person order, then trip starts in trip order) and nodes (in node order)
//! for that second, writing only state it owns and appending every effect on
//! other state to an outbox. After a barrier, each partition applies the
//! messages addressed to it, publishes its next busy second, and after a
//! second barrier all agree on the next step. Effects within a node's own
//! step happen at once; effects elsewhere wait for the end of the step
//! whether or not they cross a partition, which is what makes the
//! partitioning invisible.
//!
//! The shared arrays are accessed through raw pointers (`Col`). Soundness
//! rests on the ownership rules above: between two barriers each element is
//! touched by one thread only, and the barriers (acquire/release atomics)
//! order the accesses of successive owners.

use std::cell::UnsafeCell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::Mutex;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};

const EPS: f64 = 1e-9;
const NONE: i64 = i64::MIN;
const NIL: u32 = u32::MAX;
const IDLE: i64 = i64::MAX; // a partition with nothing scheduled
const DENSE_LIMIT: i64 = 1 << 21; // ~24 days of one-second buckets

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

// agenda entries: a node wake (top bit) or a person entry `p << 3 | kind`
// (trip starts are persons `n_plans + trip`)
const ACT_END: u64 = 0;
const TELEPORT_ARRIVE: u64 = 1;
const ACT_START: u64 = 2; // an activity starting one second after the arrival
const PAX_AT_STOP: u64 = 3;
const PAX_ARRIVE: u64 = 4;
const TRIP_START: u64 = 5;
const KIND_BITS: u64 = 3;
const NODE_TAG: u64 = 1 << 63;
// event order key within a second: persons by index, then nodes by index
const PHASE_NODE: u64 = 1 << 32;

// trip tasks, run by the node where the trip is
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

/// Tie-breaker between sources ready since the same second (splitmix64).
#[inline]
fn tie_hash(t: i64, source: u64) -> u64 {
    let mut z = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(source + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Inputs {
    free_steps: Vec<i64>,
    headway: Vec<f64>,
    storage: Vec<f64>,
    act_offsets: Vec<i64>,
    act_end_time: Vec<f64>,
    act_duration: Vec<f64>,
    leg_network: Vec<u8>,
    leg_teleport_steps: Vec<i64>,
    route_offsets: Vec<i64>,
    route_links: Vec<u32>, // the plans' route links, then (network mode) the trips'
    stuck_time: f64,
    end_time: f64,
    event_mask: i64,
    link_from: Vec<u32>,
    link_to: Vec<u32>,
    act_owner: Vec<u32>, // per activity: the node that runs the person there
    // 1: a vehicle leaves a link one second later if it crosses a junction
    // after it, and an activity starts one second after the arrival
    junction_delay: i64,
    node_part: Vec<u32>,
    n_parts: usize,
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
    rs_node: Vec<u32>, // per route stop: the node that runs it
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
    v_pce: Vec<f64>,         // per vehicle: persons 1, then the trips' PCE
    tr_rstart: Vec<usize>,   // per trip: its first route position (network mode)
    in_off: Vec<usize>, // CSR: links into each node, by link index
    in_links: Vec<u32>,
    out_off: Vec<usize>, // CSR: links out of each node, by link index
    out_links: Vec<u32>,
}

impl Inputs {
    fn n_plans(&self) -> usize {
        self.act_offsets.len() - 1
    }
    fn n_links(&self) -> usize {
        self.free_steps.len()
    }
    fn n_trips(&self) -> usize {
        self.trip_k0.len()
    }
    #[inline]
    fn part_of(&self, node: u32) -> usize {
        self.node_part[node as usize] as usize
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
    /// The node that starts trip `d`: the end of its first link (network
    /// mode), else its first stop's.
    fn trip_owner(&self, d: usize) -> u32 {
        if self.trip_network[d] != 0 {
            self.link_to[self.route_links[self.tr_rstart[d]] as usize]
        } else {
            self.rs_node[self.trip_k0[d]]
        }
    }
    /// The junction delay before `after`, the next link (-1: none).
    #[inline]
    fn jd(&self, after: i64) -> i64 {
        if after >= 0 { self.junction_delay } else { 0 }
    }
}

fn csr(n: usize, keys: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let mut off = vec![0usize; n + 1];
    for &k in keys {
        off[k as usize + 1] += 1;
    }
    for i in 0..n {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut items = vec![0u32; keys.len()];
    for (link, &k) in keys.iter().enumerate() {
        items[fill[k as usize]] = link as u32;
        fill[k as usize] += 1;
    }
    (off, items)
}

/// A shared array whose elements are owned by one thread at a time (see the
/// module docs); access is unchecked.
struct Col<T>(UnsafeCell<Vec<T>>);

unsafe impl<T: Send> Sync for Col<T> {}

impl<T> Col<T> {
    fn new(v: Vec<T>) -> Self {
        Col(UnsafeCell::new(v))
    }
    /// The element itself, for non-`Copy` elements.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize) -> &mut T {
        &mut *(*self.0.get()).as_mut_ptr().add(i)
    }
    fn into_inner(self) -> Vec<T> {
        self.0.into_inner()
    }
}

impl<T: Copy> Col<T> {
    #[inline]
    unsafe fn get(&self, i: usize) -> T {
        *(*self.0.get()).as_ptr().add(i)
    }
    #[inline]
    unsafe fn set(&self, i: usize, x: T) {
        *(*self.0.get()).as_mut_ptr().add(i) = x;
    }
}

/// State shared by the partitions, indexed globally.
struct Shared {
    // outputs
    act_start: Col<f64>,
    act_end: Col<f64>,
    dep: Col<f64>,
    arr: Col<f64>,
    enter: Col<f64>,
    // persons
    a_act: Col<usize>,
    a_leg: Col<i64>,
    a_done: Col<bool>,
    a_ride: Col<usize>, // current pt ride
    a_ride_end: Col<usize>,
    // vehicles: persons, then trips
    a_rpos: Col<usize>,
    a_rend: Col<usize>,
    a_link: Col<i64>,
    q_next: Col<u32>,     // the vehicle behind it in its queue or line
    q_after: Col<i64>,    // in a queue: its next link, or -1
    q_earliest: Col<i64>, // in a queue: its earliest exit
    // links, exit side (owned by the to-node)
    l_head: Col<u32>,
    l_tail: Col<u32>,
    l_nft: Col<f64>,
    // links, entry side (owned by the from-node)
    l_occ: Col<f64>,
    l_line_head: Col<u32>,
    l_line_tail: Col<u32>,
    l_blocked_on: Col<u32>,
    // sources: incoming link `s`, or the line of link `s - n_links`
    s_blocked: Col<i64>,
    s_target: Col<u32>,
    // rides (owned with their person)
    t_wait: Col<f64>,
    t_board: Col<f64>,
    t_alight: Col<f64>,
    r_trip: Col<i64>,
    // route stops (owned by their node)
    waiting: Col<VecDeque<u32>>,
    at_stop: Col<Vec<u32>>,
    // trips (owned with the trip)
    tr_next: Col<usize>, // next route stop to serve
    tr_load: Col<i64>,
    tr_visit: Col<i64>, // route stop being served (network mode)
    tr_state: Col<u8>,
    tr_token: Col<u64>,
    tr_seq: Col<u64>, // orders a trip's tasks of one second
    tr_alight_busy: Col<bool>,
    tr_board_busy: Col<bool>,
    tr_boarding: Col<u32>,
    tr_pulled: Col<bool>,
    tr_onboard: Col<Vec<u32>>,
    tr_alight_q: Col<VecDeque<u32>>,
    stop_arr: Col<f64>,
    stop_dep: Col<f64>,
}

impl Shared {
    fn new(inp: &Inputs) -> Self {
        let n_plans = inp.n_plans();
        let n_trips = inp.n_trips();
        let n_veh = n_plans + n_trips;
        let n_links = inp.n_links();
        let n_acts = inp.act_end_time.len();
        let n_legs = inp.leg_network.len();
        let n_rides = inp.ride_board.len();
        let n_rs = inp.rs_stop.len();
        let n_trip_stops = *inp.trip_stop_offsets.last().unwrap_or(&0);
        let nan = f64::NAN;
        Shared {
            act_start: Col::new(vec![nan; n_acts]),
            act_end: Col::new(vec![nan; n_acts]),
            dep: Col::new(vec![nan; n_legs]),
            arr: Col::new(vec![nan; n_legs]),
            enter: Col::new(vec![nan; inp.route_links.len()]),
            a_act: Col::new(inp.act_offsets[..n_plans].iter().map(|&o| o as usize).collect()),
            a_leg: Col::new(vec![-1; n_plans]),
            a_done: Col::new(vec![false; n_plans]),
            a_ride: Col::new(vec![0; n_plans]),
            a_ride_end: Col::new(vec![0; n_plans]),
            a_rpos: Col::new(vec![0; n_veh]),
            a_rend: Col::new(vec![0; n_veh]),
            a_link: Col::new(vec![-1; n_veh]),
            q_next: Col::new(vec![NIL; n_veh]),
            q_after: Col::new(vec![-1; n_veh]),
            q_earliest: Col::new(vec![0; n_veh]),
            l_head: Col::new(vec![NIL; n_links]),
            l_tail: Col::new(vec![NIL; n_links]),
            l_nft: Col::new(vec![0.0; n_links]),
            l_occ: Col::new(vec![0.0; n_links]),
            l_line_head: Col::new(vec![NIL; n_links]),
            l_line_tail: Col::new(vec![NIL; n_links]),
            l_blocked_on: Col::new(vec![0; n_links]),
            s_blocked: Col::new(vec![NONE; 2 * n_links]),
            s_target: Col::new(vec![NIL; 2 * n_links]),
            t_wait: Col::new(vec![nan; n_rides]),
            t_board: Col::new(vec![nan; n_rides]),
            t_alight: Col::new(vec![nan; n_rides]),
            r_trip: Col::new(vec![-1; n_rides]),
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
            tr_boarding: Col::new(vec![NIL; n_trips]),
            tr_pulled: Col::new(vec![false; n_trips]),
            tr_onboard: Col::new((0..n_trips).map(|_| Vec::new()).collect()),
            tr_alight_q: Col::new((0..n_trips).map(|_| VecDeque::new()).collect()),
            stop_arr: Col::new(vec![nan; n_trip_stops]),
            stop_dep: Col::new(vec![nan; n_trip_stops]),
        }
    }
}

/// Per-second buckets of agenda entries, as in HermesSim's kernel.
#[derive(Default)]
struct Agenda {
    dense: Vec<Vec<u64>>,
    pending: usize,
    cursor: usize,
    overflow: BTreeMap<i64, Vec<u64>>,
    pool: Vec<Vec<u64>>,
}

impl Agenda {
    fn push(&mut self, t: i64, e: u64) {
        if t < DENSE_LIMIT {
            let i = t as usize;
            if i >= self.dense.len() {
                self.dense.resize_with(i + 1, Vec::new);
            }
            let bucket = &mut self.dense[i];
            if bucket.capacity() == 0 {
                if let Some(v) = self.pool.pop() {
                    *bucket = v;
                }
            }
            bucket.push(e);
            self.pending += 1;
        } else {
            self.overflow.entry(t).or_default().push(e);
        }
    }

    fn next_time(&mut self) -> i64 {
        if self.pending > 0 {
            while self.dense[self.cursor].is_empty() {
                self.cursor += 1;
            }
            return self.cursor as i64;
        }
        self.overflow.keys().next().copied().unwrap_or(IDLE)
    }

    fn take(&mut self, t: i64) -> Vec<u64> {
        if t < DENSE_LIMIT {
            self.cursor = t as usize; // nothing is pending before `t`
            match self.dense.get_mut(t as usize) {
                Some(b) => {
                    let b = std::mem::take(b);
                    self.pending -= b.len();
                    b
                }
                None => self.pool.pop().unwrap_or_default(),
            }
        } else {
            self.overflow.remove(&t).unwrap_or_default()
        }
    }

    fn give_back(&mut self, mut v: Vec<u64>) {
        v.clear();
        self.pool.push(v);
    }
}

/// Messages from one partition to another (or itself), for one step.
#[derive(Default)]
struct Mail {
    enter: Vec<(u32, u32, i64, i64)>, // (link, vehicle, next link or -1, earliest exit)
    exit: Vec<(u32, f64)>,            // a vehicle left this link: its PCE of storage freed
    occupy: Vec<(u32, f64)>,          // a trip appeared at the end of this link: PCE taken
    person: Vec<(i64, u64, u32)>,     // (second, kind, person)
    task: Vec<(i64, u32, u8, u32)>,   // a trip moving to another stop: (second, trip, kind, node)
}

/// A sense-free spinning barrier: cheap enough to pass twice per simulated
/// second.
struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    fn new(n: usize) -> Self {
        SpinBarrier { n, count: AtomicUsize::new(0), generation: AtomicUsize::new(0) }
    }

    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::AcqRel);
            return;
        }
        let mut spins = 0u32;
        while self.generation.load(Ordering::Acquire) == generation {
            if spins < 2000 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// What the partitions share to step in lockstep. Mail and next-second
/// bounds are double-buffered by step parity, so one barrier per step
/// suffices: a partition can be at most one step ahead of another's reads.
struct Ctx {
    barrier: SpinBarrier,
    mail: [Vec<Mutex<Mail>>; 2], // [parity][src * n_parts + dst]
    next: [Vec<AtomicI64>; 2],   // [parity][partition]: a lower bound on its next busy second
}

/// A trip task's key: (second, node, late, trip, seq), in run order.
type TaskKey = (i64, u32, u8, u32, u64);

struct Part<'a> {
    me: usize,
    inp: &'a Inputs,
    sh: &'a Shared,
    rec: [bool; N_EVENT_KINDS],
    finite_stuck: bool,
    now: i64,
    agenda: Agenda,
    tasks: BTreeMap<TaskKey, (u8, u64)>, // trip tasks of this partition's nodes: (kind, token)
    running: i64,                        // the node whose step is running (-1: the person phase)
    out: Vec<Mail>,                      // per destination partition
    heap: BinaryHeap<Reverse<(i64, u64, u32)>>, // the running node step's sources
    wake_now: Vec<u32>, // nodes woken by the person phase of this step
    key: u64,           // event order key of the running person or node step
    events: Vec<Event>, // (second, order key, code)
}

impl<'a> Part<'a> {
    fn new(me: usize, inp: &'a Inputs, sh: &'a Shared) -> Self {
        let mut rec = [false; N_EVENT_KINDS];
        for (k, r) in rec.iter_mut().enumerate() {
            *r = inp.event_mask & (1 << k) != 0;
        }
        Part {
            me,
            inp,
            sh,
            rec,
            finite_stuck: inp.stuck_time.is_finite(),
            now: 0,
            agenda: Agenda::default(),
            tasks: BTreeMap::new(),
            running: -1,
            out: (0..inp.n_parts).map(|_| Mail::default()).collect(),
            heap: BinaryHeap::new(),
            wake_now: Vec::new(),
            key: 0,
            events: Vec::new(),
        }
    }

    #[inline]
    fn record(&mut self, kind: u32, p: usize, r: usize) {
        let code = ((kind as i64) << KIND_SHIFT) | ((p as i64) << PERSON_SHIFT) | r as i64;
        self.events.push((self.now, self.key, code));
    }

    #[inline]
    fn wake(&mut self, t: i64, node: u32) {
        debug_assert_eq!(self.inp.part_of(node), self.me);
        if t == self.now {
            // only the person phase wakes a node for the running second
            debug_assert!(self.running < 0);
            self.wake_now.push(node);
        } else {
            self.agenda.push(t, NODE_TAG | node as u64);
        }
    }

    #[inline]
    fn send_person(&mut self, t: i64, kind: u64, p: usize, node: u32) {
        let dst = self.inp.part_of(node);
        self.out[dst].person.push((t, kind, p as u32));
    }

    /// A task of trip `d` at `node` (where the trip is), at second `t`.
    unsafe fn task(&mut self, t: i64, late: u8, d: usize, kind: u8, node: u32, token: u64) {
        let seq = self.sh.tr_seq.get(d) + 1;
        self.sh.tr_seq.set(d, seq);
        self.tasks.insert((t, node, late, d as u32, seq), (kind, token));
        if !(t == self.now && self.running == node as i64) {
            // else the running node step takes it
            self.wake(t, node);
        }
    }

    /// Persons whose first activity starts, and trips that start, on this
    /// partition.
    unsafe fn init(&mut self) {
        let inp = self.inp;
        for p in 0..inp.n_plans() {
            let a = self.sh.a_act.get(p);
            if inp.part_of(inp.act_owner[a]) != self.me {
                continue;
            }
            let last = inp.act_offsets[p + 1] as usize - 1;
            if a == last {
                self.sh.a_done.set(p, true);
                continue;
            }
            if let Some(t) = inp.planned_end(a, 0) {
                self.agenda.push(t, (p as u64) << KIND_BITS | ACT_END);
            }
        }
        for d in 0..inp.n_trips() {
            if inp.part_of(inp.trip_owner(d)) == self.me {
                let v = (inp.n_plans() + d) as u64;
                self.agenda.push(inp.trip_start[d].max(0), v << KIND_BITS | TRIP_START);
            }
        }
    }

    // -- intrusive queues (a vehicle is in at most one queue or line) -------
    #[inline]
    unsafe fn queue_push(&mut self, link: usize, v: u32, after: i64, earliest: i64) -> bool {
        let sh = self.sh;
        sh.q_next.set(v as usize, NIL);
        sh.q_after.set(v as usize, after);
        sh.q_earliest.set(v as usize, earliest);
        let tail = sh.l_tail.get(link);
        sh.l_tail.set(link, v);
        if tail == NIL {
            sh.l_head.set(link, v);
            true
        } else {
            sh.q_next.set(tail as usize, v);
            false
        }
    }

    /// Put `v` at the head of `link`'s queue.
    unsafe fn queue_push_front(&mut self, link: usize, v: u32, after: i64, earliest: i64) {
        let sh = self.sh;
        let head = sh.l_head.get(link);
        sh.q_next.set(v as usize, head);
        sh.q_after.set(v as usize, after);
        sh.q_earliest.set(v as usize, earliest);
        sh.l_head.set(link, v);
        if head == NIL {
            sh.l_tail.set(link, v);
        }
    }

    #[inline]
    unsafe fn queue_pop(&mut self, link: usize) {
        let sh = self.sh;
        let next = sh.q_next.get(sh.l_head.get(link) as usize);
        sh.l_head.set(link, next);
        if next == NIL {
            sh.l_tail.set(link, NIL);
        }
    }

    /// Append `p` to the line of `link`; true if it is now the front.
    #[inline]
    unsafe fn line_push(&mut self, link: usize, p: u32) -> bool {
        let sh = self.sh;
        sh.q_next.set(p as usize, NIL);
        let tail = sh.l_line_tail.get(link);
        sh.l_line_tail.set(link, p);
        if tail == NIL {
            sh.l_line_head.set(link, p);
            true
        } else {
            sh.q_next.set(tail as usize, p);
            false
        }
    }

    #[inline]
    unsafe fn line_pop(&mut self, link: usize) {
        let sh = self.sh;
        let next = sh.q_next.get(sh.l_line_head.get(link) as usize);
        sh.l_line_head.set(link, next);
        if next == NIL {
            sh.l_line_tail.set(link, NIL);
        }
    }

    /// The link after vehicle `v`'s current one, or -1.
    #[inline]
    unsafe fn after_of(&self, v: usize) -> i64 {
        let rpos = self.sh.a_rpos.get(v);
        if rpos + 1 < self.sh.a_rend.get(v) {
            self.inp.route_links[rpos + 1] as i64
        } else {
            -1
        }
    }

    // -- persons --------------------------------------------------------------
    /// `node`: the node the person is at, or None for the node running its
    /// activity (in the person phase).
    unsafe fn end_activity(&mut self, p: usize, node: Option<u32>) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        let a = sh.a_act.get(p);
        sh.act_end.set(a, now as f64);
        if self.rec[EV_ACT_END as usize] {
            self.record(EV_ACT_END, p, a);
        }
        let leg = a - p;
        sh.a_leg.set(p, leg as i64);
        sh.dep.set(leg, now as f64);
        if self.rec[EV_DEPARTURE as usize] {
            self.record(EV_DEPARTURE, p, leg);
        }
        if inp.leg_network[leg] != 0 {
            let r0 = inp.route_offsets[leg] as usize;
            let r1 = inp.route_offsets[leg + 1] as usize;
            if r0 == r1 {
                self.arrive(p, node);
                return;
            }
            sh.a_rpos.set(p, r0);
            sh.a_rend.set(p, r1);
            let link = inp.route_links[r0] as usize;
            let front = self.line_push(link, p as u32);
            if self.running < 0 {
                self.wake_now.push(inp.link_from[link]);
            } else if front {
                // a new source for the running node step
                self.push_source(now, inp.n_links() + link);
            }
        } else if inp.leg_transit[leg] != 0 {
            let here = node.unwrap_or(inp.act_owner[a]);
            let r0 = inp.ride_offsets[leg] as usize;
            let r1 = inp.ride_offsets[leg + 1] as usize;
            if r0 == r1 {
                // walk only
                self.walk_to_destination(p, here);
                return;
            }
            sh.a_ride.set(p, r0);
            sh.a_ride_end.set(p, r1);
            self.walk_to_stop(p, here);
        } else {
            let t = now + inp.leg_teleport_steps[leg].max(1);
            self.send_person(t, TELEPORT_ARRIVE, p, inp.act_owner[a + 1]);
        }
    }

    unsafe fn teleport_arrive(&mut self, p: usize) {
        if self.rec[EV_TELEPORT as usize] {
            let leg = self.sh.a_leg.get(p) as usize;
            self.record(EV_TELEPORT, p, leg);
        }
        self.arrive(p, None);
    }

    unsafe fn arrive(&mut self, p: usize, node: Option<u32>) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        let leg = sh.a_leg.get(p) as usize;
        sh.arr.set(leg, now as f64);
        if self.rec[EV_ARRIVAL as usize] {
            self.record(EV_ARRIVAL, p, leg);
        }
        sh.a_leg.set(p, -1);
        sh.a_link.set(p, -1);
        if inp.junction_delay != 0 {
            // the activity starts one second later, where it is run
            let a = sh.a_act.get(p) + 1;
            self.send_person(now + 1, ACT_START, p, inp.act_owner[a]);
            return;
        }
        self.start_activity(p, node);
    }

    unsafe fn start_activity(&mut self, p: usize, node: Option<u32>) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        let a = sh.a_act.get(p) + 1;
        sh.a_act.set(p, a);
        sh.act_start.set(a, now as f64);
        if self.rec[EV_ACT_START as usize] {
            self.record(EV_ACT_START, p, a);
        }
        if a == inp.act_offsets[p + 1] as usize - 1 {
            sh.a_done.set(p, true);
            return;
        }
        let Some(t) = inp.planned_end(a, now) else { return };
        // ends now if it may (late arrival) and departs from here
        if t <= now && node.is_none_or(|n| inp.act_owner[a] == n) {
            self.end_activity(p, node);
        } else {
            self.send_person(t.max(now + 1), ACT_END, p, inp.act_owner[a]);
        }
    }

    // -- passengers -----------------------------------------------------------
    // `here`: the node the person is at. Moves to another node, or into a
    // later second, are messages; a walk of 0 s on the spot happens at once.
    unsafe fn walk_to_stop(&mut self, p: usize, here: u32) {
        let inp = self.inp;
        let r = self.sh.a_ride.get(p);
        let t = self.now + inp.walk_before[r];
        let node = inp.rs_node[inp.ride_board[r] as usize];
        if t == self.now && node == here {
            self.pax_at_stop(p);
        } else {
            self.send_person(t.max(self.now + 1), PAX_AT_STOP, p, node);
        }
    }

    unsafe fn walk_to_destination(&mut self, p: usize, here: u32) {
        let inp = self.inp;
        let leg = self.sh.a_leg.get(p) as usize;
        let t = self.now + inp.walk_after[leg];
        if t == self.now {
            self.arrive(p, Some(here));
        } else {
            let a = self.sh.a_act.get(p) + 1;
            self.send_person(t, PAX_ARRIVE, p, inp.act_owner[a]);
        }
    }

    unsafe fn pax_at_stop(&mut self, p: usize) {
        let (inp, sh) = (self.inp, self.sh);
        let r = sh.a_ride.get(p);
        let k = inp.ride_board[r] as usize;
        sh.t_wait.set(r, self.now as f64);
        if self.rec[EV_WAITING as usize] {
            self.record(EV_WAITING, p, k);
        }
        let n_trips_here = sh.at_stop.at(k).len();
        if inp.rs_network[k] == 0 {
            for i in 0..n_trips_here {
                let d = sh.at_stop.at(k)[i] as usize;
                if (sh.tr_load.get(d) as f64) < inp.trip_capacity[d] {
                    sh.tr_load.set(d, sh.tr_load.get(d) + 1);
                    self.board(d, p);
                    return;
                }
            }
            sh.waiting.at(k).push_back(p as u32);
            return;
        }
        sh.waiting.at(k).push_back(p as u32);
        for i in 0..n_trips_here {
            let d = sh.at_stop.at(k)[i] as usize;
            if self.serve_new_boarder(d) {
                return;
            }
        }
    }

    unsafe fn board(&mut self, d: usize, p: usize) {
        let sh = self.sh;
        let r = sh.a_ride.get(p);
        sh.t_board.set(r, self.now as f64);
        sh.r_trip.set(r, d as i64);
        sh.tr_onboard.at(d).push(p as u32);
        if self.rec[EV_ENTER_VEH as usize] {
            self.record(EV_ENTER_VEH, p, d);
        }
    }

    unsafe fn alight(&mut self, d: usize, p: usize, node: u32) {
        let sh = self.sh;
        let r = sh.a_ride.get(p);
        sh.t_alight.set(r, self.now as f64);
        let onboard = sh.tr_onboard.at(d);
        let i = onboard.iter().position(|&x| x as usize == p).expect("passenger on board");
        onboard.remove(i);
        sh.tr_load.set(d, sh.tr_load.get(d) - 1);
        if self.rec[EV_LEAVE_VEH as usize] {
            self.record(EV_LEAVE_VEH, p, d);
        }
        if r + 1 < sh.a_ride_end.get(p) {
            sh.a_ride.set(p, r + 1);
            self.walk_to_stop(p, node);
        } else {
            self.walk_to_destination(p, node);
        }
    }

    unsafe fn alighters(&self, d: usize, k: usize) -> Vec<u32> {
        let (inp, sh) = (self.inp, self.sh);
        sh.tr_onboard
            .at(d)
            .iter()
            .copied()
            .filter(|&p| inp.ride_alight[sh.a_ride.get(p as usize)] as usize == k)
            .collect()
    }

    // -- trips ----------------------------------------------------------------
    unsafe fn trip_start(&mut self, d: usize) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        if self.rec[EV_DRIVER_START as usize] {
            self.record(EV_DRIVER_START, d, 0);
        }
        if inp.trip_network[d] == 0 {
            let k = inp.trip_k0[d];
            let t = now.max(inp.trip_stop_arr[inp.trip_stop_offsets[d]]);
            self.task(t, 0, d, STOP_ARRIVE, inp.rs_node[k], 0);
            return;
        }
        // appears at the end of its first link, taking its storage from the
        // next second (a message to the link's from-node)
        let v = inp.n_plans() + d;
        let rpos = inp.tr_rstart[d];
        sh.a_rpos.set(v, rpos);
        sh.a_rend.set(v, rpos + inp.trip_link_offsets[d + 1] - inp.trip_link_offsets[d]);
        let link = inp.route_links[rpos] as usize;
        sh.a_link.set(v, link as i64);
        sh.enter.set(rpos, now as f64);
        if self.rec[EV_VEH_LINK_ENTER as usize] {
            self.record(EV_VEH_LINK_ENTER, d, link);
        }
        let dst = inp.part_of(inp.link_from[link]);
        self.out[dst].occupy.push((link as u32, inp.v_pce[v]));
        let after = self.after_of(v);
        let earliest = now + inp.trip_link_steps[rpos - inp.n_plan_links] + inp.jd(after);
        if self.queue_push(link, v as u32, after, earliest) {
            let (ready, _) = self.candidate(link).expect("the trip is the head");
            self.wake(ready.max(now), inp.link_to[link]);
        }
    }

    unsafe fn has_stop_here(&self, d: usize) -> bool {
        let (inp, sh) = (self.inp, self.sh);
        let k = sh.tr_next.get(d);
        k < inp.trip_k1[d]
            && inp.rs_link_pos[k]
                == (sh.a_rpos.get(inp.n_plans() + d) - inp.tr_rstart[d]) as i64
    }

    #[inline]
    fn trip_stop(&self, d: usize, k: usize) -> usize {
        self.inp.trip_stop_offsets[d] + k - self.inp.trip_k0[d]
    }

    unsafe fn record_arrival(&mut self, d: usize, k: usize) {
        let j = self.trip_stop(d, k);
        self.sh.stop_arr.set(j, self.now as f64);
        if self.rec[EV_VEH_ARRIVE as usize] {
            self.record(EV_VEH_ARRIVE, d, k);
        }
        self.sh.at_stop.at(k).push(d as u32);
    }

    unsafe fn stop_depart(&mut self, d: usize, token: u64, node: u32) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        if token != sh.tr_token.get(d) {
            return;
        }
        let net = inp.trip_network[d] != 0;
        if net && sh.tr_state.get(d) != CLOSED {
            return;
        }
        let k = sh.tr_next.get(d);
        let j = self.trip_stop(d, k);
        sh.stop_dep.set(j, now as f64);
        if self.rec[EV_VEH_DEPART as usize] {
            self.record(EV_VEH_DEPART, d, k);
        }
        let trips = sh.at_stop.at(k);
        let i = trips.iter().position(|&x| x as usize == d).expect("trip at stop");
        trips.remove(i);
        sh.tr_next.set(d, k + 1);
        sh.tr_visit.set(d, -1);
        sh.tr_state.set(d, 0);
        if !net {
            if k + 1 < inp.trip_k1[d] {
                // on to the next stop: here, or a message
                let t = now.max(inp.trip_stop_arr[j + 1]);
                let there = inp.rs_node[k + 1];
                if there == node {
                    self.task(t, 0, d, STOP_ARRIVE, node, 0);
                } else {
                    let dst = inp.part_of(there);
                    self.out[dst].task.push((t.max(now + 1), d as u32, STOP_ARRIVE, there));
                }
            }
            return;
        }
        if self.has_stop_here(d) {
            // several stops on one link
            self.reach_stop(d, node);
            return;
        }
        let v = inp.n_plans() + d;
        let link = inp.route_links[sh.a_rpos.get(v)] as usize;
        let after = self.after_of(v);
        // it leaves the link from now, crossing a junction one second later
        let earliest = now + inp.jd(after);
        if sh.tr_pulled.get(d) {
            // back from the stop into the lane, as its head
            sh.tr_pulled.set(d, false);
            self.queue_push_front(link, v as u32, after, earliest);
            self.unblock(link); // the lane's head changed
        } else {
            sh.q_earliest.set(v, earliest);
        }
    }

    unsafe fn sched_arrive(&mut self, d: usize, node: u32) {
        let (inp, sh) = (self.inp, self.sh);
        let k = sh.tr_next.get(d);
        self.record_arrival(d, k);
        for p in self.alighters(d, k) {
            self.alight(d, p as usize, node);
        }
        while !sh.waiting.at(k).is_empty() && (sh.tr_load.get(d) as f64) < inp.trip_capacity[d] {
            sh.tr_load.set(d, sh.tr_load.get(d) + 1);
            let p = sh.waiting.at(k).pop_front().unwrap() as usize;
            self.board(d, p);
        }
        let token = sh.tr_token.get(d) + 1;
        sh.tr_token.set(d, token);
        let t = self.now.max(inp.trip_stop_dep[self.trip_stop(d, k)]);
        self.task(t, 1, d, STOP_DEPART, node, token);
    }

    unsafe fn reach_stop(&mut self, d: usize, node: u32) {
        let sh = self.sh;
        let k = sh.tr_next.get(d);
        sh.tr_visit.set(d, k as i64);
        self.record_arrival(d, k);
        let q: VecDeque<u32> = self.alighters(d, k).into_iter().collect();
        *sh.tr_alight_q.at(d) = q;
        if !sh.tr_alight_q.at(d).is_empty() || self.boarders(d) {
            self.open_doors(d, node);
        } else {
            self.ready(d, node);
        }
    }

    unsafe fn boarders(&self, d: usize) -> bool {
        let sh = self.sh;
        let q = sh.waiting.at(sh.tr_visit.get(d) as usize);
        !q.is_empty()
            && ((sh.tr_load.get(d) - sh.tr_alight_q.at(d).len() as i64) as f64)
                < self.inp.trip_capacity[d]
    }

    unsafe fn open_doors(&mut self, d: usize, node: u32) {
        let sh = self.sh;
        sh.tr_state.set(d, OPENING);
        let token = sh.tr_token.get(d) + 1;
        sh.tr_token.set(d, token);
        self.task(self.now + self.inp.door_open, 0, d, DOOR_OPEN, node, token);
    }

    unsafe fn door_open(&mut self, d: usize, token: u64, node: u32) {
        let sh = self.sh;
        if token != sh.tr_token.get(d) || sh.tr_state.get(d) != OPENING {
            return;
        }
        sh.tr_state.set(d, OPEN);
        self.next_alight(d, node);
        if self.inp.trip_parallel[d] != 0 || !sh.tr_alight_busy.get(d) {
            self.next_board(d, node);
        }
        self.maybe_close(d, node);
    }

    unsafe fn next_alight(&mut self, d: usize, node: u32) {
        let busy = !self.sh.tr_alight_q.at(d).is_empty();
        self.sh.tr_alight_busy.set(d, busy);
        if busy {
            self.task(self.now + self.inp.trip_egress[d], 0, d, ALIGHT_DONE, node, 0);
        }
    }

    unsafe fn alight_done(&mut self, d: usize, node: u32) {
        let sh = self.sh;
        let p = sh.tr_alight_q.at(d).pop_front().expect("an alighting passenger") as usize;
        self.alight(d, p, node);
        self.next_alight(d, node);
        if self.inp.trip_parallel[d] == 0 && !sh.tr_alight_busy.get(d) && !sh.tr_board_busy.get(d)
        {
            self.next_board(d, node);
        }
        self.maybe_close(d, node);
    }

    unsafe fn next_board(&mut self, d: usize, node: u32) {
        let sh = self.sh;
        if self.boarders(d) {
            let k = sh.tr_visit.get(d) as usize;
            let p = sh.waiting.at(k).pop_front().unwrap();
            sh.tr_load.set(d, sh.tr_load.get(d) + 1);
            sh.tr_board_busy.set(d, true);
            sh.tr_boarding.set(d, p);
            self.task(self.now + self.inp.trip_access[d], 0, d, BOARD_DONE, node, 0);
        } else {
            sh.tr_board_busy.set(d, false);
        }
    }

    unsafe fn board_done(&mut self, d: usize, node: u32) {
        let sh = self.sh;
        let p = sh.tr_boarding.get(d) as usize;
        sh.tr_boarding.set(d, NIL);
        self.board(d, p);
        self.next_board(d, node);
        self.maybe_close(d, node);
    }

    unsafe fn maybe_close(&mut self, d: usize, node: u32) {
        let sh = self.sh;
        if sh.tr_state.get(d) == OPEN && !sh.tr_alight_busy.get(d) && !sh.tr_board_busy.get(d) {
            sh.tr_state.set(d, CLOSING);
            self.task(self.now + self.inp.door_close, 0, d, DOOR_CLOSE, node, 0);
        }
    }

    unsafe fn door_close(&mut self, d: usize, node: u32) {
        if self.sh.tr_state.get(d) == CLOSING {
            self.ready(d, node);
        }
    }

    /// Doors closed and nobody to serve: depart (or await the departure).
    unsafe fn ready(&mut self, d: usize, node: u32) {
        let (inp, sh) = (self.inp, self.sh);
        if self.boarders(d) {
            self.open_doors(d, node);
            return;
        }
        sh.tr_state.set(d, CLOSED);
        let k = sh.tr_visit.get(d) as usize;
        let mut t = self.now;
        if inp.rs_await[k] != 0 {
            t = t.max(inp.trip_stop_dep[self.trip_stop(d, k)]);
        }
        let token = sh.tr_token.get(d) + 1;
        sh.tr_token.set(d, token);
        self.task(t, 1, d, STOP_DEPART, node, token);
    }

    /// A passenger joined the line at trip `d`'s stop. True if it takes them.
    unsafe fn serve_new_boarder(&mut self, d: usize) -> bool {
        let (inp, sh) = (self.inp, self.sh);
        if !self.boarders(d) {
            return false;
        }
        let node = inp.rs_node[sh.tr_visit.get(d) as usize];
        match sh.tr_state.get(d) {
            OPEN => {
                if !sh.tr_board_busy.get(d)
                    && (inp.trip_parallel[d] != 0 || !sh.tr_alight_busy.get(d))
                {
                    self.next_board(d, node);
                }
                true
            }
            CLOSED => {
                self.open_doors(d, node);
                true
            }
            s => s == OPENING || s == CLOSING,
        }
    }

    /// The range of `node`'s tasks due now.
    fn due_range(&self, node: u32) -> std::ops::RangeInclusive<TaskKey> {
        (self.now, node, 0, 0, 0)..=(self.now, node, u8::MAX, u32::MAX, u64::MAX)
    }

    /// The first of `node`'s tasks due now, if any.
    fn next_task(&self, node: u32) -> Option<(TaskKey, (u8, u64))> {
        self.tasks.range(self.due_range(node)).next().map(|(&k, &v)| (k, v))
    }

    fn tasks_due(&self, node: u32) -> bool {
        self.tasks.range(self.due_range(node)).next().is_some()
    }

    /// The node's trip tasks due now, in order (ordinary, then late; by trip).
    unsafe fn run_tasks(&mut self, node: u32) {
        while let Some((key, (kind, token))) = self.next_task(node) {
            self.tasks.remove(&key);
            let d = key.3 as usize;
            match kind {
                STOP_ARRIVE => self.sched_arrive(d, node),
                STOP_DEPART => self.stop_depart(d, token, node),
                DOOR_OPEN => self.door_open(d, token, node),
                ALIGHT_DONE => self.alight_done(d, node),
                BOARD_DONE => self.board_done(d, node),
                _ => self.door_close(d, node),
            }
        }
    }

    // -- nodes ----------------------------------------------------------------
    /// `(ready, vehicle)` of source `s`'s head, if it has one (and it is not
    /// a trip serving a stop).
    #[inline]
    unsafe fn candidate(&self, s: usize) -> Option<(i64, u32)> {
        let (inp, sh) = (self.inp, self.sh);
        let n_links = inp.n_links();
        if s < n_links {
            let v = sh.l_head.get(s);
            if v == NIL {
                return None;
            }
            let earliest = sh.q_earliest.get(v as usize);
            if v as usize >= inp.n_plans() {
                let d = v as usize - inp.n_plans();
                if sh.tr_visit.get(d) >= 0 {
                    return None;
                }
                if self.has_stop_here(d) {
                    // it serves the stop on reaching the link's end
                    return Some((earliest - inp.jd(sh.q_after.get(v as usize)), v));
                }
            }
            Some((earliest.max(ceil_step(sh.l_nft.get(s))), v))
        } else {
            let v = sh.l_line_head.get(s - n_links);
            if v == NIL {
                return None;
            }
            let leg = sh.a_leg.get(v as usize) as usize;
            Some((sh.dep.get(leg) as i64, v))
        }
    }

    /// Add source `s` to the running node step. A source refused before
    /// ranks by how long it has waited since, others by when they got ready.
    #[inline]
    unsafe fn push_source(&mut self, ready: i64, s: usize) {
        let since = self.sh.s_blocked.get(s);
        let key = if since == NONE { ready } else { since };
        self.heap.push(Reverse((key, tie_hash(self.now, s as u64), s as u32)));
    }

    #[inline]
    unsafe fn unblock(&mut self, s: usize) {
        let sh = self.sh;
        if sh.s_blocked.get(s) != NONE {
            sh.s_blocked.set(s, NONE);
            let target = sh.s_target.get(s) as usize;
            sh.l_blocked_on.set(target, sh.l_blocked_on.get(target) - 1);
        }
    }

    unsafe fn step_node(&mut self, node: u32) {
        self.key = PHASE_NODE | node as u64;
        self.running = node as i64;
        loop {
            self.run_tasks(node);
            self.move_node(node);
            if !self.tasks_due(node) {
                break;
            }
        }
        self.running = -1;
    }

    unsafe fn move_node(&mut self, node: u32) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        let n_links = inp.n_links();
        let n_plans = inp.n_plans();
        self.heap.clear();
        let n = node as usize;
        for &s in &inp.in_links[inp.in_off[n]..inp.in_off[n + 1]] {
            if let Some((ready, _)) = self.candidate(s as usize) {
                if ready <= now {
                    self.push_source(ready, s as usize);
                }
            }
        }
        for &link in &inp.out_links[inp.out_off[n]..inp.out_off[n + 1]] {
            let s = n_links + link as usize;
            if let Some((ready, _)) = self.candidate(s) {
                self.push_source(ready, s);
            }
        }
        while let Some(Reverse((_, _, s))) = self.heap.pop() {
            let s = s as usize;
            let (ready, v) = self.candidate(s).expect("a source in the heap has a head");
            let vi = v as usize;
            let (nxt, earliest) = if s < n_links {
                if vi >= n_plans && self.has_stop_here(vi - n_plans) {
                    // serve the stop
                    let d = vi - n_plans;
                    let stop = inp.rs_stop[sh.tr_next.get(d)] as usize;
                    if inp.stop_blocking[stop] == 0 {
                        self.queue_pop(s); // into the stop: off the lane, still in storage
                        sh.tr_pulled.set(d, true);
                    }
                    self.reach_stop(d, node);
                    if let Some((r, _)) = self.candidate(s) {
                        if r <= now {
                            self.push_source(r, s);
                        }
                    }
                    continue;
                }
                (sh.q_after.get(vi), sh.q_earliest.get(vi))
            } else {
                // a departure: onto the first link of its route
                (inp.route_links[sh.a_rpos.get(vi)] as i64, 0)
            };
            let pce = inp.v_pce[vi];
            if nxt >= 0 {
                // needs space on the next link
                let nx = nxt as usize;
                let since = sh.s_blocked.get(s);
                let forced = self.finite_stuck
                    && since != NONE
                    && now >= ceil_step(since as f64 + inp.stuck_time);
                let has_space = sh.l_occ.get(nx) < inp.storage[nx];
                if !forced && !has_space {
                    if since == NONE {
                        sh.s_blocked.set(s, now);
                        sh.s_target.set(s, nx as u32);
                        sh.l_blocked_on.set(nx, sh.l_blocked_on.get(nx) + 1);
                        if self.finite_stuck {
                            let t = ceil_step(now as f64 + inp.stuck_time).max(now + 1);
                            self.wake(t, node);
                        }
                    }
                    continue; // FIFO: nothing behind it moves this step
                }
                self.unblock(s);
            }
            if s < n_links {
                // leave link s
                self.queue_pop(s);
                let nft = sh.l_nft.get(s);
                let base = if now == ready {
                    let e = earliest as f64;
                    if e > nft { e } else { nft }
                } else {
                    now as f64
                };
                sh.l_nft.set(s, base + inp.headway[s] * pce);
                if vi < n_plans {
                    if self.rec[EV_LINK_LEAVE as usize] {
                        self.record(EV_LINK_LEAVE, vi, s);
                    }
                } else if self.rec[EV_VEH_LINK_LEAVE as usize] {
                    self.record(EV_VEH_LINK_LEAVE, vi - n_plans, s);
                }
                let dst = inp.part_of(inp.link_from[s]);
                self.out[dst].exit.push((s as u32, pce));
                if nxt < 0 {
                    if vi < n_plans {
                        self.arrive(vi, Some(node));
                    } else {
                        sh.a_link.set(vi, -1); // a trip leaves the network
                    }
                } else {
                    sh.a_rpos.set(vi, sh.a_rpos.get(vi) + 1);
                }
            } else {
                self.line_pop(s - n_links);
            }
            if nxt >= 0 {
                // enter link nxt
                let nx = nxt as usize;
                sh.l_occ.set(nx, sh.l_occ.get(nx) + pce);
                let rpos = sh.a_rpos.get(vi);
                sh.enter.set(rpos, now as f64);
                sh.a_link.set(vi, nxt);
                let after = self.after_of(vi);
                let steps = if vi < n_plans {
                    if self.rec[EV_LINK_ENTER as usize] {
                        self.record(EV_LINK_ENTER, vi, nx);
                    }
                    inp.free_steps[nx].max(1)
                } else {
                    if self.rec[EV_VEH_LINK_ENTER as usize] {
                        self.record(EV_VEH_LINK_ENTER, vi - n_plans, nx);
                    }
                    inp.trip_link_steps[rpos - inp.n_plan_links].max(1)
                };
                // a junction delay when it crosses into a next link
                let earliest = now + steps + inp.jd(after);
                let dst = inp.part_of(inp.link_to[nx]);
                self.out[dst].enter.push((nx as u32, v, after, earliest));
            }
            if let Some((r, _)) = self.candidate(s) {
                if r <= now {
                    self.push_source(r, s);
                }
            }
        }
        // wake for heads that become ready later
        for &s in &inp.in_links[inp.in_off[n]..inp.in_off[n + 1]] {
            let s = s as usize;
            if sh.s_blocked.get(s) == NONE {
                if let Some((r, _)) = self.candidate(s) {
                    self.wake(r.max(now + 1), node);
                }
            }
        }
    }

    /// One second: persons (and trip starts), then nodes. Effects elsewhere
    /// go to `out`. Returns whether anything was due.
    unsafe fn step(&mut self, t: i64) -> bool {
        self.now = t;
        let n_plans = self.inp.n_plans();
        let bucket = self.agenda.take(t);
        let mut persons: Vec<u64> = Vec::new();
        self.wake_now.clear();
        for &e in &bucket {
            if e & NODE_TAG != 0 {
                self.wake_now.push((e & !NODE_TAG) as u32);
            } else {
                persons.push(e);
            }
        }
        self.agenda.give_back(bucket);
        persons.sort_unstable();
        for &e in &persons {
            let p = (e >> KIND_BITS) as usize;
            self.key = p as u64;
            match e & ((1 << KIND_BITS) - 1) {
                ACT_END => self.end_activity(p, None),
                TELEPORT_ARRIVE => self.teleport_arrive(p),
                ACT_START => self.start_activity(p, None),
                PAX_AT_STOP => self.pax_at_stop(p),
                PAX_ARRIVE => self.arrive(p, None),
                _ => self.trip_start(p - n_plans),
            }
        }
        let mut nodes = std::mem::take(&mut self.wake_now);
        nodes.sort_unstable();
        nodes.dedup();
        for &node in &nodes {
            self.step_node(node);
        }
        let busy = !persons.is_empty() || !nodes.is_empty();
        self.wake_now = nodes;
        busy
    }

    /// Apply the messages of one source partition, at the end of the step.
    unsafe fn deliver(&mut self, mail: &mut Mail) {
        let (inp, sh, now) = (self.inp, self.sh, self.now);
        for &(link, v, after, earliest) in &mail.enter {
            let l = link as usize;
            if self.queue_push(l, v, after, earliest) {
                let (ready, _) = self.candidate(l).expect("the vehicle is the head");
                self.wake(ready.max(now + 1), inp.link_to[l]);
            }
        }
        for &(link, pce) in &mail.exit {
            let l = link as usize;
            sh.l_occ.set(l, sh.l_occ.get(l) - pce);
            if sh.l_blocked_on.get(l) > 0 {
                self.wake(now + 1, inp.link_from[l]);
            }
        }
        for &(link, pce) in &mail.occupy {
            let l = link as usize;
            sh.l_occ.set(l, sh.l_occ.get(l) + pce);
        }
        for &(t, kind, p) in &mail.person {
            self.agenda.push(t, (p as u64) << KIND_BITS | kind);
        }
        for &(t, d, kind, node) in &mail.task {
            self.task(t, 0, d as usize, kind, node, 0);
        }
        mail.enter.clear();
        mail.exit.clear();
        mail.occupy.clear();
        mail.person.clear();
        mail.task.clear();
    }

    /// The earliest second the messages in `out` can make anything happen:
    /// a lower bound, which is all the step loop needs (a second in which
    /// nothing is due runs empty and changes nothing).
    fn mail_bound(&self) -> i64 {
        let mut t = IDLE;
        for m in &self.out {
            if !m.exit.is_empty() {
                t = t.min(self.now + 1);
            }
            for e in &m.enter {
                t = t.min(e.3);
            }
            for e in &m.person {
                t = t.min(e.0);
            }
            for e in &m.task {
                t = t.min(e.0);
            }
        }
        t
    }

    /// The partition's thread: steps in lockstep with the others. Returns
    /// its events and the last second in which it had something to do.
    unsafe fn run(mut self, ctx: &Ctx) -> (Vec<Event>, i64) {
        let k = self.inp.n_parts;
        let me = self.me;
        self.init();
        let mut t = self.agenda.next_time();
        if k > 1 {
            ctx.next[0][me].store(t, Ordering::Release);
            ctx.barrier.wait();
            t = ctx.next[0].iter().map(|x| x.load(Ordering::Acquire)).min().unwrap_or(IDLE);
        }
        let mut last = 0;
        let mut parity = 1;
        while t != IDLE && (t as f64) <= self.inp.end_time {
            if self.step(t) {
                last = t;
            }
            if k == 1 {
                let mut mail = std::mem::take(&mut self.out[0]);
                self.deliver(&mut mail);
                self.out[0] = mail;
                t = self.agenda.next_time();
                continue;
            }
            // publish this step's mail and a bound on everyone's next second
            let bound = self.agenda.next_time().min(self.mail_bound());
            for dst in 0..k {
                let mut slot = ctx.mail[parity][me * k + dst].lock().unwrap();
                std::mem::swap(&mut *slot, &mut self.out[dst]);
            }
            ctx.next[parity][me].store(bound, Ordering::Release);
            ctx.barrier.wait();
            for src in 0..k {
                let mut slot = ctx.mail[parity][src * k + me].lock().unwrap();
                self.deliver(&mut slot);
            }
            t = ctx.next[parity].iter().map(|x| x.load(Ordering::Acquire)).min().unwrap_or(IDLE);
            parity ^= 1;
        }
        (self.events, last)
    }
}

/// A recorded event: (second, order key, packed code).
type Event = (i64, u64, i64);

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

fn simulate(inp: &Inputs) -> Outputs {
    let sh = Shared::new(inp);
    let k = inp.n_parts;
    let ctx = Ctx {
        barrier: SpinBarrier::new(k),
        mail: [0, 1].map(|_| (0..k * k).map(|_| Mutex::new(Mail::default())).collect()),
        next: [0, 1].map(|_| (0..k).map(|_| AtomicI64::new(IDLE)).collect()),
    };
    let results: Vec<(Vec<Event>, i64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (1..k)
            .map(|me| {
                let (sh, ctx) = (&sh, &ctx);
                scope.spawn(move || unsafe { Part::new(me, inp, sh).run(ctx) })
            })
            .collect();
        let mut out = vec![unsafe { Part::new(0, inp, &sh).run(&ctx) }];
        out.extend(handles.into_iter().map(|h| h.join().expect("partition thread panicked")));
        out
    });
    let last = results.iter().map(|r| r.1).max().unwrap_or(0);
    let mut events: Vec<Event> = Vec::with_capacity(results.iter().map(|r| r.0.len()).sum());
    for (ev, _) in results {
        events.extend(ev);
    }
    // canonical order: by second, then persons by index, then nodes by index;
    // one owner's events are all on one partition, in order (stable sort)
    events.sort_by_key(|&(t, key, _)| (t, key));

    let mut last_time = last as f64;
    if inp.end_time.is_finite() {
        last_time = last_time.max(inp.end_time);
    }
    let n_plans = inp.n_plans();
    let a_done = sh.a_done.into_inner();
    let a_leg = sh.a_leg.into_inner();
    let a_link = sh.a_link.into_inner();
    let stuck: Vec<bool> = a_done.iter().map(|&d| !d).collect();
    let stuck_leg = (0..n_plans).filter(|&p| stuck[p]).map(|p| a_leg[p]).collect();
    let stuck_link = (0..n_plans).filter(|&p| stuck[p]).map(|p| a_link[p]).collect();
    Outputs {
        act_start: sh.act_start.into_inner(),
        act_end: sh.act_end.into_inner(),
        dep: sh.dep.into_inner(),
        arr: sh.arr.into_inner(),
        enter: sh.enter.into_inner(),
        stuck,
        ev_time: events.iter().map(|e| e.0 as f64).collect(),
        ev_code: events.iter().map(|e| e.2).collect(),
        stuck_leg,
        stuck_link,
        last_time,
        ride_wait: sh.t_wait.into_inner(),
        ride_board: sh.t_board.into_inner(),
        ride_alight: sh.t_alight.into_inner(),
        ride_trip: sh.r_trip.into_inner(),
        stop_arr: sh.stop_arr.into_inner(),
        stop_dep: sh.stop_dep.into_inner(),
    }
}

fn err(msg: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.to_string())
}

/// Run the PSim kernel on `n_parts` threads. Same arguments and
/// results as `mobslim.mobsim.psim.run_kernel`, plus the partition of
/// each node (`node_part`, in `0 .. n_parts`).
#[pyfunction]
#[pyo3(signature = (*, free_steps, headway, storage, act_offsets, act_end_time, act_duration,
                    leg_network, leg_teleport_steps, route_offsets, route_links, stuck_time,
                    end_time, event_mask, link_from, link_to, n_nodes, act_owner, node_part,
                    n_parts, network_vehicles, leg_transit, ride_offsets, ride_board,
                    ride_alight, walk_before_steps, walk_after_steps, rs_stop, rs_link_pos,
                    rs_await, stop_blocking, trip_k0, trip_k1, trip_start_step,
                    trip_stop_offsets, trip_stop_arr_step, trip_stop_dep_step,
                    trip_link_offsets, trip_links, trip_link_steps, trip_capacity, trip_pce,
                    trip_access_steps, trip_egress_steps, trip_parallel, door_open_steps,
                    door_close_steps, trip_network, rs_network, rs_node, junction_delay=0))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    free_steps: PyReadonlyArray1<'py, i64>,
    headway: PyReadonlyArray1<'py, f64>,
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
    rs_node: PyReadonlyArray1<'py, i64>,
    junction_delay: i64,
) -> PyResult<Bound<'py, PyTuple>> {
    let n_links = free_steps.as_array().len();
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
    if link_from.len() != n_links || link_to.len() != n_links || node_part.len() != n_nodes {
        return Err(err("link and node arrays must match the network"));
    }
    if act_owner.len() != act_end_time.as_array().len() {
        return Err(err("act_owner must have one entry per activity"));
    }
    // transit
    let n_rs = rs_stop.as_array().len();
    let rs_node = idx(&rs_node, n_nodes, "route stop node")?;
    let ride_board = idx(&ride_board, n_rs.max(1), "board stop")?;
    let ride_alight = idx(&ride_alight, n_rs.max(1), "alight stop")?;
    let trip_links = idx(&trip_links, n_links, "trip link")?;
    let trip_k0 = usizes(&trip_k0);
    let trip_k1 = usizes(&trip_k1);
    let trip_stop_offsets = usizes(&trip_stop_offsets);
    let trip_link_offsets = usizes(&trip_link_offsets);
    if rs_node.len() != n_rs
        || trip_k1.len() != n_trips
        || trip_stop_offsets.len() != n_trips + 1
        || trip_link_offsets.len() != n_trips + 1
        || trip_k0.iter().zip(&trip_k1).any(|(&a, &b)| a > b || b > n_rs)
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
    let trip_pce = trip_pce.as_array().to_vec();
    let mut v_pce = vec![1.0; n_plans];
    v_pce.extend_from_slice(&trip_pce);
    let (in_off, in_links) = csr(n_nodes, &link_to);
    let (out_off, out_links) = csr(n_nodes, &link_from);
    let inp = Inputs {
        free_steps: free_steps.as_array().to_vec(),
        headway: headway.as_array().to_vec(),
        storage: storage.as_array().to_vec(),
        act_offsets,
        act_end_time: act_end_time.as_array().to_vec(),
        act_duration: act_duration.as_array().to_vec(),
        leg_network: leg_network.as_array().to_vec(),
        leg_teleport_steps: leg_teleport_steps.as_array().to_vec(),
        route_offsets: route_offsets.as_array().to_vec(),
        route_links,
        stuck_time,
        end_time,
        event_mask,
        link_from,
        link_to,
        act_owner,
        junction_delay,
        node_part,
        n_parts,
        trip_network,
        rs_network: rs_network.as_array().to_vec(),
        leg_transit: leg_transit.as_array().to_vec(),
        ride_offsets: ride_offsets.as_array().to_vec(),
        ride_board,
        ride_alight,
        walk_before: walk_before_steps.as_array().to_vec(),
        walk_after: walk_after_steps.as_array().to_vec(),
        rs_stop: rs_stop.as_array().iter().map(|&x| x as u32).collect(),
        rs_link_pos: rs_link_pos.as_array().to_vec(),
        rs_await: rs_await.as_array().to_vec(),
        stop_blocking: stop_blocking.as_array().to_vec(),
        rs_node,
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
        v_pce,
        tr_rstart,
        in_off,
        in_links,
        out_off,
        out_links,
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
