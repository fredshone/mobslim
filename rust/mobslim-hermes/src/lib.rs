//! Compiled kernel for mobslim's `HermesSim`.
//!
//! A line-for-line port of `run_kernel` in `src/mobslim/mobsim/hermes.py`
//! (the reference, with the model documented there). Both must give
//! identical results; `tests/test_hermes.py` checks that they do.
//!
//! Time is whole seconds. The agenda is a vector of per-second buckets
//! (dense up to `DENSE_LIMIT`, then a `BTreeMap` for far-future entries such
//! as long stuck deadlines). Entries in a bucket run in push order, and
//! entries pushed for the current second are appended to the bucket being
//! drained. "Late" entries (transit stop departures) run after every other
//! entry of their second.
//!
//! Vehicles `0 .. n_plans` are the persons; `n_plans + d` is transit trip `d`.

use std::collections::{BTreeMap, VecDeque};

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};

const EPS: f64 = 1e-9;
const NONE: i64 = i64::MIN;
const DENSE_LIMIT: i64 = 1 << 21; // ~24 days of one-second buckets

// recorded event kinds (bit `1 << kind` of the event mask)
const EV_ACT_START: u32 = 0;
const EV_ACT_END: u32 = 1;
const EV_DEPARTURE: u32 = 2;
const EV_ARRIVAL: u32 = 3;
const EV_LINK_ENTER: u32 = 4;
const EV_LINK_LEAVE: u32 = 5;
const EV_TELEPORT: u32 = 6;
const EV_DRIVER_START: u32 = 8;
const EV_VEH_ARRIVE: u32 = 9;
const EV_VEH_DEPART: u32 = 10;
const EV_WAITING: u32 = 11;
const EV_ENTER_VEH: u32 = 12;
const EV_LEAVE_VEH: u32 = 13;
const EV_VEH_LINK_ENTER: u32 = 14;
const EV_VEH_LINK_LEAVE: u32 = 15;
const N_EVENT_KINDS: usize = 16;
// packed event code: kind << 58 | person << 29 | ref
const KIND_SHIFT: u32 = 58;
const PERSON_SHIFT: u32 = 29;

// agenda entries: kind << 56 | token << 32 | index
const ACT_END: u64 = 0;
const TRY_LEAVE: u64 = 1;
const FORCE_LEAVE: u64 = 2;
const TRY_DEPART: u64 = 3;
const FORCE_DEPART: u64 = 4;
const TELEPORT_ARRIVE: u64 = 5;
const TRIP_START: u64 = 6;
const STOP_ARRIVE: u64 = 7;
const STOP_DEPART: u64 = 8;
const DOOR_OPEN: u64 = 9;
const ALIGHT_DONE: u64 = 10;
const BOARD_DONE: u64 = 11;
const DOOR_CLOSE: u64 = 12;
const PAX_AT_STOP: u64 = 13;
const PAX_ARRIVE: u64 = 14;
const ACT_START: u64 = 15;
const TOKEN_MASK: u64 = (1 << 24) - 1;

// doors of a trip in network mode
const OPENING: u8 = 1;
const OPEN: u8 = 2;
const CLOSING: u8 = 3;
const CLOSED: u8 = 4;

// waiter tokens: links have the top bit set
const LINK_TOKEN: u64 = 1 << 63;
const END: u32 = u32::MAX; // queue entry: the link is the vehicle's last
const NIL: u32 = u32::MAX; // no vehicle

#[inline]
fn ceil_step(t: f64) -> i64 {
    (t - EPS).ceil() as i64
}

#[inline]
fn entry(kind: u64, token: u64, index: usize) -> u64 {
    (kind << 56) | ((token & TOKEN_MASK) << 32) | index as u64
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
    n_plan_links: usize,
    stuck_time: f64,
    end_time: f64,
    event_mask: i64,
    // transit
    network_vehicles: bool,
    leg_transit: Vec<u8>,
    ride_offsets: Vec<i64>,
    ride_board: Vec<i64>,
    ride_alight: Vec<i64>,
    walk_before_steps: Vec<i64>,
    walk_after_steps: Vec<i64>,
    rs_stop: Vec<i64>,
    rs_link_pos: Vec<i64>,
    rs_await: Vec<u8>,
    stop_blocking: Vec<u8>,
    trip_k0: Vec<i64>,
    trip_k1: Vec<i64>,
    trip_start_step: Vec<i64>,
    trip_stop_offsets: Vec<i64>,
    trip_stop_arr_step: Vec<i64>,
    trip_stop_dep_step: Vec<i64>,
    trip_link_offsets: Vec<i64>,
    trip_link_steps: Vec<i64>,
    trip_capacity: Vec<f64>,
    trip_access_steps: Vec<i64>,
    trip_egress_steps: Vec<i64>,
    trip_parallel: Vec<u8>,
    door_open_steps: i64,
    door_close_steps: i64,
    trip_network: Vec<u8>, // per trip: 1 on the network, 0 on the timetable
    rs_network: Vec<u8>,   // per route stop: its route's trips' mode
    v_pce: Vec<f64>, // per vehicle: persons 1, then the trips' PCE
    // 1: a vehicle leaves a link one second later if it crosses a junction
    // after it, and an activity starts one second after the arrival
    junction_delay: i64,
}

#[derive(Default)]
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

/// A link's state and parameters that a vehicle move reads or writes, packed
/// into one cache line: the kernel is bound by memory latency, not compute.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct LinkHot {
    occ: f64,     // storage in use (PCE)
    nft: f64,     // next free flow slot (unrounded)
    headway: f64, // seconds per PCE
    storage: f64,
    blocked: i64, // since when the head is blocked downstream, or NONE
    pending: i64, // the time of the live TRY_LEAVE wake-up, or NONE
    waiting: i32, // the downstream link the head waits for, or -1
    free: i32,    // free-speed time in whole seconds
    nwait: u32,   // tokens in `l_waiters` (checked without touching the deque)
    head: u32,    // first vehicle in the queue, or NIL
}

/// A vehicle's route position, current link and PCE, packed together.
#[derive(Clone, Copy)]
#[repr(C)]
struct VehHot {
    rpos: usize, // position in `route_links` of its current link
    rend: usize, // end of its route in `route_links`
    link: i64,   // current link, or -1
    pce: f64,
    // its entry in its link's queue (a vehicle is in at most one)
    qnext: u32,     // the vehicle behind it, or NIL
    qnx: u32,       // its next link, or END
    qearliest: i64, // its earliest exit
}

struct Sim<'a> {
    lk: Vec<LinkHot>,
    vh: Vec<VehHot>,
    inp: &'a Inputs,
    out: Outputs,
    rec: [bool; N_EVENT_KINDS],
    finite_stuck: bool,
    now: i64,
    n_plans: usize,
    // agenda
    dense: Vec<Vec<u64>>,
    dense_pending: usize,
    cursor: usize,
    overflow: BTreeMap<i64, Vec<u64>>,
    late: BTreeMap<i64, Vec<u64>>,
    cur: Vec<u64>,
    cur_late: Vec<u64>,
    pool: Vec<Vec<u64>>, // drained bucket vectors, reused to avoid reallocating
    // persons
    a_act: Vec<usize>,
    a_last: Vec<usize>,
    a_leg: Vec<i64>,
    a_done: Vec<bool>,
    a_ride: Vec<usize>,
    a_ride_end: Vec<usize>,
    // vehicles (persons, then trips)
    a_blocked: Vec<i64>,
    a_waiting: Vec<i64>,
    // links
    l_tail: Vec<u32>, // last vehicle in each link's queue (the head is in `lk`)
    l_waiters: Vec<VecDeque<u64>>,
    l_departing: Vec<VecDeque<usize>>, // vehicles blocked departing onto the link, FIFO
    // rides
    waiting: Vec<VecDeque<u32>>,
    at_stop: Vec<Vec<u32>>,
    // trips
    tr_rstart: Vec<usize>,
    tr_next: Vec<usize>,
    tr_onboard: Vec<Vec<u32>>,
    tr_load: Vec<i64>,
    tr_visit: Vec<i64>,
    tr_state: Vec<u8>,
    tr_token: Vec<u64>,
    tr_alight_q: Vec<VecDeque<u32>>,
    tr_alight_busy: Vec<bool>,
    tr_board_busy: Vec<bool>,
    tr_boarding: Vec<i64>,
    tr_pulled: Vec<bool>,
}

impl<'a> Sim<'a> {
    fn new(inp: &'a Inputs) -> Self {
        let n_plans = inp.act_offsets.len() - 1;
        let n_links = inp.free_steps.len();
        let n_acts = inp.act_end_time.len();
        let n_legs = inp.leg_network.len();
        let n_trips = inp.trip_k0.len();
        let n_veh = n_plans + n_trips;
        let n_rides = inp.ride_board.len();
        let n_rs = inp.rs_stop.len();
        let n_trip_stops = *inp.trip_stop_offsets.last().unwrap_or(&0) as usize;
        let nan = f64::NAN;
        let mut rec = [false; N_EVENT_KINDS];
        for (k, r) in rec.iter_mut().enumerate() {
            *r = inp.event_mask & (1 << k) != 0;
        }
        // reserve the most events each wanted person kind can produce
        let n_links_entered = inp.n_plan_links;
        let bound = [n_acts, n_acts, n_legs, n_legs, n_links_entered, n_links_entered, n_legs];
        let n_events: usize = bound.iter().zip(rec.iter()).filter(|(_, &r)| r).map(|(b, _)| b).sum();
        let tr_rstart = if inp.network_vehicles {
            inp.trip_link_offsets[..n_trips].iter().map(|&o| inp.n_plan_links + o as usize).collect()
        } else {
            vec![0; n_trips]
        };
        Sim {
            lk: (0..n_links)
                .map(|l| LinkHot {
                    occ: 0.0,
                    nft: 0.0,
                    headway: inp.headway[l],
                    storage: inp.storage[l],
                    blocked: NONE,
                    pending: NONE,
                    waiting: -1,
                    free: inp.free_steps[l] as i32,
                    nwait: 0,
                    head: NIL,
                })
                .collect(),
            vh: (0..n_veh).map(|v| VehHot { rpos: 0, rend: 0, link: -1, pce: inp.v_pce[v], qnext: NIL, qnx: END, qearliest: 0 }).collect(),
            inp,
            out: Outputs {
                act_start: vec![nan; n_acts],
                act_end: vec![nan; n_acts],
                dep: vec![nan; n_legs],
                arr: vec![nan; n_legs],
                enter: vec![nan; inp.route_links.len()],
                stuck: vec![false; n_plans],
                ev_time: Vec::with_capacity(n_events),
                ev_code: Vec::with_capacity(n_events),
                ride_wait: vec![nan; n_rides],
                ride_board: vec![nan; n_rides],
                ride_alight: vec![nan; n_rides],
                ride_trip: vec![-1; n_rides],
                stop_arr: vec![nan; n_trip_stops],
                stop_dep: vec![nan; n_trip_stops],
                ..Default::default()
            },
            rec,
            finite_stuck: inp.stuck_time.is_finite(),
            now: 0,
            n_plans,
            dense: Vec::new(),
            dense_pending: 0,
            cursor: 0,
            overflow: BTreeMap::new(),
            late: BTreeMap::new(),
            cur: Vec::new(),
            cur_late: Vec::new(),
            pool: Vec::new(),
            a_act: inp.act_offsets[..n_plans].iter().map(|&o| o as usize).collect(),
            a_last: inp.act_offsets[1..].iter().map(|&o| o as usize - 1).collect(),
            a_leg: vec![-1; n_plans],
            a_done: vec![false; n_plans],
            a_ride: vec![0; n_plans],
            a_ride_end: vec![0; n_plans],
            a_blocked: vec![NONE; n_veh],
            a_waiting: vec![-1; n_veh],
            l_tail: vec![NIL; n_links],
            l_waiters: (0..n_links).map(|_| VecDeque::new()).collect(),
            l_departing: (0..n_links).map(|_| VecDeque::new()).collect(),
            waiting: (0..n_rs).map(|_| VecDeque::new()).collect(),
            at_stop: (0..n_rs).map(|_| Vec::new()).collect(),
            tr_rstart,
            tr_next: inp.trip_k0.iter().map(|&k| k as usize).collect(),
            tr_onboard: (0..n_trips).map(|_| Vec::new()).collect(),
            tr_load: vec![0; n_trips],
            tr_visit: vec![-1; n_trips],
            tr_state: vec![0; n_trips],
            tr_token: vec![0; n_trips],
            tr_alight_q: (0..n_trips).map(|_| VecDeque::new()).collect(),
            tr_alight_busy: vec![false; n_trips],
            tr_board_busy: vec![false; n_trips],
            tr_boarding: vec![-1; n_trips],
            tr_pulled: vec![false; n_trips],
        }
    }

    #[inline]
    fn record(&mut self, kind: u32, p: usize, r: usize) {
        self.out.ev_time.push(self.now as f64);
        self.out
            .ev_code
            .push(((kind as i64) << KIND_SHIFT) | ((p as i64) << PERSON_SHIFT) | r as i64);
    }

    // -- agenda -------------------------------------------------------------
    #[inline]
    fn schedule(&mut self, t: i64, kind: u64, index: usize) {
        self.push(t, entry(kind, 0, index));
    }

    fn push(&mut self, t: i64, e: u64) {
        debug_assert!(t >= self.now, "scheduled in the past: {t} < {}", self.now);
        if t == self.now {
            self.cur.push(e); // appended to the bucket being drained
        } else if t < DENSE_LIMIT {
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
            self.dense_pending += 1;
        } else {
            self.overflow.entry(t).or_default().push(e);
        }
    }

    /// Like `schedule`, but after every ordinary entry of second `t`.
    fn schedule_late(&mut self, t: i64, kind: u64, token: u64, index: usize) {
        let e = entry(kind, token, index);
        if t == self.now {
            self.cur_late.push(e);
        } else {
            self.late.entry(t).or_default().push(e);
        }
    }

    fn next_time(&mut self) -> Option<i64> {
        let normal = if self.dense_pending > 0 {
            while self.dense[self.cursor].is_empty() {
                self.cursor += 1;
            }
            Some(self.cursor as i64)
        } else {
            self.overflow.keys().next().copied()
        };
        match (normal, self.late.keys().next().copied()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn take_bucket(&mut self, t: i64) -> Vec<u64> {
        if t < DENSE_LIMIT {
            match self.dense.get_mut(t as usize) {
                Some(b) => {
                    let b = std::mem::take(b);
                    self.dense_pending -= b.len();
                    b
                }
                None => self.pool.pop().unwrap_or_default(),
            }
        } else {
            self.overflow.remove(&t).unwrap_or_default()
        }
    }

    fn dispatch(&mut self, e: u64) {
        let index = (e & 0xFFFF_FFFF) as usize;
        let token = (e >> 32) & TOKEN_MASK;
        match e >> 56 {
            ACT_END => self.end_activity(index),
            TRY_LEAVE => self.try_leave_event(index),
            FORCE_LEAVE => self.force_leave(index),
            TRY_DEPART => self.try_depart_event(index),
            FORCE_DEPART => self.force_depart(index),
            TELEPORT_ARRIVE => self.teleport_arrive(index),
            TRIP_START => self.trip_start(index),
            STOP_ARRIVE => self.sched_arrive(index),
            STOP_DEPART => self.stop_depart(index, token),
            DOOR_OPEN => self.door_open(index, token),
            ALIGHT_DONE => self.alight_done(index),
            BOARD_DONE => self.board_done(index),
            DOOR_CLOSE => self.door_close(index),
            PAX_AT_STOP => self.pax_at_stop(index),
            PAX_ARRIVE => self.arrive(index),
            ACT_START => self.start_activity(index),
            _ => unreachable!(),
        }
    }

    fn planned_end(&self, a: usize, start: i64) -> Option<i64> {
        let e = self.inp.act_end_time[a];
        if !e.is_nan() {
            return Some(if (start as f64) > e { start } else { ceil_step(e) });
        }
        let d = self.inp.act_duration[a];
        if !d.is_nan() {
            return Some(ceil_step(start as f64 + d).max(start));
        }
        None
    }

    // -- activities and legs ------------------------------------------------
    fn end_activity(&mut self, p: usize) {
        let now = self.now as f64;
        let a = self.a_act[p];
        self.out.act_end[a] = now;
        if self.rec[EV_ACT_END as usize] {
            self.record(EV_ACT_END, p, a);
        }
        let leg = a - p;
        self.a_leg[p] = leg as i64;
        self.out.dep[leg] = now;
        if self.rec[EV_DEPARTURE as usize] {
            self.record(EV_DEPARTURE, p, leg);
        }
        if self.inp.leg_network[leg] != 0 {
            let r0 = self.inp.route_offsets[leg] as usize;
            let r1 = self.inp.route_offsets[leg + 1] as usize;
            if r0 == r1 {
                self.arrive(p);
                return;
            }
            self.vh[p].rpos = r0;
            self.vh[p].rend = r1;
            self.try_depart(p, false);
        } else if self.inp.leg_transit[leg] != 0 {
            self.pt_depart(p);
        } else {
            let t = self.now + self.inp.leg_teleport_steps[leg];
            self.schedule(t, TELEPORT_ARRIVE, p);
        }
    }

    fn teleport_arrive(&mut self, p: usize) {
        if self.rec[EV_TELEPORT as usize] {
            self.record(EV_TELEPORT, p, self.a_leg[p] as usize);
        }
        self.arrive(p);
    }

    fn arrive(&mut self, p: usize) {
        let now = self.now as f64;
        let leg = self.a_leg[p] as usize;
        self.out.arr[leg] = now;
        if self.rec[EV_ARRIVAL as usize] {
            self.record(EV_ARRIVAL, p, leg);
        }
        self.a_leg[p] = -1;
        if self.inp.junction_delay != 0 {
            // the activity starts one second later
            self.schedule(self.now + 1, ACT_START, p);
        } else {
            self.start_activity(p);
        }
    }

    fn start_activity(&mut self, p: usize) {
        let now = self.now as f64;
        let a = self.a_act[p] + 1;
        self.a_act[p] = a;
        self.out.act_start[a] = now;
        if self.rec[EV_ACT_START as usize] {
            self.record(EV_ACT_START, p, a);
        }
        if a == self.a_last[p] {
            self.a_done[p] = true;
            return;
        }
        if let Some(t) = self.planned_end(a, self.now) {
            self.schedule(t, ACT_END, p);
        }
    }

    // -- departing onto the first link --------------------------------------
    fn try_depart_event(&mut self, v: usize) {
        if self.a_blocked[v] != NONE {
            self.try_depart(v, false);
        }
    }

    fn force_depart(&mut self, v: usize) {
        let since = self.a_blocked[v];
        if since != NONE && self.now >= ceil_step(since as f64 + self.inp.stuck_time) {
            self.try_depart(v, true);
        }
    }

    fn try_depart(&mut self, v: usize, force: bool) {
        let link = self.inp.route_links[self.vh[v].rpos] as usize;
        // others are waiting to depart here (MATSim's waiting list): queue behind
        if let Some(&front) = self.l_departing[link].front() {
            if front != v {
                self.l_departing[link].push_back(v);
                return;
            }
        }
        if force || self.lk[link].occ < self.lk[link].storage {
            self.a_blocked[v] = NONE;
            self.a_waiting[v] = -1;
            let was_front = self.l_departing[link].pop_front().is_some();
            self.enter_link(v, link);
            if was_front {
                if let Some(&next) = self.l_departing[link].front() {
                    self.try_depart(next, false); // the next in line tries now
                }
            }
            return;
        }
        if self.l_departing[link].is_empty() {
            self.l_departing[link].push_back(v);
        }
        let was_blocked = self.a_blocked[v] != NONE;
        if !was_blocked {
            self.a_blocked[v] = self.now;
            if self.finite_stuck {
                let t = ceil_step(self.now as f64 + self.inp.stuck_time);
                self.schedule(t, FORCE_DEPART, v);
            }
        }
        if self.a_waiting[v] != link as i64 {
            self.a_waiting[v] = link as i64;
            let token = v as u64;
            if was_blocked {
                self.l_waiters[link].push_front(token);
                self.lk[link].nwait += 1;
            } else {
                self.l_waiters[link].push_back(token);
                self.lk[link].nwait += 1;
            }
        }
    }

    // -- links --------------------------------------------------------------
    /// Append vehicle `v` to `link`'s queue; true if it is now the head.
    #[inline]
    fn q_push_back(&mut self, link: usize, v: usize, nx: u32, earliest: i64) -> bool {
        let vh = &mut self.vh[v];
        vh.qnext = NIL;
        vh.qnx = nx;
        vh.qearliest = earliest;
        let tail = self.l_tail[link];
        self.l_tail[link] = v as u32;
        if tail == NIL {
            self.lk[link].head = v as u32;
            true
        } else {
            self.vh[tail as usize].qnext = v as u32;
            false
        }
    }

    /// Put vehicle `v` at the head of `link`'s queue.
    fn q_push_front(&mut self, link: usize, v: usize, nx: u32, earliest: i64) {
        let head = self.lk[link].head;
        let vh = &mut self.vh[v];
        vh.qnext = head;
        vh.qnx = nx;
        vh.qearliest = earliest;
        self.lk[link].head = v as u32;
        if head == NIL {
            self.l_tail[link] = v as u32;
        }
    }

    /// Remove and return the head of `link`'s (non-empty) queue.
    #[inline]
    fn q_pop_front(&mut self, link: usize) -> usize {
        let v = self.lk[link].head as usize;
        let next = self.vh[v].qnext;
        self.lk[link].head = next;
        if next == NIL {
            self.l_tail[link] = NIL;
        }
        v
    }

    /// The link after vehicle `v`'s current one, or END. Read when the vehicle
    /// enters a link, while that part of the route is in cache, and kept in
    /// its queue entry for `try_leave`.
    #[inline]
    fn next_of(&self, v: usize) -> u32 {
        let r = self.vh[v].rpos + 1;
        if r < self.vh[v].rend { self.inp.route_links[r] } else { END }
    }

    fn enter_link(&mut self, v: usize, link: usize) {
        let rpos = self.vh[v].rpos;
        self.out.enter[rpos] = self.now as f64;
        let nx = self.next_of(v);
        let earliest = if v < self.n_plans {
            if self.rec[EV_LINK_ENTER as usize] {
                self.record(EV_LINK_ENTER, v, link);
            }
            // a junction delay when it crosses into a next link
            let delay = if nx != END { self.inp.junction_delay } else { 0 };
            self.now + (self.lk[link].free as i64) + delay
        } else {
            if self.rec[EV_VEH_LINK_ENTER as usize] {
                self.record(EV_VEH_LINK_ENTER, v - self.n_plans, link);
            }
            let delay = if nx != END { self.inp.junction_delay } else { 0 };
            self.now + self.inp.trip_link_steps[rpos - self.inp.n_plan_links] + delay
        };
        self.vh[v].link = link as i64;
        self.lk[link].occ += self.vh[v].pce;
        if self.q_push_back(link, v, nx, earliest) {
            let t = self.head_time(link);
            self.schedule_link(link, t);
        }
    }

    /// When the head may act: its earliest exit, and also the next flow slot
    /// unless it is a trip about to serve a stop.
    fn head_time(&self, link: usize) -> i64 {
        let v = self.lk[link].head as usize;
        let earliest = self.vh[v].qearliest;
        if v >= self.n_plans && self.has_stop_here(v - self.n_plans) {
            // it may have queued past the time it reaches the stop
            return (earliest - self.jd(v)).max(self.now);
        }
        earliest.max(ceil_step(self.lk[link].nft))
    }

    /// The junction delay after vehicle `v`'s current link (none after its last).
    #[inline]
    fn jd(&self, v: usize) -> i64 {
        if self.vh[v].rpos + 1 < self.vh[v].rend { self.inp.junction_delay } else { 0 }
    }

    #[inline]
    fn schedule_link(&mut self, link: usize, t: i64) {
        let pending = self.lk[link].pending;
        if pending != NONE && pending <= t {
            return;
        }
        self.lk[link].pending = t;
        self.schedule(t, TRY_LEAVE, link);
    }

    fn try_leave_event(&mut self, link: usize) {
        if self.lk[link].pending != self.now {
            return; // superseded by an earlier wake-up
        }
        self.lk[link].pending = NONE;
        self.try_leave(link, false);
    }

    fn force_leave(&mut self, link: usize) {
        let since = self.lk[link].blocked;
        if since != NONE && self.now >= ceil_step(since as f64 + self.inp.stuck_time) {
            self.try_leave(link, true);
        }
    }

    fn try_leave(&mut self, link: usize, force: bool) {
        let v = self.lk[link].head;
        if v == NIL {
            return;
        }
        let v = v as usize;
        let (nx, earliest) = (self.vh[v].qnx, self.vh[v].qearliest);
        if v >= self.n_plans && self.at_link_end(v - self.n_plans, link, earliest) {
            return;
        }
        let ready = earliest.max(ceil_step(self.lk[link].nft));
        if self.now < ready {
            self.schedule_link(link, ready);
            return;
        }
        if nx == END {
            // leaving the last link: arrive
            self.pop_head(link, ready);
            if v < self.n_plans {
                self.arrive(v);
            }
            return;
        }
        let nxt = nx as usize;
        if force || self.lk[nxt].occ < self.lk[nxt].storage {
            self.pop_head(link, ready);
            self.vh[v].rpos += 1;
            self.enter_link(v, nxt);
            return;
        }
        // blocked by downstream storage: wait for space (or stuck time)
        let was_blocked = self.lk[link].blocked != NONE;
        if !was_blocked {
            self.lk[link].blocked = self.now;
            if self.finite_stuck {
                let t = ceil_step(self.now as f64 + self.inp.stuck_time);
                self.schedule(t, FORCE_LEAVE, link);
            }
        }
        if self.lk[link].waiting != nxt as i32 {
            self.lk[link].waiting = nxt as i32;
            let token = LINK_TOKEN | link as u64;
            if was_blocked {
                self.l_waiters[nxt].push_front(token);
                self.lk[nxt].nwait += 1;
            } else {
                self.l_waiters[nxt].push_back(token);
                self.lk[nxt].nwait += 1;
            }
        }
    }

    fn pop_head(&mut self, link: usize, ready: i64) {
        let v = self.q_pop_front(link);
        let earliest = self.vh[v].qearliest;
        if v < self.n_plans {
            if self.rec[EV_LINK_LEAVE as usize] {
                self.record(EV_LINK_LEAVE, v, link);
            }
        } else if self.rec[EV_VEH_LINK_LEAVE as usize] {
            self.record(EV_VEH_LINK_LEAVE, v - self.n_plans, link);
        }
        self.vh[v].link = -1;
        let pce = self.vh[v].pce;
        self.lk[link].occ -= pce;
        // left at its first possible second: count the next slot from the
        // unrounded time (no drift); held up: from now
        let nft = self.lk[link].nft;
        let base = if self.now == ready {
            let e = earliest as f64;
            if e > nft { e } else { nft }
        } else {
            self.now as f64
        };
        let nft = base + self.lk[link].headway * pce;
        self.lk[link].nft = nft;
        self.lk[link].blocked = NONE;
        self.lk[link].waiting = -1;
        if self.lk[link].head != NIL {
            let t = self.head_time(link);
            self.schedule_link(link, t);
        }
        // a slot was freed: wake the first valid waiter
        while self.lk[link].nwait > 0 {
            self.lk[link].nwait -= 1;
            let token = self.l_waiters[link].pop_front().expect("counted waiter");
            if token & LINK_TOKEN != 0 {
                let up = (token & !LINK_TOKEN) as usize;
                if self.lk[up].waiting != link as i32 {
                    continue; // stale token
                }
                self.lk[up].waiting = -1;
                let now = self.now;
                self.schedule_link(up, now);
            } else {
                let a = token as usize;
                if self.a_waiting[a] != link as i64 {
                    continue;
                }
                self.a_waiting[a] = -1;
                let now = self.now;
                self.schedule(now, TRY_DEPART, a);
            }
            return;
        }
    }

    // -- passengers ---------------------------------------------------------
    fn pt_depart(&mut self, p: usize) {
        let leg = self.a_leg[p] as usize;
        let r0 = self.inp.ride_offsets[leg] as usize;
        let r1 = self.inp.ride_offsets[leg + 1] as usize;
        if r0 == r1 {
            // walk only
            let t = self.now + self.inp.walk_after_steps[leg];
            self.schedule(t, PAX_ARRIVE, p);
            return;
        }
        self.a_ride[p] = r0;
        self.a_ride_end[p] = r1;
        let t = self.now + self.inp.walk_before_steps[r0];
        self.schedule(t, PAX_AT_STOP, p);
    }

    fn pax_at_stop(&mut self, p: usize) {
        let r = self.a_ride[p];
        let k = self.inp.ride_board[r] as usize;
        self.out.ride_wait[r] = self.now as f64;
        if self.rec[EV_WAITING as usize] {
            self.record(EV_WAITING, p, k);
        }
        let n = self.at_stop[k].len();
        if self.inp.rs_network[k] == 0 {
            for i in 0..n {
                let d = self.at_stop[k][i] as usize;
                if (self.tr_load[d] as f64) < self.inp.trip_capacity[d] {
                    self.tr_load[d] += 1;
                    self.board(d, p);
                    return;
                }
            }
            self.waiting[k].push_back(p as u32);
            return;
        }
        self.waiting[k].push_back(p as u32);
        for i in 0..n {
            let d = self.at_stop[k][i] as usize;
            if self.serve_new_boarder(d) {
                return;
            }
        }
    }

    fn board(&mut self, d: usize, p: usize) {
        let r = self.a_ride[p];
        self.out.ride_board[r] = self.now as f64;
        self.out.ride_trip[r] = d as i64;
        self.tr_onboard[d].push(p as u32);
        if self.rec[EV_ENTER_VEH as usize] {
            self.record(EV_ENTER_VEH, p, d);
        }
    }

    fn alight(&mut self, d: usize, p: usize) {
        let r = self.a_ride[p];
        self.out.ride_alight[r] = self.now as f64;
        let onboard = &mut self.tr_onboard[d];
        let i = onboard.iter().position(|&x| x as usize == p).expect("passenger on board");
        onboard.remove(i);
        self.tr_load[d] -= 1;
        if self.rec[EV_LEAVE_VEH as usize] {
            self.record(EV_LEAVE_VEH, p, d);
        }
        if r + 1 < self.a_ride_end[p] {
            self.a_ride[p] = r + 1;
            let t = self.now + self.inp.walk_before_steps[r + 1];
            self.schedule(t, PAX_AT_STOP, p);
        } else {
            let t = self.now + self.inp.walk_after_steps[self.a_leg[p] as usize];
            self.schedule(t, PAX_ARRIVE, p);
        }
    }

    fn alighters(&self, d: usize, k: usize) -> Vec<u32> {
        self.tr_onboard[d]
            .iter()
            .copied()
            .filter(|&p| self.inp.ride_alight[self.a_ride[p as usize]] as usize == k)
            .collect()
    }

    #[inline]
    fn trip_stop(&self, d: usize, k: usize) -> usize {
        (self.inp.trip_stop_offsets[d] + k as i64 - self.inp.trip_k0[d]) as usize
    }

    // -- trips: common ------------------------------------------------------
    fn trip_start(&mut self, d: usize) {
        if self.rec[EV_DRIVER_START as usize] {
            self.record(EV_DRIVER_START, d, 0);
        }
        if self.inp.trip_network[d] != 0 {
            let v = self.n_plans + d;
            let n = (self.inp.trip_link_offsets[d + 1] - self.inp.trip_link_offsets[d]) as usize;
            self.vh[v].rpos = self.tr_rstart[d];
            self.vh[v].rend = self.tr_rstart[d] + n;
            self.try_depart(v, false);
        } else {
            let j = self.inp.trip_stop_offsets[d] as usize;
            let t = self.now.max(self.inp.trip_stop_arr_step[j]);
            self.schedule(t, STOP_ARRIVE, d);
        }
    }

    fn record_arrival(&mut self, d: usize, k: usize) {
        let j = self.trip_stop(d, k);
        self.out.stop_arr[j] = self.now as f64;
        if self.rec[EV_VEH_ARRIVE as usize] {
            self.record(EV_VEH_ARRIVE, d, k);
        }
        self.at_stop[k].push(d as u32);
    }

    fn stop_depart(&mut self, d: usize, token: u64) {
        if token != self.tr_token[d] & TOKEN_MASK {
            return;
        }
        let net = self.inp.trip_network[d] != 0;
        if net && self.tr_state[d] != CLOSED {
            return;
        }
        let k = self.tr_next[d];
        let j = self.trip_stop(d, k);
        self.out.stop_dep[j] = self.now as f64;
        if self.rec[EV_VEH_DEPART as usize] {
            self.record(EV_VEH_DEPART, d, k);
        }
        let trips = &mut self.at_stop[k];
        let i = trips.iter().position(|&x| x as usize == d).expect("trip at stop");
        trips.remove(i);
        self.tr_next[d] = k + 1;
        self.tr_visit[d] = -1;
        self.tr_state[d] = 0;
        if !net {
            if ((k + 1) as i64) < self.inp.trip_k1[d] {
                let t = self.now.max(self.inp.trip_stop_arr_step[j + 1]);
                self.schedule(t, STOP_ARRIVE, d);
            }
            return;
        }
        if self.has_stop_here(d) {
            // several stops on one link
            self.reach_stop(d);
            return;
        }
        let v = self.n_plans + d;
        let link = self.inp.route_links[self.vh[v].rpos] as usize;
        // it leaves the link from now, crossing a junction one second later
        let earliest = self.now + self.jd(v);
        if self.tr_pulled[d] {
            // back from the stop into the lane, as its head
            self.tr_pulled[d] = false;
            let nx = self.next_of(v);
            self.q_push_front(link, v, nx, earliest);
            self.lk[link].blocked = NONE;
            self.lk[link].waiting = -1;
        } else {
            self.vh[v].qearliest = earliest;
        }
        let t = earliest.max(ceil_step(self.lk[link].nft));
        self.schedule_link(link, t);
    }

    // -- trips: schedule mode -----------------------------------------------
    fn sched_arrive(&mut self, d: usize) {
        let k = self.tr_next[d];
        self.record_arrival(d, k);
        for p in self.alighters(d, k) {
            self.alight(d, p as usize);
        }
        while !self.waiting[k].is_empty() && (self.tr_load[d] as f64) < self.inp.trip_capacity[d] {
            self.tr_load[d] += 1;
            let p = self.waiting[k].pop_front().unwrap() as usize;
            self.board(d, p);
        }
        self.tr_token[d] += 1;
        let t = self.now.max(self.inp.trip_stop_dep_step[self.trip_stop(d, k)]);
        self.schedule_late(t, STOP_DEPART, self.tr_token[d], d);
    }

    // -- trips: network mode ------------------------------------------------
    fn has_stop_here(&self, d: usize) -> bool {
        let k = self.tr_next[d];
        (k as i64) < self.inp.trip_k1[d]
            && self.inp.rs_link_pos[k]
                == (self.vh[self.n_plans + d].rpos - self.tr_rstart[d]) as i64
    }

    /// `try_leave` hook for trip `d` at the head of `link`: true if it is
    /// (now) serving a stop instead of leaving.
    fn at_link_end(&mut self, d: usize, link: usize, earliest: i64) -> bool {
        if self.tr_visit[d] >= 0 {
            return true;
        }
        if !self.has_stop_here(d) {
            return false;
        }
        // it serves stops on reaching the link's end
        let reach = earliest - self.jd(self.n_plans + d);
        if self.now < reach {
            self.schedule_link(link, reach);
            return true;
        }
        let stop = self.inp.rs_stop[self.tr_next[d]] as usize;
        if self.inp.stop_blocking[stop] == 0 {
            self.q_pop_front(link); // into the stop: off the lane, still in storage
            self.tr_pulled[d] = true;
            if self.lk[link].head != NIL {
                let t = self.head_time(link);
                self.schedule_link(link, t);
            }
        }
        self.reach_stop(d);
        true
    }

    fn reach_stop(&mut self, d: usize) {
        let k = self.tr_next[d];
        self.tr_visit[d] = k as i64;
        self.record_arrival(d, k);
        self.tr_alight_q[d] = self.alighters(d, k).into_iter().collect();
        if !self.tr_alight_q[d].is_empty() || self.boarders(d) {
            self.open_doors(d);
        } else {
            self.ready(d);
        }
    }

    fn boarders(&self, d: usize) -> bool {
        let q = &self.waiting[self.tr_visit[d] as usize];
        !q.is_empty()
            && ((self.tr_load[d] - self.tr_alight_q[d].len() as i64) as f64) < self.inp.trip_capacity[d]
    }

    fn open_doors(&mut self, d: usize) {
        self.tr_state[d] = OPENING;
        self.tr_token[d] += 1;
        let t = self.now + self.inp.door_open_steps;
        self.push(t, entry(DOOR_OPEN, self.tr_token[d], d));
    }

    fn door_open(&mut self, d: usize, token: u64) {
        if token != self.tr_token[d] & TOKEN_MASK || self.tr_state[d] != OPENING {
            return;
        }
        self.tr_state[d] = OPEN;
        self.next_alight(d);
        if self.inp.trip_parallel[d] != 0 || !self.tr_alight_busy[d] {
            self.next_board(d);
        }
        self.maybe_close(d);
    }

    fn next_alight(&mut self, d: usize) {
        let busy = !self.tr_alight_q[d].is_empty();
        self.tr_alight_busy[d] = busy;
        if busy {
            let t = self.now + self.inp.trip_egress_steps[d];
            self.schedule(t, ALIGHT_DONE, d);
        }
    }

    fn alight_done(&mut self, d: usize) {
        let p = self.tr_alight_q[d].pop_front().expect("an alighting passenger") as usize;
        self.alight(d, p);
        self.next_alight(d);
        if self.inp.trip_parallel[d] == 0 && !self.tr_alight_busy[d] && !self.tr_board_busy[d] {
            self.next_board(d);
        }
        self.maybe_close(d);
    }

    fn next_board(&mut self, d: usize) {
        if self.boarders(d) {
            let k = self.tr_visit[d] as usize;
            let p = self.waiting[k].pop_front().unwrap();
            self.tr_load[d] += 1;
            self.tr_board_busy[d] = true;
            self.tr_boarding[d] = p as i64;
            let t = self.now + self.inp.trip_access_steps[d];
            self.schedule(t, BOARD_DONE, d);
        } else {
            self.tr_board_busy[d] = false;
        }
    }

    fn board_done(&mut self, d: usize) {
        let p = self.tr_boarding[d] as usize;
        self.tr_boarding[d] = -1;
        self.board(d, p);
        self.next_board(d);
        self.maybe_close(d);
    }

    fn maybe_close(&mut self, d: usize) {
        if self.tr_state[d] == OPEN && !self.tr_alight_busy[d] && !self.tr_board_busy[d] {
            self.tr_state[d] = CLOSING;
            let t = self.now + self.inp.door_close_steps;
            self.schedule(t, DOOR_CLOSE, d);
        }
    }

    fn door_close(&mut self, d: usize) {
        if self.tr_state[d] == CLOSING {
            self.ready(d);
        }
    }

    /// Doors closed and nobody to serve: depart (or await the departure).
    fn ready(&mut self, d: usize) {
        if self.boarders(d) {
            self.open_doors(d);
            return;
        }
        self.tr_state[d] = CLOSED;
        let k = self.tr_visit[d] as usize;
        let mut t = self.now;
        if self.inp.rs_await[k] != 0 {
            t = t.max(self.inp.trip_stop_dep_step[self.trip_stop(d, k)]);
        }
        self.tr_token[d] += 1;
        self.schedule_late(t, STOP_DEPART, self.tr_token[d], d);
    }

    /// A passenger joined the line at trip `d`'s stop: true if it takes them.
    fn serve_new_boarder(&mut self, d: usize) -> bool {
        if !self.boarders(d) {
            return false;
        }
        match self.tr_state[d] {
            OPEN => {
                if !self.tr_board_busy[d]
                    && (self.inp.trip_parallel[d] != 0 || !self.tr_alight_busy[d])
                {
                    self.next_board(d);
                }
                true
            }
            CLOSED => {
                self.open_doors(d);
                true
            }
            s => s == OPENING || s == CLOSING,
        }
    }

    // -- main loop ----------------------------------------------------------
    fn run(mut self) -> Outputs {
        let n_plans = self.n_plans;
        self.now = -1; // nothing is "current" before the loop starts
        for d in 0..self.inp.trip_k0.len() {
            // trips first, as the QSim schedules them
            let t = self.inp.trip_start_step[d].max(0);
            self.schedule(t, TRIP_START, d);
        }
        for p in 0..n_plans {
            if self.a_act[p] == self.a_last[p] {
                self.a_done[p] = true;
                continue;
            }
            if let Some(t) = self.planned_end(self.a_act[p], 0) {
                self.schedule(t, ACT_END, p);
            }
        }
        self.now = 0;
        let end_time = self.inp.end_time;
        while let Some(t) = self.next_time() {
            if (t as f64) > end_time {
                break;
            }
            self.now = t;
            if t < DENSE_LIMIT {
                self.cursor = t as usize; // nothing is pending before `t`
            }
            self.cur = self.take_bucket(t);
            self.cur_late = self.late.remove(&t).unwrap_or_default();
            let (mut i, mut j) = (0, 0);
            loop {
                while i < self.cur.len() {
                    let e = self.cur[i];
                    i += 1;
                    self.dispatch(e);
                }
                if j == self.cur_late.len() {
                    break;
                }
                let e = self.cur_late[j];
                j += 1;
                self.dispatch(e);
            }
            self.cur.clear();
            let drained = std::mem::take(&mut self.cur);
            self.pool.push(drained);
            self.cur_late.clear();
        }

        let mut now = self.now as f64;
        if end_time.is_finite() {
            now = now.max(end_time);
        }
        for p in 0..n_plans {
            if !self.a_done[p] {
                self.out.stuck[p] = true;
                self.out.stuck_leg.push(self.a_leg[p]);
                self.out.stuck_link.push(self.vh[p].link);
            }
        }
        self.out.last_time = now;
        self.out
    }
}

fn err(msg: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.to_string())
}

/// Run the HermesSim kernel. Same arguments and results as
/// `mobslim.mobsim.hermes.run_kernel`.
#[pyfunction]
#[pyo3(signature = (*, free_steps, headway, storage, act_offsets, act_end_time, act_duration,
                    leg_network, leg_teleport_steps, route_offsets, route_links,
                    stuck_time, end_time, event_mask,
                    network_vehicles, leg_transit, ride_offsets, ride_board, ride_alight,
                    walk_before_steps, walk_after_steps, rs_stop, rs_link_pos, rs_await,
                    stop_blocking, trip_k0, trip_k1, trip_start_step, trip_stop_offsets,
                    trip_stop_arr_step, trip_stop_dep_step, trip_link_offsets, trip_links,
                    trip_link_steps, trip_capacity, trip_pce, trip_access_steps,
                    trip_egress_steps, trip_parallel, door_open_steps, door_close_steps,
                    trip_network, rs_network,
                    junction_delay=0))]
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
    junction_delay: i64,
) -> PyResult<Bound<'py, PyTuple>> {
    let n_links = free_steps.as_array().len();
    let net = network_vehicles != 0;
    let mut links: Vec<i64> = route_links.as_array().to_vec();
    let n_plan_links = links.len();
    if net {
        links.extend(trip_links.as_array().iter());
    }
    if links.iter().any(|&l| l < 0 || l as usize >= n_links) {
        return Err(err("route link index out of range"));
    }
    let act_offsets = act_offsets.as_array().to_vec();
    if act_offsets.is_empty() {
        return Err(err("act_offsets must not be empty"));
    }
    let n_plans = act_offsets.len() - 1;
    let n_trips = trip_k0.as_array().len();
    // vehicle ids must stay below NIL; link ids fit `LinkHot::waiting` (i32)
    if n_plans + n_trips >= u32::MAX as usize || n_links > i32::MAX as usize {
        return Err(err("too many persons, trips or links"));
    }
    if free_steps.as_array().iter().any(|&s| s < 0 || s > i32::MAX as i64) {
        return Err(err("free-speed times must be between 0 and 2^31 - 1 seconds"));
    }
    let n_rs = rs_stop.as_array().len();
    let n_rides = ride_board.as_array().len();
    let rides_ok = ride_board.as_array().iter().chain(ride_alight.as_array().iter())
        .all(|&k| k >= 0 && (k as usize) < n_rs);
    let trips_ok = trip_k0.as_array().iter().zip(trip_k1.as_array().iter())
        .all(|(&a, &b)| 0 <= a && a <= b && (b as usize) <= n_rs);
    if !rides_ok || !trips_ok || ride_offsets.as_array().iter().any(|&o| o < 0 || o as usize > n_rides) {
        return Err(err("route stop index out of range"));
    }
    let mut v_pce = vec![1.0; n_plans];
    v_pce.extend(trip_pce.as_array().iter());
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
        route_links: links.iter().map(|&l| l as u32).collect(),
        n_plan_links,
        stuck_time,
        end_time,
        event_mask,
        network_vehicles: net,
        leg_transit: leg_transit.as_array().to_vec(),
        ride_offsets: ride_offsets.as_array().to_vec(),
        ride_board: ride_board.as_array().to_vec(),
        ride_alight: ride_alight.as_array().to_vec(),
        walk_before_steps: walk_before_steps.as_array().to_vec(),
        walk_after_steps: walk_after_steps.as_array().to_vec(),
        rs_stop: rs_stop.as_array().to_vec(),
        rs_link_pos: rs_link_pos.as_array().to_vec(),
        rs_await: rs_await.as_array().to_vec(),
        stop_blocking: stop_blocking.as_array().to_vec(),
        trip_k0: trip_k0.as_array().to_vec(),
        trip_k1: trip_k1.as_array().to_vec(),
        trip_start_step: trip_start_step.as_array().to_vec(),
        trip_stop_offsets: trip_stop_offsets.as_array().to_vec(),
        trip_stop_arr_step: trip_stop_arr_step.as_array().to_vec(),
        trip_stop_dep_step: trip_stop_dep_step.as_array().to_vec(),
        trip_link_offsets: trip_link_offsets.as_array().to_vec(),
        trip_link_steps: trip_link_steps.as_array().to_vec(),
        trip_capacity: trip_capacity.as_array().to_vec(),
        trip_access_steps: trip_access_steps.as_array().to_vec(),
        trip_egress_steps: trip_egress_steps.as_array().to_vec(),
        trip_parallel: trip_parallel.as_array().to_vec(),
        door_open_steps,
        door_close_steps,
        trip_network: trip_network.as_array().to_vec(),
        rs_network: rs_network.as_array().to_vec(),
        v_pce,
        junction_delay,
    };
    let o = py.allow_threads(|| Sim::new(&inp).run());
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
