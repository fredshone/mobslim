//! Least-cost RAPTOR: a port of `mobslim.transit.raptor.RaptorRouter._search`.
//!
//! Same labels, same scan order, same strict-improvement and pruning rules
//! and the same arithmetic (distances are `sqrt(dx * dx + dy * dy)` on both
//! sides), so it returns *identical* routes. The tables come from
//! `RustRaptorRouter` in Python: per route, the stop times of its trips
//! (trip-major `arr` / `dep` and position-major `cols` for the binary
//! search), the stop -> route stop index, and the footpaths.
//! `route_many` groups queries by `(origin, departure time, costs)`, answers each
//! group with one search, releases the GIL and spreads groups over threads.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

const NONE: u32 = u32::MAX;
const ACCESS: u8 = 0;
const TRANSFER: u8 = 1;
const RIDE: u8 = 2;

#[derive(Clone, Copy)]
struct Label {
    kind: u8,
    cost: f64,
    time: f64,
    prev: u32,
    a: u32,    // RIDE: board route stop
    b: u32,    // RIDE: alight route stop
    dist: f64, // ACCESS / TRANSFER: walk distance
}

#[derive(Clone, Copy)]
struct Board {
    lab: u32,
    pos: usize,
    trip: usize,
    cost: f64,
    dep: f64,
}

#[derive(Clone, Copy)]
struct Egress {
    dest: usize,
    wt: f64,
    wc: f64,
    wd: f64,
}

/// Per-thread scratch, reset through touched lists.
struct Scratch {
    labels: Vec<Label>,
    stop_label: Vec<u32>,
    arr_label: Vec<u32>,
    rs_best: Vec<f64>,
    touched_stops: Vec<usize>,
    touched_rs: Vec<usize>,
    marked: Vec<bool>,
    marked_list: Vec<usize>,
    improved: Vec<bool>,
    improved_list: Vec<usize>,
    scan: Vec<usize>,
    scan_list: Vec<usize>,
    egress: Vec<Vec<Egress>>,
    egress_list: Vec<usize>,
}

impl Scratch {
    fn new(n_stops: usize, n_rs: usize, n_routes: usize) -> Scratch {
        Scratch {
            labels: Vec::new(),
            stop_label: vec![NONE; n_stops],
            arr_label: vec![NONE; n_stops],
            rs_best: vec![f64::INFINITY; n_rs],
            touched_stops: Vec::new(),
            touched_rs: Vec::new(),
            marked: vec![false; n_stops],
            marked_list: Vec::new(),
            improved: vec![false; n_stops],
            improved_list: Vec::new(),
            scan: vec![usize::MAX; n_routes],
            scan_list: Vec::new(),
            egress: (0..n_stops).map(|_| Vec::new()).collect(),
            egress_list: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.labels.clear();
        for &s in &self.touched_stops {
            self.stop_label[s] = NONE;
            self.arr_label[s] = NONE;
        }
        self.touched_stops.clear();
        for &k in &self.touched_rs {
            self.rs_best[k] = f64::INFINITY;
        }
        self.touched_rs.clear();
        for &s in &self.marked_list {
            self.marked[s] = false;
        }
        self.marked_list.clear();
        for &s in &self.egress_list {
            self.egress[s].clear();
        }
        self.egress_list.clear();
    }
}

struct Answer {
    rides: Vec<(u32, u32)>,
    travel_time: f64,
    distance: f64,
}

#[pyclass(frozen)]
pub struct Raptor {
    stop_x: Vec<f64>,
    stop_y: Vec<f64>,
    rs_offsets: Vec<usize>, // per route (n_routes + 1)
    rs_stop: Vec<usize>,
    rs_distance: Vec<f64>,
    n_trips: Vec<usize>,    // per route
    time_base: Vec<usize>,  // per route: offset into arr / dep / cols
    arr: Vec<f64>,          // [time_base[r] + trip * n_r + i]
    dep: Vec<f64>,
    cols: Vec<f64>,         // [time_base[r] + i * n_trips[r] + trip]
    stop_rs_offsets: Vec<usize>,
    stop_rs: Vec<(usize, usize)>, // (route, position)
    tr_offsets: Vec<usize>,
    tr_to: Vec<usize>,
    tr_walk: Vec<f64>,
    tr_time: Vec<f64>,
    tr_dist: Vec<f64>,
    walk_speed: f64,
    walk_factor: f64,
    radius: f64,
    extension: f64,
    max_transfers: usize,
    after_first: usize,
    c_travel: f64,
    c_wait: f64,
    c_walk: f64,
    c_switch: f64,
}

/// Python's `bisect.bisect_left`.
fn bisect_left(a: &[f64], x: f64) -> usize {
    let (mut lo, mut hi) = (0, a.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

impl Raptor {
    fn n_stops(&self) -> usize {
        self.stop_x.len()
    }

    #[inline]
    fn walk_time(&self, d: f64) -> f64 {
        d * self.walk_factor / self.walk_speed
    }

    /// `[(stop, beeline distance)]` within `radius`, else within the nearest
    /// stop's distance + `extension`.
    fn nearby(&self, x: f64, y: f64, out: &mut Vec<(usize, f64)>) {
        out.clear();
        let n = self.n_stops();
        if n == 0 {
            return;
        }
        let d: Vec<f64> = (0..n)
            .map(|s| {
                let dx = self.stop_x[s] - x;
                let dy = self.stop_y[s] - y;
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        out.extend(d.iter().enumerate().filter(|(_, &v)| v <= self.radius).map(|(s, &v)| (s, v)));
        if out.is_empty() {
            let m = d.iter().copied().fold(f64::INFINITY, f64::min) + self.extension;
            out.extend(d.iter().enumerate().filter(|(_, &v)| v <= m).map(|(s, &v)| (s, v)));
        }
    }

    /// `costs`: `[travel, wait, walk]` per second and per transfer.
    fn search(&self, sc: &mut Scratch, ox: f64, oy: f64, t0: f64, dests: &[(f64, f64)],
              costs: [f64; 4]) -> Vec<Answer> {
        sc.reset();
        let [c_travel, c_wait, c_walk, c_switch] = costs;
        let factor = self.walk_factor;
        let n_dest = dests.len();

        // direct walks: the initial answer (and pruning bound) per destination
        let mut walks = Vec::with_capacity(n_dest);
        let mut best_cost = Vec::with_capacity(n_dest);
        let mut best: Vec<Option<(u32, f64, f64)>> = vec![None; n_dest];
        for &(dx, dy) in dests {
            let ex = dx - ox;
            let ey = dy - oy;
            let dist = (ex * ex + ey * ey).sqrt() * factor;
            let wt = dist / self.walk_speed;
            walks.push((wt, dist));
            best_cost.push(wt * c_walk);
        }
        let mut near = Vec::new();
        for (j, &(dx, dy)) in dests.iter().enumerate() {
            self.nearby(dx, dy, &mut near);
            for &(st, d) in &near {
                let wt = self.walk_time(d);
                if sc.egress[st].is_empty() {
                    sc.egress_list.push(st);
                }
                sc.egress[st].push(Egress { dest: j, wt, wc: wt * c_walk, wd: d * factor });
            }
        }
        let mut active = vec![true; n_dest];
        let mut first_round: Vec<Option<usize>> = vec![None; n_dest];
        let bound = |best_cost: &[f64], active: &[bool]| -> f64 {
            let mut b = f64::NEG_INFINITY;
            let mut any = false;
            for (c, &a) in best_cost.iter().zip(active) {
                if a {
                    b = if any { b.max(*c) } else { *c };
                    any = true;
                }
            }
            b
        };

        self.nearby(ox, oy, &mut near);
        for &(st, d) in &near {
            let wt = self.walk_time(d);
            let l = sc.labels.len() as u32;
            sc.labels.push(Label { kind: ACCESS, cost: wt * c_walk, time: t0 + wt, prev: NONE,
                                   a: st as u32, b: 0, dist: d * factor });
            if sc.stop_label[st] == NONE && sc.arr_label[st] == NONE {
                sc.touched_stops.push(st);
            }
            sc.stop_label[st] = l;
            if !sc.marked[st] {
                sc.marked[st] = true;
                sc.marked_list.push(st);
            }
        }
        let mut b_cost = bound(&best_cost, &active);

        for rnd in 1..self.max_transfers + 2 {
            if sc.marked_list.is_empty() || b_cost == f64::NEG_INFINITY {
                break;
            }
            for &st in &sc.marked_list {
                for &(r, i) in &self.stop_rs[self.stop_rs_offsets[st]..self.stop_rs_offsets[st + 1]] {
                    if sc.scan[r] == usize::MAX {
                        sc.scan_list.push(r);
                        sc.scan[r] = i;
                    } else if i < sc.scan[r] {
                        sc.scan[r] = i;
                    }
                }
            }
            sc.scan_list.sort_unstable();
            let scan_list = std::mem::take(&mut sc.scan_list);
            for &r in &scan_list {
                let start = sc.scan[r];
                sc.scan[r] = usize::MAX;
                let k0 = self.rs_offsets[r];
                let n_r = self.rs_offsets[r + 1] - k0;
                let nt = self.n_trips[r];
                let base = self.time_base[r];
                let mut board: Option<Board> = None;
                for i in start..n_r {
                    let k = k0 + i;
                    let st = self.rs_stop[k];
                    if let Some(bd) = board {
                        let ta = self.arr[base + bd.trip * n_r + i];
                        let c = bd.cost + (ta - bd.dep) * c_travel;
                        if c < b_cost && c < sc.rs_best[k] {
                            if sc.rs_best[k] == f64::INFINITY {
                                sc.touched_rs.push(k);
                            }
                            sc.rs_best[k] = c;
                            let elem = sc.labels.len() as u32;
                            sc.labels.push(Label { kind: RIDE, cost: c, time: ta, prev: bd.lab,
                                                   a: (k0 + bd.pos) as u32, b: k as u32, dist: 0.0 });
                            let cur = sc.arr_label[st];
                            if cur == NONE || c < sc.labels[cur as usize].cost {
                                if sc.stop_label[st] == NONE && cur == NONE {
                                    sc.touched_stops.push(st);
                                }
                                sc.arr_label[st] = elem;
                                if !sc.improved[st] {
                                    sc.improved[st] = true;
                                    sc.improved_list.push(st);
                                }
                            }
                            for e in &sc.egress[st] {
                                let j = e.dest;
                                if active[j] && c + e.wc < best_cost[j] {
                                    best_cost[j] = c + e.wc;
                                    best[j] = Some((elem, e.wt, e.wd));
                                    if first_round[j].is_none() {
                                        first_round[j] = Some(rnd);
                                    }
                                    b_cost = bound(&best_cost, &active);
                                }
                            }
                        }
                    }
                    if sc.marked[st] {
                        let lab = sc.stop_label[st];
                        let l = sc.labels[lab as usize];
                        let col = &self.cols[base + i * nt..base + (i + 1) * nt];
                        let idx = bisect_left(col, l.time);
                        if idx < col.len() {
                            let td = col[idx];
                            let nc = l.cost + (td - l.time) * c_wait;
                            let better = match board {
                                None => true,
                                Some(bd) => nc < bd.cost + (self.dep[base + bd.trip * n_r + i] - bd.dep) * c_travel,
                            };
                            if nc < b_cost && better {
                                board = Some(Board { lab, pos: i, trip: idx, cost: nc, dep: td });
                            }
                        }
                    }
                }
            }
            sc.scan_list = scan_list;
            sc.scan_list.clear();

            // footpaths (transfers) from stops reached by vehicle this round
            for &st in &sc.marked_list {
                sc.marked[st] = false;
            }
            sc.marked_list.clear();
            sc.improved_list.sort_unstable();
            let improved = std::mem::take(&mut sc.improved_list);
            for &st in &improved {
                sc.improved[st] = false;
                let e = sc.labels[sc.arr_label[st] as usize];
                let el = sc.arr_label[st];
                for t in self.tr_offsets[st]..self.tr_offsets[st + 1] {
                    let (to, wt, tt, wd) = (self.tr_to[t], self.tr_walk[t], self.tr_time[t], self.tr_dist[t]);
                    let c = e.cost + wt * c_walk + (tt - wt) * c_wait + c_switch;
                    if c >= b_cost {
                        continue;
                    }
                    let cur = sc.stop_label[to];
                    if cur == NONE || c < sc.labels[cur as usize].cost {
                        if cur == NONE && sc.arr_label[to] == NONE {
                            sc.touched_stops.push(to);
                        }
                        let l = sc.labels.len() as u32;
                        sc.labels.push(Label { kind: TRANSFER, cost: c, time: e.time + tt, prev: el,
                                               a: st as u32, b: to as u32, dist: wd });
                        sc.stop_label[to] = l;
                        if !sc.marked[to] {
                            sc.marked[to] = true;
                            sc.marked_list.push(to);
                        }
                    }
                }
            }
            sc.improved_list = improved;
            sc.improved_list.clear();
            for j in 0..n_dest {
                if let Some(f) = first_round[j] {
                    if active[j] && rnd >= f + self.after_first {
                        active[j] = false;
                    }
                }
            }
            b_cost = bound(&best_cost, &active);
        }
        for &st in &sc.marked_list {
            sc.marked[st] = false;
        }
        sc.marked_list.clear();

        (0..n_dest).map(|j| self.result(sc, best[j], walks[j], t0)).collect()
    }

    fn result(&self, sc: &Scratch, found: Option<(u32, f64, f64)>, walk: (f64, f64), t0: f64) -> Answer {
        let (elem, egress_time, egress_dist) = match found {
            None => return Answer { rides: Vec::new(), travel_time: walk.0, distance: walk.1 },
            Some(f) => f,
        };
        let mut rides = Vec::new();
        let mut walks = Vec::new(); // access, then one per transfer
        let mut e = elem;
        while e != NONE {
            let l = &sc.labels[e as usize];
            match l.kind {
                RIDE => rides.push((l.a, l.b)),
                _ => walks.push(l.dist),
            }
            e = l.prev;
        }
        rides.reverse();
        walks.reverse();
        let mut distance = walks[0];
        for (n, &(b, a)) in rides.iter().enumerate() {
            distance += self.rs_distance[a as usize] - self.rs_distance[b as usize];
            if n + 1 < rides.len() {
                distance += walks[n + 1];
            }
        }
        distance += egress_dist;
        let travel_time = sc.labels[elem as usize].time + egress_time - t0;
        Answer { rides, travel_time, distance }
    }

    /// `costs`: per query (`None`: the router's own for every query).
    #[allow(clippy::too_many_arguments)]
    fn solve(&self, ox: &[f64], oy: &[f64], t: &[f64], dx: &[f64], dy: &[f64],
             costs: Option<&[[f64; 4]]>, threads: usize) -> Vec<Answer> {
        let nq = ox.len();
        let own = [self.c_travel, self.c_wait, self.c_walk, self.c_switch];
        let cost = |q: usize| costs.map_or(own, |c| c[q]);
        let key = |q: usize| {
            let c = cost(q);
            (ox[q].to_bits(), oy[q].to_bits(), t[q].to_bits(),
             [c[0].to_bits(), c[1].to_bits(), c[2].to_bits(), c[3].to_bits()])
        };
        let mut order: Vec<usize> = (0..nq).collect();
        order.sort_by(|&a, &b| key(a).cmp(&key(b)).then(a.cmp(&b)));
        let mut groups: Vec<&[usize]> = Vec::new();
        let mut start = 0;
        for i in 1..=nq {
            if i == nq || key(order[i]) != key(order[start]) {
                groups.push(&order[start..i]);
                start = i;
            }
        }
        let threads = threads.max(1).min(groups.len().max(1));
        let next = AtomicUsize::new(0);
        let n_routes = self.rs_offsets.len() - 1;
        let work = || {
            let mut sc = Scratch::new(self.n_stops(), self.rs_stop.len(), n_routes);
            let mut out: Vec<(usize, Answer)> = Vec::new();
            loop {
                let g = next.fetch_add(1, AtomicOrdering::Relaxed);
                if g >= groups.len() {
                    break;
                }
                let group = groups[g];
                let q0 = group[0];
                let dests: Vec<(f64, f64)> = group.iter().map(|&q| (dx[q], dy[q])).collect();
                let answers = self.search(&mut sc, ox[q0], oy[q0], t[q0], &dests, cost(q0));
                out.extend(group.iter().copied().zip(answers));
            }
            out
        };
        let mut answers: Vec<(usize, Answer)> = if threads == 1 {
            work()
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads).map(|_| scope.spawn(&work)).collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            })
        };
        answers.sort_unstable_by_key(|(q, _)| *q);
        answers.into_iter().map(|(_, a)| a).collect()
    }
}

fn to_usize(name: &str, a: PyReadonlyArray1<i64>, bound: usize) -> PyResult<Vec<usize>> {
    a.as_array()
        .iter()
        .map(|&v| {
            if v < 0 || v as usize > bound {
                Err(PyValueError::new_err(format!("{name}: index {v} out of range")))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

fn floats(a: PyReadonlyArray1<f64>) -> Vec<f64> {
    a.as_array().to_vec()
}

#[pymethods]
impl Raptor {
    #[new]
    #[pyo3(signature = (stop_x, stop_y, rs_offsets, rs_stop, rs_distance, n_trips, arr, dep, cols,
                        stop_rs_offsets, stop_rs_route, stop_rs_pos, tr_offsets, tr_to, tr_walk,
                        tr_time, tr_dist, *, walk_speed, walk_factor, radius, extension,
                        max_transfers, after_first, c_travel, c_wait, c_walk, c_switch))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        stop_x: PyReadonlyArray1<f64>,
        stop_y: PyReadonlyArray1<f64>,
        rs_offsets: PyReadonlyArray1<i64>,
        rs_stop: PyReadonlyArray1<i64>,
        rs_distance: PyReadonlyArray1<f64>,
        n_trips: PyReadonlyArray1<i64>,
        arr: PyReadonlyArray1<f64>,
        dep: PyReadonlyArray1<f64>,
        cols: PyReadonlyArray1<f64>,
        stop_rs_offsets: PyReadonlyArray1<i64>,
        stop_rs_route: PyReadonlyArray1<i64>,
        stop_rs_pos: PyReadonlyArray1<i64>,
        tr_offsets: PyReadonlyArray1<i64>,
        tr_to: PyReadonlyArray1<i64>,
        tr_walk: PyReadonlyArray1<f64>,
        tr_time: PyReadonlyArray1<f64>,
        tr_dist: PyReadonlyArray1<f64>,
        walk_speed: f64,
        walk_factor: f64,
        radius: f64,
        extension: f64,
        max_transfers: usize,
        after_first: usize,
        c_travel: f64,
        c_wait: f64,
        c_walk: f64,
        c_switch: f64,
    ) -> PyResult<Raptor> {
        let n_stops = stop_x.as_array().len();
        let n_rs = rs_stop.as_array().len();
        let n_times = arr.as_array().len();
        let rs_offsets = to_usize("rs_offsets", rs_offsets, n_rs)?;
        let n_routes = rs_offsets.len().saturating_sub(1);
        let n_trips = to_usize("n_trips", n_trips, usize::MAX >> 1)?;
        if rs_offsets.is_empty() || n_trips.len() != n_routes {
            return Err(PyValueError::new_err("rs_offsets / n_trips: wrong length"));
        }
        let mut time_base = Vec::with_capacity(n_routes);
        let mut total = 0usize;
        for r in 0..n_routes {
            time_base.push(total);
            total += n_trips[r] * (rs_offsets[r + 1] - rs_offsets[r]);
        }
        if total != n_times || dep.as_array().len() != n_times || cols.as_array().len() != n_times {
            return Err(PyValueError::new_err("arr / dep / cols: wrong length"));
        }
        let routes = to_usize("stop_rs_route", stop_rs_route, n_routes)?;
        let pos = to_usize("stop_rs_pos", stop_rs_pos, n_rs)?;
        let tr_to = to_usize("tr_to", tr_to, n_stops)?;
        Ok(Raptor {
            stop_x: floats(stop_x),
            stop_y: floats(stop_y),
            rs_offsets,
            rs_stop: to_usize("rs_stop", rs_stop, n_stops)?,
            rs_distance: floats(rs_distance),
            n_trips,
            time_base,
            arr: floats(arr),
            dep: floats(dep),
            cols: floats(cols),
            stop_rs_offsets: to_usize("stop_rs_offsets", stop_rs_offsets, routes.len())?,
            stop_rs: routes.into_iter().zip(pos).collect(),
            tr_offsets: to_usize("tr_offsets", tr_offsets, tr_to.len())?,
            tr_to,
            tr_walk: floats(tr_walk),
            tr_time: floats(tr_time),
            tr_dist: floats(tr_dist),
            walk_speed,
            walk_factor,
            radius,
            extension,
            max_transfers,
            after_first,
            c_travel,
            c_wait,
            c_walk,
            c_switch,
        })
    }

    /// Answer queries between coordinates. Returns `(ride_offsets, rides,
    /// travel_time, distance)`: query `q`'s rides are
    /// `rides[2 * ride_offsets[q] .. 2 * ride_offsets[q + 1]]`, packed
    /// `(board route stop, alight route stop)` int32 pairs. `costs`, if
    /// given, is `(n_queries, 4)`: each query's travel, wait and walk cost
    /// per second and cost per transfer (else the router's own).
    #[pyo3(signature = (origin_x, origin_y, departure_times, dest_x, dest_y, threads=1, costs=None))]
    #[allow(clippy::too_many_arguments)]
    fn route_many<'py>(
        &self,
        py: Python<'py>,
        origin_x: PyReadonlyArray1<f64>,
        origin_y: PyReadonlyArray1<f64>,
        departure_times: PyReadonlyArray1<f64>,
        dest_x: PyReadonlyArray1<f64>,
        dest_y: PyReadonlyArray1<f64>,
        threads: usize,
        costs: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (ox, oy, t) = (floats(origin_x), floats(origin_y), floats(departure_times));
        let (dx, dy) = (floats(dest_x), floats(dest_y));
        let nq = ox.len();
        if oy.len() != nq || t.len() != nq || dx.len() != nq || dy.len() != nq {
            return Err(PyValueError::new_err("query arrays must have equal lengths"));
        }
        let costs: Option<Vec<[f64; 4]>> = match costs {
            None => None,
            Some(c) => {
                let c = c.as_array();
                if c.shape() != [nq, 4] {
                    return Err(PyValueError::new_err("costs must have shape (n_queries, 4)"));
                }
                Some(c.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3]]).collect())
            }
        };
        let answers = py.allow_threads(|| self.solve(&ox, &oy, &t, &dx, &dy, costs.as_deref(), threads));
        let mut offsets = Vec::with_capacity(nq + 1);
        let mut rides: Vec<i32> = Vec::new();
        let mut tt = Vec::with_capacity(nq);
        let mut dist = Vec::with_capacity(nq);
        offsets.push(0i64);
        for a in answers {
            for (b, al) in a.rides {
                rides.push(b as i32);
                rides.push(al as i32);
            }
            offsets.push((rides.len() / 2) as i64);
            tt.push(a.travel_time);
            dist.push(a.distance);
        }
        PyTuple::new(py, [
            offsets.into_pyarray(py).into_any(),
            rides.into_pyarray(py).into_any(),
            tt.into_pyarray(py).into_any(),
            dist.into_pyarray(py).into_any(),
        ])
    }
}
