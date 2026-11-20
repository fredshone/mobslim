//! Compiled routers for mobslim.
//!
//! Two algorithms over the same time-dependent link costs:
//!
//! * `dijkstra`: a port of `mobslim.routing.dijkstra.DijkstraRouter` with the
//!   same heap order (by `(time, node)`), the same strict-improvement rule and
//!   the same arithmetic, so it returns *identical* routes. `route_many`
//!   groups queries that share an origin and departure time (for a static
//!   table: an origin) and answers each group with one search that stops once
//!   every destination is settled. That gives the same routes as one search
//!   per query.
//! * `alt`: A* with landmark lower bounds (ALT). Landmark distances are
//!   computed once on a lower-bound cost per link (free-speed times, which
//!   `travel_time_table` guarantees every learned time is at least), so the
//!   bounds stay valid for every later table. Same least costs as Dijkstra;
//!   paths may differ where several routes tie.
//!
//! Both minimise travel time by default. With a distance cost
//! (`with_times(.., cost_time, cost_distance)`) they minimise MATSim's
//! `RandomizingTimeDistanceTravelDisutility` instead: per link,
//! `cost_time * travel_time + cost_distance * factor * length`, `factor`
//! being the query's (a per-person random draw in MATSim). The search then
//! keeps two labels per node, cost (the order) and arrival time (the time
//! bin of the next link); ALT bounds the cost by
//! `cost_time * time bound + cost_distance * factor * distance bound`, from
//! landmarks on free-speed times and on lengths, valid for any factor.
//! `with_times(.., bound_factor=f)` bounds with factor `f` instead of the
//! query's, as MATSim's A* does (`f = 1`, from
//! `getLinkMinimumTravelDisutility`): an overestimate for a query factor
//! below `f`, so ALT then returns MATSim's not-least-cost routes.
//!
//! Travel times are stored as distinct *columns* (one per distinct time bin)
//! plus a bin -> column map, so the free-flow bins of a day share one column.
//! `route_many` releases the GIL and spreads work over `threads` threads.
//!
//! `raptor` holds the least-cost transit router (a port of
//! `mobslim.transit.raptor.RaptorRouter`); `points` the nearest-point index
//! that snaps activities to links (`mobslim.activity_links`); `traversals`
//! the link traversal sums that travel-time tables are learned from.

mod points;
mod raptor;
mod traversals;

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const NO_LINK: u32 = u32::MAX;

/// f64 with a total order (times are never NaN).
#[derive(Clone, Copy, PartialEq)]
struct F(f64);

impl Eq for F {}

impl PartialOrd for F {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Python's float floor division (`a // b`), bit for bit.
fn py_floordiv(vx: f64, wx: f64) -> f64 {
    let m = vx % wx; // C fmod, like CPython
    let mut div = (vx - m) / wx;
    if m != 0.0 && ((wx < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 {
            f + 1.0
        } else {
            f
        }
    } else {
        0.0_f64.copysign(vx / wx)
    }
}

/// `py_floordiv(t, w)`, without its `fmod` unless `t / w` is close to an
/// integer: elsewhere the rounded quotient has the same floor as the exact
/// one (its error is far below 1e-9 for any quotient below 2^20).
#[inline]
fn bin_of(t: f64, w: f64) -> f64 {
    let q = t / w;
    let f = q.floor();
    let frac = q - f;
    if q.abs() < 1048576.0 && frac > 1e-9 && frac < 1.0 - 1e-9 {
        f
    } else {
        py_floordiv(t, w)
    }
}

#[derive(Clone)]
struct Csr {
    offsets: Vec<usize>,
    link: Vec<u32>,
    to: Vec<u32>,
}

impl Csr {
    fn edges(&self, u: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        let (a, b) = (self.offsets[u], self.offsets[u + 1]);
        self.link[a..b].iter().copied().zip(self.to[a..b].iter().copied())
    }
}

struct Landmarks {
    k: usize,
    /// `from_l[v * k + i]`: lower-bound cost from landmark `i` to `v`
    /// (node-major, so one node's bounds share a cache line).
    from_l: Vec<f64>,
    /// `to_l[v * k + i]`: lower-bound cost from `v` to landmark `i`.
    to_l: Vec<f64>,
    nodes: Vec<u32>,
    /// every node reaches and is reached by every landmark (no infinities)
    finite: bool,
}

/// A query's target, as [`Landmarks::bound`] needs it.
struct Target {
    /// `to_l` and `from_l` of the target
    to_l: Vec<f64>,
    from_l: Vec<f64>,
}

/// Static Dijkstra (all nodes) on `cost`, returning distances.
fn static_tree(csr: &Csr, n: usize, cost: &[f64], source: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    dist[source] = 0.0;
    heap.push(Reverse((F(0.0), source as u32)));
    while let Some(Reverse((F(d), u))) = heap.pop() {
        let u = u as usize;
        if d > dist[u] {
            continue;
        }
        for (l, v) in csr.edges(u) {
            let nd = d + cost[l as usize];
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(Reverse((F(nd), v)));
            }
        }
    }
    dist
}

impl Landmarks {
    /// Farthest-point landmark selection on the lower-bound costs.
    fn build(fwd: &Csr, bwd: &Csr, n: usize, cost: &[f64], k: usize) -> Landmarks {
        let k = k.min(n);
        let mut nodes = Vec::with_capacity(k);
        let mut from_l: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut to_l: Vec<Vec<f64>> = Vec::with_capacity(k);
        // separation of each node from the chosen landmarks (both directions)
        let mut sep = vec![f64::INFINITY; n];
        // seed: the node farthest (by finite round-trip) from the first node
        // on the mode's links (in a network of several modes, node 0 may
        // touch only other modes' links, e.g. a pt stop: nothing is reachable
        // from it)
        let start = (0..n)
            .find(|&v| fwd.offsets[v + 1] > fwd.offsets[v] || bwd.offsets[v + 1] > bwd.offsets[v])
            .unwrap_or(0);
        let seed_f = static_tree(fwd, n, cost, start);
        let seed_b = static_tree(bwd, n, cost, start);
        let mut next = argmax_finite(&seed_f, &seed_b, &vec![f64::INFINITY; n]).unwrap_or(0);
        for _ in 0..k {
            nodes.push(next as u32);
            let f = static_tree(fwd, n, cost, next);
            let b = static_tree(bwd, n, cost, next);
            for v in 0..n {
                let s = f[v] + b[v];
                if s < sep[v] {
                    sep[v] = s;
                }
            }
            sep[next] = 0.0;
            from_l.push(f);
            to_l.push(b);
            match argmax_finite(&from_l[from_l.len() - 1], &to_l[to_l.len() - 1], &sep) {
                Some(v) if sep[v] > 0.0 => next = v,
                _ => break,
            }
        }
        let k = nodes.len();
        let node_major = |trees: &[Vec<f64>]| -> Vec<f64> {
            (0..n).flat_map(|v| trees.iter().map(move |tree| tree[v])).collect()
        };
        let (from_l, to_l) = (node_major(&from_l), node_major(&to_l));
        let finite = from_l.iter().chain(&to_l).all(|x| x.is_finite());
        Landmarks { k, from_l, to_l, nodes, finite }
    }

    fn target(&self, t: usize) -> Target {
        let r = t * self.k..(t + 1) * self.k;
        Target { to_l: self.to_l[r.clone()].to_vec(), from_l: self.from_l[r].to_vec() }
    }

    /// Lower bound on the cost from `v` to `t` (`inf` if `t` is unreachable).
    #[inline]
    fn bound(&self, v: usize, t: &Target) -> f64 {
        let k = self.k;
        let (vto, vfrom) = (&self.to_l[v * k..(v + 1) * k], &self.from_l[v * k..(v + 1) * k]);
        let mut best: f64 = 0.0;
        if self.finite {
            for i in 0..k {
                // d(v, t) >= d(v, L) - d(t, L)  and  d(v, t) >= d(L, t) - d(L, v)
                best = best.max(vto[i] - t.to_l[i]).max(t.from_l[i] - vfrom[i]);
            }
        } else {
            for i in 0..k {
                let (vl, tl) = (vto[i], t.to_l[i]);
                if vl.is_infinite() && tl.is_finite() {
                    return f64::INFINITY; // t reaches L but v does not: v cannot reach t
                }
                if vl.is_finite() && tl.is_finite() && vl - tl > best {
                    best = vl - tl;
                }
                let (lt, lv) = (t.from_l[i], vfrom[i]);
                if lt.is_finite() && lv.is_finite() && lt - lv > best {
                    best = lt - lv;
                }
            }
        }
        // shave off rounding so the bound never exceeds the true cost
        best * (1.0 - 1e-9)
    }
}

/// The node with the largest finite `sep` (ties: lowest index), considering
/// only nodes reachable both ways in `f`/`b`.
fn argmax_finite(f: &[f64], b: &[f64], sep: &[f64]) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for v in 0..f.len() {
        if !(f[v].is_finite() && b[v].is_finite()) {
            continue;
        }
        let s = if sep[v].is_finite() { sep[v] } else { f[v] + b[v] };
        if best.map_or(true, |(bs, _)| s > bs) {
            best = Some((s, v));
        }
    }
    best.map(|(_, v)| v)
}

struct Answer {
    query: usize,
    path: Vec<u32>,
    travel_time: f64,
    distance: f64,
    found: bool,
}

/// Per-thread search state, reset lazily with a stamp.
struct Search {
    /// the label: arrival time, or cost with a distance cost
    dist: Vec<f64>,
    /// arrival time, with a distance cost (`dist` is then the cost)
    time: Vec<f64>,
    via: Vec<u32>,
    stamp: Vec<u32>,
    target: Vec<u32>,
    pi: Vec<f64>,
    pi_stamp: Vec<u32>,
    cur: u32,
    heap: BinaryHeap<Reverse<(F, u32)>>,
}

impl Search {
    fn new(n: usize) -> Search {
        Search {
            dist: vec![f64::INFINITY; n],
            time: vec![0.0; n],
            via: vec![NO_LINK; n],
            stamp: vec![0; n],
            target: vec![0; n],
            pi: vec![0.0; n],
            pi_stamp: vec![0; n],
            cur: 0,
            heap: BinaryHeap::new(),
        }
    }

    fn reset(&mut self) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.target.iter_mut().for_each(|s| *s = 0);
            self.pi_stamp.iter_mut().for_each(|s| *s = 0);
            self.cur = 1;
        }
        self.heap.clear();
    }

    #[inline]
    fn get(&self, v: usize) -> f64 {
        if self.stamp[v] == self.cur {
            self.dist[v]
        } else {
            f64::INFINITY
        }
    }

    #[inline]
    fn set(&mut self, v: usize, d: f64, link: u32) {
        self.stamp[v] = self.cur;
        self.dist[v] = d;
        self.via[v] = link;
    }
}

/// The mode's graph in the router's own node numbering (nodes in Hilbert
/// curve order when coordinates are given, so that a search touches nearby
/// memory), shared by every router made `with_times`. Edge `e` (the CSR
/// position, `fwd.link[e] == e`) is the network link `edge_link[e]`; each
/// node's edges keep the network's order.
struct Graph {
    fwd: Csr,
    edge_link: Vec<u32>,
    edge_from: Vec<u32>,
    edge_length: Vec<f64>,
    /// router node of each network node, and network node of each router node
    new_of: Vec<u32>,
    old_of: Vec<u32>,
}

#[pyclass(frozen)]
struct Router {
    n: usize,
    graph: Arc<Graph>,
    n_links: usize,
    /// `columns[c * n_edges + edge]`: link times in edge order
    columns: Vec<f64>,
    bin_col: Vec<u32>,
    bin_size: f64,
    is_static: bool,
    landmarks: Option<Arc<Landmarks>>,
    /// ALT landmarks on link lengths (with a distance cost)
    distance_landmarks: Option<Arc<Landmarks>>,
    /// link cost = cost_time * travel time + cost_distance * factor * length
    cost_time: f64,
    cost_distance: f64,
    /// distance factor ALT's bound assumes (`None`: the query's, exact)
    bound_factor: Option<f64>,
    /// reusable search buffers (one per concurrently running search)
    pool: Mutex<Vec<Search>>,
}

impl Router {
    /// Whether `algorithm` is ALT, if this router can run it.
    fn alt(&self, algorithm: &str) -> PyResult<bool> {
        match algorithm {
            "dijkstra" => Ok(false),
            "alt" if self.landmarks.is_none() => {
                Err(PyValueError::new_err("algorithm 'alt' needs landmarks > 0"))
            }
            "alt" if self.cost_distance > 0.0 && self.distance_landmarks.is_none() => {
                Err(PyValueError::new_err("algorithm 'alt' with a distance cost needs distance_landmarks"))
            }
            "alt" => Ok(true),
            other => Err(PyValueError::new_err(format!("unknown algorithm {other:?}"))),
        }
    }

    fn take(&self) -> Search {
        self.pool.lock().unwrap().pop().unwrap_or_else(|| Search::new(self.n))
    }

    fn give(&self, s: Search) {
        self.pool.lock().unwrap().push(s);
    }

    #[inline]
    fn column(&self, time: f64) -> &[f64] {
        let c = if self.is_static {
            0
        } else {
            let last = self.bin_col.len() as i64 - 1;
            let b = bin_of(time, self.bin_size) as i64;
            self.bin_col[b.clamp(0, last) as usize] as usize
        };
        let ne = self.graph.edge_link.len();
        &self.columns[c * ne..(c + 1) * ne]
    }

    /// Time-dependent Dijkstra from `(origin, t0)` until every node marked
    /// in `s.target` (there are `n_targets`) is settled. The heap breaks
    /// ties by network node (as `DijkstraRouter`), whatever the numbering.
    fn dijkstra(&self, s: &mut Search, origin: usize, t0: f64, mut n_targets: usize) {
        let g = &*self.graph;
        s.set(origin, t0, NO_LINK);
        s.heap.push(Reverse((F(t0), g.old_of[origin])));
        while let Some(Reverse((F(t), u))) = s.heap.pop() {
            let u = g.new_of[u as usize] as usize;
            if t > s.dist[u] {
                continue;
            }
            if s.target[u] == s.cur {
                s.target[u] = 0; // settled
                n_targets -= 1;
                if n_targets == 0 {
                    break;
                }
            }
            let tt = self.column(t);
            for (l, v) in g.fwd.edges(u) {
                let nt = t + tt[l as usize];
                let v = v as usize;
                if nt < s.get(v) {
                    s.set(v, nt, l);
                    s.heap.push(Reverse((F(nt), g.old_of[v])));
                }
            }
        }
    }

    #[inline]
    fn potential(&self, lm: &Landmarks, s: &mut Search, v: usize, target: &Target) -> f64 {
        if s.pi_stamp[v] != s.cur {
            s.pi_stamp[v] = s.cur;
            s.pi[v] = lm.bound(v, target);
        }
        s.pi[v]
    }

    /// Time-dependent A* (ALT) from `(origin, t0)` to `target`.
    /// Ties break by network node, as in [`Router::dijkstra`].
    fn astar(&self, s: &mut Search, origin: usize, t0: f64, target: usize) {
        let g = &*self.graph;
        let lm = self.landmarks.as_deref().unwrap();
        let goal = lm.target(target);
        s.set(origin, t0, NO_LINK);
        let p = self.potential(lm, s, origin, &goal);
        if p.is_infinite() {
            return;
        }
        s.heap.push(Reverse((F(t0 + p), g.old_of[origin])));
        while let Some(Reverse((F(key), u))) = s.heap.pop() {
            let u = g.new_of[u as usize] as usize;
            let t = s.dist[u];
            if key > t + s.pi[u] {
                continue; // stale
            }
            if u == target {
                break;
            }
            let tt = self.column(t);
            for (l, v) in g.fwd.edges(u) {
                let nt = t + tt[l as usize];
                let v = v as usize;
                if nt < s.get(v) {
                    let p = self.potential(lm, s, v, &goal);
                    if p.is_infinite() {
                        continue;
                    }
                    s.set(v, nt, l);
                    s.heap.push(Reverse((F(nt + p), g.old_of[v])));
                }
            }
        }
    }

    /// [`Router::dijkstra`] on the generalized cost (`bf`: `cost_distance *
    /// factor`): the heap orders by `(cost, network node)`, and each node also
    /// keeps its arrival time, whose bin times its out-links.
    fn dijkstra_cost(&self, s: &mut Search, origin: usize, t0: f64, bf: f64, mut n_targets: usize) {
        let g = &*self.graph;
        let a = self.cost_time;
        s.set(origin, 0.0, NO_LINK);
        s.time[origin] = t0;
        s.heap.push(Reverse((F(0.0), g.old_of[origin])));
        while let Some(Reverse((F(c), u))) = s.heap.pop() {
            let u = g.new_of[u as usize] as usize;
            if c > s.dist[u] {
                continue;
            }
            if s.target[u] == s.cur {
                s.target[u] = 0; // settled
                n_targets -= 1;
                if n_targets == 0 {
                    break;
                }
            }
            let t = s.time[u];
            let tt = self.column(t);
            for (l, v) in g.fwd.edges(u) {
                let x = tt[l as usize];
                let nc = c + (a * x + bf * g.edge_length[l as usize]);
                let v = v as usize;
                if nc < s.get(v) {
                    s.set(v, nc, l);
                    s.time[v] = t + x;
                    s.heap.push(Reverse((F(nc), g.old_of[v])));
                }
            }
        }
    }

    #[inline]
    fn potential_cost(&self, lm: (&Landmarks, &Landmarks), s: &mut Search, v: usize,
                      goal: (&Target, &Target), bf: f64) -> f64 {
        if s.pi_stamp[v] != s.cur {
            s.pi_stamp[v] = s.cur;
            let time = lm.0.bound(v, goal.0);
            s.pi[v] = if time.is_infinite() {
                f64::INFINITY
            } else {
                self.cost_time * time + bf * lm.1.bound(v, goal.1)
            };
        }
        s.pi[v]
    }

    /// [`Router::astar`] on the generalized cost.
    /// `bb`: the bound's distance cost per metre (`bf` unless
    /// `bound_factor` is set).
    #[allow(clippy::too_many_arguments)]
    fn astar_cost(&self, s: &mut Search, origin: usize, t0: f64, bf: f64, bb: f64, target: usize) {
        let g = &*self.graph;
        let a = self.cost_time;
        let lm = (self.landmarks.as_deref().unwrap(), self.distance_landmarks.as_deref().unwrap());
        let goal = (lm.0.target(target), lm.1.target(target));
        let goal = (&goal.0, &goal.1);
        s.set(origin, 0.0, NO_LINK);
        s.time[origin] = t0;
        let p = self.potential_cost(lm, s, origin, goal, bb);
        if p.is_infinite() {
            return;
        }
        s.heap.push(Reverse((F(p), g.old_of[origin])));
        while let Some(Reverse((F(key), u))) = s.heap.pop() {
            let u = g.new_of[u as usize] as usize;
            let c = s.dist[u];
            if key > c + s.pi[u] {
                continue; // stale
            }
            if u == target {
                break;
            }
            let t = s.time[u];
            let tt = self.column(t);
            for (l, v) in g.fwd.edges(u) {
                let x = tt[l as usize];
                let nc = c + (a * x + bf * g.edge_length[l as usize]);
                let v = v as usize;
                if nc < s.get(v) {
                    let p = self.potential_cost(lm, s, v, goal, bb);
                    if p.is_infinite() {
                        continue;
                    }
                    s.set(v, nc, l);
                    s.time[v] = t + x;
                    s.heap.push(Reverse((F(nc + p), g.old_of[v])));
                }
            }
        }
    }

    fn answer(&self, s: &Search, query: usize, origin: usize, dest: usize, t0: f64) -> Answer {
        if s.get(dest).is_infinite() {
            return Answer { query, path: Vec::new(), travel_time: 0.0, distance: 0.0, found: false };
        }
        let g = &*self.graph;
        let mut edges = Vec::new();
        let mut node = dest;
        while node != origin {
            let e = s.via[node];
            edges.push(e);
            node = g.edge_from[e as usize] as usize;
        }
        edges.reverse();
        let mut distance = 0.0;
        for &e in &edges {
            distance += g.edge_length[e as usize];
        }
        let path = edges.iter().map(|&e| g.edge_link[e as usize]).collect();
        let arrival = if self.cost_distance > 0.0 { s.time[dest] } else { s.get(dest) };
        Answer { query, path, travel_time: arrival - t0, distance, found: true }
    }

    /// Answer queries `group` (indices into the arrays), which share an
    /// origin, departure time and factor.
    #[allow(clippy::too_many_arguments)]
    fn solve_group(&self, s: &mut Search, group: &[usize], o: &[u32], d: &[u32], t: &[f64],
                   f: Option<&[f64]>, alt: bool, out: &mut Vec<Answer>) {
        let origin = o[group[0]] as usize;
        let t0 = if self.is_static { 0.0 } else { t[group[0]] };
        let cost = self.cost_distance > 0.0;
        let bf = self.cost_distance * f.map_or(1.0, |f| f[group[0]]);
        let bb = self.bound_factor.map_or(bf, |b| self.cost_distance * b);
        if alt {
            for &q in group {
                let dest = d[q] as usize;
                if dest == origin {
                    out.push(Answer { query: q, path: Vec::new(), travel_time: 0.0, distance: 0.0, found: true });
                    continue;
                }
                s.reset();
                if cost {
                    self.astar_cost(s, origin, t0, bf, bb, dest);
                } else {
                    self.astar(s, origin, t0, dest);
                }
                out.push(self.answer(s, q, origin, dest, t0));
            }
            return;
        }
        s.reset();
        let mut n_targets = 0;
        for &q in group {
            let dest = d[q] as usize;
            if dest != origin && s.target[dest] != s.cur {
                s.target[dest] = s.cur;
                n_targets += 1;
            }
        }
        if n_targets > 0 {
            if cost {
                self.dijkstra_cost(s, origin, t0, bf, n_targets);
            } else {
                self.dijkstra(s, origin, t0, n_targets);
            }
        }
        for &q in group {
            let dest = d[q] as usize;
            if dest == origin {
                out.push(Answer { query: q, path: Vec::new(), travel_time: 0.0, distance: 0.0, found: true });
            } else {
                out.push(self.answer(s, q, origin, dest, t0));
            }
        }
    }

    fn solve(&self, o: &[u32], d: &[u32], t: &[f64], f: Option<&[f64]>, alt: bool,
             threads: usize) -> Vec<Answer> {
        let nq = o.len();
        // group by (origin, departure, factor) -- or without departure for a
        // static table, and without factor without a distance cost
        let mut order: Vec<usize> = (0..nq).collect();
        let is_static = self.is_static;
        let f = if self.cost_distance > 0.0 { f } else { None };
        let fq = |q: usize| f.map_or(1.0, |f| f[q]);
        order.sort_by(|&a, &b| {
            o[a].cmp(&o[b])
                .then_with(|| if is_static { Ordering::Equal } else { t[a].total_cmp(&t[b]) })
                .then_with(|| fq(a).total_cmp(&fq(b)))
                .then(a.cmp(&b))
        });
        let mut groups: Vec<&[usize]> = Vec::new();
        let mut start = 0;
        for i in 1..=nq {
            if i == nq || o[order[i]] != o[order[start]]
                || (!is_static && t[order[i]].to_bits() != t[order[start]].to_bits())
                || fq(order[i]).to_bits() != fq(order[start]).to_bits()
            {
                if start < i {
                    groups.push(&order[start..i]);
                }
                start = i;
            }
        }
        let threads = threads.max(1).min(groups.len().max(1));
        let next = AtomicUsize::new(0);
        let work = || {
            let mut s = self.take();
            let mut out = Vec::new();
            loop {
                let g = next.fetch_add(1, AtomicOrdering::Relaxed);
                if g >= groups.len() {
                    break;
                }
                self.solve_group(&mut s, groups[g], o, d, t, f, alt, &mut out);
            }
            self.give(s);
            out
        };
        let mut answers: Vec<Answer> = if threads == 1 {
            work()
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads).map(|_| scope.spawn(&work)).collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            })
        };
        answers.sort_unstable_by_key(|a| a.query);
        answers
    }
}

fn to_u32(name: &str, a: &[i64], bound: usize) -> PyResult<Vec<u32>> {
    a.iter()
        .map(|&x| {
            if x < 0 || x as usize >= bound {
                Err(PyValueError::new_err(format!("{name}: index {x} out of range 0..{bound}")))
            } else {
                Ok(x as u32)
            }
        })
        .collect()
}

#[allow(clippy::type_complexity)]
fn parse_columns(n_links: usize, columns: PyReadonlyArray2<f64>, bin_col: PyReadonlyArray1<i64>,
                 bin_size: f64) -> PyResult<(Vec<f64>, usize, Vec<u32>)> {
    let cols = columns.as_array();
    if cols.ncols() != n_links || cols.nrows() == 0 {
        return Err(PyValueError::new_err("columns must have shape (n_columns >= 1, n_links)"));
    }
    let n_cols = cols.nrows();
    let columns: Vec<f64> = cols.iter().copied().collect(); // row-major
    let bin_col = to_u32("bin_col", bin_col.as_slice()?, n_cols)?;
    if bin_col.is_empty() {
        return Err(PyValueError::new_err("bin_col must not be empty"));
    }
    if !(bin_size > 0.0) {
        return Err(PyValueError::new_err("bin_size must be positive"));
    }
    Ok((columns, n_cols, bin_col))
}

/// Position of `(x, y)` along a Hilbert curve over a `2^BITS` grid.
fn hilbert_key(mut x: u32, mut y: u32) -> u64 {
    const BITS: u32 = 16;
    let full = (1u32 << BITS) - 1;
    let mut d = 0u64;
    let mut s = 1u32 << (BITS - 1);
    while s > 0 {
        let rx = (x & s) > 0;
        let ry = (y & s) > 0;
        d += (s as u64) * (s as u64) * ((3 * rx as u64) ^ ry as u64);
        if !ry {
            if rx {
                x = full - x;
                y = full - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

/// Network nodes in Hilbert curve order of their coordinates (ties, and
/// nodes without finite coordinates, by node index).
fn hilbert_order(x: &[f64], y: &[f64]) -> Vec<u32> {
    let finite = |v: &&f64| v.is_finite();
    let lo = |a: &[f64]| a.iter().filter(finite).copied().fold(f64::INFINITY, f64::min);
    let hi = |a: &[f64]| a.iter().filter(finite).copied().fold(f64::NEG_INFINITY, f64::max);
    let (x0, y0) = (lo(x), lo(y));
    let span = (hi(x) - x0).max(hi(y) - y0).max(1e-9);
    let cell = |v: f64, v0: f64| -> u32 {
        if v.is_finite() { (((v - v0) / span) * 65535.0).clamp(0.0, 65535.0) as u32 } else { 0 }
    };
    let mut order: Vec<u32> = (0..x.len() as u32).collect();
    let keys: Vec<u64> = (0..x.len()).map(|i| hilbert_key(cell(x[i], x0), cell(y[i], y0))).collect();
    order.sort_by_key(|&i| (keys[i as usize], i));
    order
}

impl Graph {
    /// The mode's links (`offsets`/`link`, a CSR over network nodes) with
    /// the network nodes renumbered in `order` (`order[new] = network node`).
    fn build(offsets: &[usize], link: &[u32], to_node: &[u32], length: &[f64], order: Vec<u32>)
             -> Graph {
        let n = order.len();
        let mut new_of = vec![0u32; n];
        for (w, &u) in order.iter().enumerate() {
            new_of[u as usize] = w as u32;
        }
        let mut fwd_offsets = Vec::with_capacity(n + 1);
        let (mut edge_link, mut edge_from, mut to) = (Vec::new(), Vec::new(), Vec::new());
        fwd_offsets.push(0);
        for (w, &u) in order.iter().enumerate() {
            for &l in &link[offsets[u as usize]..offsets[u as usize + 1]] {
                edge_link.push(l);
                edge_from.push(w as u32);
                to.push(new_of[to_node[l as usize] as usize]);
            }
            fwd_offsets.push(edge_link.len());
        }
        let edge_length = edge_link.iter().map(|&l| length[l as usize]).collect();
        let fwd = Csr { offsets: fwd_offsets, link: (0..edge_link.len() as u32).collect(), to };
        Graph { fwd, edge_link, edge_from, edge_length, new_of, old_of: order }
    }

    /// The reverse adjacency (edges by their to-node).
    fn reverse(&self) -> Csr {
        let n = self.new_of.len();
        let mut counts = vec![0usize; n + 1];
        for &v in &self.fwd.to {
            counts[v as usize + 1] += 1;
        }
        for i in 0..n {
            counts[i + 1] += counts[i];
        }
        let mut fill = counts.clone();
        let ne = self.edge_link.len();
        let (mut rlink, mut rto) = (vec![0u32; ne], vec![0u32; ne]);
        for e in 0..ne {
            let v = self.fwd.to[e] as usize;
            rlink[fill[v]] = e as u32;
            rto[fill[v]] = self.edge_from[e];
            fill[v] += 1;
        }
        Csr { offsets: counts, link: rlink, to: rto }
    }

    /// Link-major `(n_cols, n_links)` values as edge-major `(n_cols, n_edges)`.
    fn edge_columns(&self, columns: &[f64], n_links: usize, n_cols: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n_cols * self.edge_link.len());
        for c in 0..n_cols {
            let col = &columns[c * n_links..(c + 1) * n_links];
            out.extend(self.edge_link.iter().map(|&l| col[l as usize]));
        }
        out
    }
}

#[pymethods]
impl Router {
    /// `offsets`/`links`: the mode's CSR adjacency (`Network.out_links`).
    /// `columns`: `(n_columns, n_links)` link travel times; `bin_col[b]` is the
    /// column of time bin `b`. `lower_bound`: per-link lower bound on any
    /// column (free-speed times); with `landmarks > 0` ALT landmarks are
    /// computed on it. `node_xy`: `(n_nodes, 2)` coordinates; given, the
    /// router numbers nodes along a Hilbert curve internally (faster
    /// searches on networks numbered without regard to location; nodes,
    /// links and routes in and out stay the network's).
    #[new]
    #[pyo3(signature = (offsets, links, from_node, to_node, length, columns, bin_col, bin_size,
                        lower_bound=None, landmarks=0, node_xy=None, distance_landmarks=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        offsets: PyReadonlyArray1<i64>,
        links: PyReadonlyArray1<i64>,
        from_node: PyReadonlyArray1<i64>,
        to_node: PyReadonlyArray1<i64>,
        length: PyReadonlyArray1<f64>,
        columns: PyReadonlyArray2<f64>,
        bin_col: PyReadonlyArray1<i64>,
        bin_size: f64,
        lower_bound: Option<PyReadonlyArray1<f64>>,
        landmarks: usize,
        node_xy: Option<PyReadonlyArray2<f64>>,
        distance_landmarks: bool,
    ) -> PyResult<Router> {
        let offsets = offsets.as_slice()?;
        let n = offsets.len().checked_sub(1).ok_or_else(|| PyValueError::new_err("empty offsets"))?;
        let n_links = from_node.len()?;
        to_u32("from_node", from_node.as_slice()?, n)?; // validated; edges come from offsets/links
        let to_node = to_u32("to_node", to_node.as_slice()?, n)?;
        let link = to_u32("links", links.as_slice()?, n_links)?;
        let length = length.as_slice()?.to_vec();
        if length.len() != n_links || to_node.len() != n_links {
            return Err(PyValueError::new_err("link arrays must have the same length"));
        }
        let (columns, n_cols, bin_col) = parse_columns(n_links, columns, bin_col, bin_size)?;
        let offsets: Vec<usize> = offsets.iter().map(|&x| x as usize).collect();
        if offsets[n] != link.len() {
            return Err(PyValueError::new_err("offsets do not match links"));
        }
        let order = match node_xy {
            Some(xy) => {
                let xy = xy.as_array();
                if xy.nrows() != n || xy.ncols() != 2 {
                    return Err(PyValueError::new_err("node_xy must have shape (n_nodes, 2)"));
                }
                let (x, y): (Vec<f64>, Vec<f64>) = xy.rows().into_iter().map(|r| (r[0], r[1])).unzip();
                hilbert_order(&x, &y)
            }
            None => (0..n as u32).collect(),
        };
        let graph = Graph::build(&offsets, &link, &to_node, &length, order);
        let lower = match lower_bound {
            Some(a) => Some(a.as_slice()?.to_vec()),
            None => None,
        };
        let n_landmarks = landmarks;
        let landmarks = if landmarks > 0 {
            let lower = lower.ok_or_else(|| PyValueError::new_err("landmarks need lower_bound"))?;
            if lower.len() != n_links {
                return Err(PyValueError::new_err("lower_bound must have n_links entries"));
            }
            let lower: Vec<f64> = graph.edge_link.iter().map(|&l| lower[l as usize]).collect();
            let bwd = graph.reverse();
            let g = &graph;
            Some(Arc::new(py.allow_threads(|| Landmarks::build(&g.fwd, &bwd, n, &lower, landmarks))))
        } else {
            None
        };
        // landmarks on lengths, for the distance part of a generalized cost
        let distance_landmarks = if n_landmarks > 0 && distance_landmarks {
            let bwd = graph.reverse();
            let g = &graph;
            Some(Arc::new(py.allow_threads(|| {
                Landmarks::build(&g.fwd, &bwd, n, &g.edge_length, n_landmarks)
            })))
        } else {
            None
        };
        let columns = graph.edge_columns(&columns, n_links, n_cols);
        Ok(Router {
            n,
            graph: Arc::new(graph),
            n_links,
            columns,
            is_static: n_cols == 1,
            bin_col,
            bin_size,
            landmarks,
            distance_landmarks,
            cost_time: 1.0,
            cost_distance: 0.0,
            bound_factor: None,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// The same graph and landmarks with other travel times (cheap: the
    /// landmarks are shared, not recomputed). The lower bound the landmarks
    /// were built on must still hold for the new times. `cost_time` and
    /// `cost_distance` (both >= 0) set the link cost; a distance cost with ALT
    /// needs `distance_landmarks`. `bound_factor`: the distance factor ALT's
    /// bound assumes (default: each query's own, which keeps routes least-cost).
    #[pyo3(signature = (columns, bin_col, bin_size, cost_time=1.0, cost_distance=0.0,
                        bound_factor=None))]
    fn with_times(&self, columns: PyReadonlyArray2<f64>, bin_col: PyReadonlyArray1<i64>,
                  bin_size: f64, cost_time: f64, cost_distance: f64,
                  bound_factor: Option<f64>) -> PyResult<Router> {
        if !(cost_time >= 0.0 && cost_distance >= 0.0 && cost_time + cost_distance > 0.0) {
            return Err(PyValueError::new_err("costs must be >= 0, and not both 0"));
        }
        if bound_factor.is_some_and(|b| !(b >= 0.0 && b.is_finite())) {
            return Err(PyValueError::new_err("bound_factor must be finite and >= 0"));
        }
        let (columns, n_cols, bin_col) = parse_columns(self.n_links, columns, bin_col, bin_size)?;
        let columns = self.graph.edge_columns(&columns, self.n_links, n_cols);
        Ok(Router {
            n: self.n,
            graph: self.graph.clone(),
            n_links: self.n_links,
            columns,
            is_static: n_cols == 1,
            bin_col,
            bin_size,
            landmarks: self.landmarks.clone(),
            distance_landmarks: self.distance_landmarks.clone(),
            cost_time,
            cost_distance,
            bound_factor,
            pool: Mutex::new(Vec::new()),
        })
    }

    #[getter]
    fn landmark_nodes(&self) -> Vec<u32> {
        let old_of = &self.graph.old_of;
        self.landmarks.as_ref().map_or_else(Vec::new, |l| l.nodes.iter().map(|&v| old_of[v as usize]).collect())
    }

    #[getter]
    fn is_static(&self) -> bool {
        self.is_static
    }

    /// Route every query. Returns `(offsets, links, travel_time, distance,
    /// found)`: query `i`'s links are `links[offsets[i]:offsets[i + 1]]`
    /// (int32, so `links.tobytes()` slices straight into packed routes).
    /// `factors`: each query's distance-cost factor (default 1).
    #[pyo3(signature = (origins, destinations, departure_times, algorithm="dijkstra", threads=1,
                        factors=None))]
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn route_many<'py>(
        &self,
        py: Python<'py>,
        origins: PyReadonlyArray1<i64>,
        destinations: PyReadonlyArray1<i64>,
        departure_times: PyReadonlyArray1<f64>,
        algorithm: &str,
        threads: usize,
        factors: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        let alt = self.alt(algorithm)?;
        let new_of = &self.graph.new_of;
        let o: Vec<u32> = to_u32("origins", origins.as_slice()?, self.n)?.iter().map(|&v| new_of[v as usize]).collect();
        let d: Vec<u32> = to_u32("destinations", destinations.as_slice()?, self.n)?.iter().map(|&v| new_of[v as usize]).collect();
        let t = departure_times.as_slice()?.to_vec();
        let f = match factors {
            Some(f) => Some(f.as_slice()?.to_vec()),
            None => None,
        };
        if d.len() != o.len() || t.len() != o.len() || f.as_ref().is_some_and(|f| f.len() != o.len()) {
            return Err(PyValueError::new_err("query arrays must have the same length"));
        }
        let answers = py.allow_threads(|| self.solve(&o, &d, &t, f.as_deref(), alt, threads));
        let mut offsets = Vec::with_capacity(answers.len() + 1);
        let mut links = Vec::new();
        let mut tt = Vec::with_capacity(answers.len());
        let mut dist = Vec::with_capacity(answers.len());
        let mut found = Vec::with_capacity(answers.len());
        offsets.push(0i64);
        for a in &answers {
            links.extend(a.path.iter().map(|&l| l as i32));
            offsets.push(links.len() as i64);
            tt.push(a.travel_time);
            dist.push(a.distance);
            found.push(a.found);
        }
        Ok((
            offsets.into_pyarray(py),
            links.into_pyarray(py),
            tt.into_pyarray(py),
            dist.into_pyarray(py),
            found.into_pyarray(py),
        ))
    }

    /// One query: `(links, travel_time, distance)` with the links as packed
    /// native int32 bytes, or `None` if unreachable.
    #[pyo3(signature = (origin, destination, departure_time, algorithm="dijkstra", factor=1.0))]
    fn route<'py>(&self, py: Python<'py>, origin: usize, destination: usize,
                  departure_time: f64, algorithm: &str, factor: f64)
                  -> PyResult<Option<(Bound<'py, PyBytes>, f64, f64)>> {
        if origin >= self.n || destination >= self.n {
            return Err(PyValueError::new_err("node index out of range"));
        }
        let alt = self.alt(algorithm)?;
        let mut s = self.take();
        let mut out = Vec::with_capacity(1);
        let new_of = &self.graph.new_of;
        self.solve_group(&mut s, &[0], &[new_of[origin]], &[new_of[destination]],
                         &[departure_time], Some(&[factor]), alt, &mut out);
        self.give(s);
        let a = out.pop().unwrap();
        if !a.found {
            return Ok(None);
        }
        let bytes: Vec<u8> = a.path.iter().flat_map(|&l| (l as i32).to_ne_bytes()).collect();
        Ok(Some((PyBytes::new(py, &bytes), a.travel_time, a.distance)))
    }
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Router>()?;
    m.add_class::<raptor::Raptor>()?;
    m.add_class::<points::PointIndex>()?;
    m.add_function(wrap_pyfunction!(traversals::accumulate_traversals, m)?)?;
    Ok(())
}
