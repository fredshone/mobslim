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
//! Travel times are stored as distinct *columns* (one per distinct time bin)
//! plus a bin -> column map, so the free-flow bins of a day share one column.
//! `route_many` releases the GIL and spreads work over `threads` threads.
//!
//! `raptor` holds the least-cost transit router (a port of
//! `mobslim.transit.raptor.RaptorRouter`).

mod raptor;

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
    /// `from_l[i * n + v]`: lower-bound cost from landmark `i` to `v`.
    from_l: Vec<f64>,
    /// `to_l[i * n + v]`: lower-bound cost from `v` to landmark `i`.
    to_l: Vec<f64>,
    nodes: Vec<u32>,
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
        let mut from_l = Vec::with_capacity(k * n);
        let mut to_l = Vec::with_capacity(k * n);
        // separation of each node from the chosen landmarks (both directions)
        let mut sep = vec![f64::INFINITY; n];
        // seed: the node farthest (by finite round-trip) from node 0
        let seed_f = static_tree(fwd, n, cost, 0);
        let seed_b = static_tree(bwd, n, cost, 0);
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
            from_l.extend_from_slice(&f);
            to_l.extend_from_slice(&b);
            match argmax_finite(&f, &b, &sep) {
                Some(v) if sep[v] > 0.0 => next = v,
                _ => break,
            }
        }
        Landmarks { k: nodes.len(), from_l, to_l, nodes }
    }

    /// Lower bound on the cost from `v` to `t` (`inf` if `t` is unreachable).
    #[inline]
    fn bound(&self, n: usize, v: usize, t: usize) -> f64 {
        let mut best: f64 = 0.0;
        for i in 0..self.k {
            let o = i * n;
            // d(v, t) >= d(v, L) - d(t, L)  and  d(v, t) >= d(L, t) - d(L, v)
            let (vl, tl) = (self.to_l[o + v], self.to_l[o + t]);
            if vl.is_infinite() && tl.is_finite() {
                return f64::INFINITY; // t reaches L but v does not: v cannot reach t
            }
            if vl.is_finite() && tl.is_finite() && vl - tl > best {
                best = vl - tl;
            }
            let (lt, lv) = (self.from_l[o + t], self.from_l[o + v]);
            if lt.is_finite() && lv.is_finite() && lt - lv > best {
                best = lt - lv;
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
    dist: Vec<f64>,
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

#[pyclass(frozen)]
struct Router {
    n: usize,
    fwd: Csr,
    from_node: Vec<u32>,
    length: Vec<f64>,
    n_links: usize,
    /// `columns[c * n_links + link]`
    columns: Vec<f64>,
    bin_col: Vec<u32>,
    bin_size: f64,
    is_static: bool,
    landmarks: Option<Arc<Landmarks>>,
    /// reusable search buffers (one per concurrently running search)
    pool: Mutex<Vec<Search>>,
}

impl Router {
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
            let b = py_floordiv(time, self.bin_size) as i64;
            self.bin_col[b.clamp(0, last) as usize] as usize
        };
        &self.columns[c * self.n_links..(c + 1) * self.n_links]
    }

    /// Time-dependent Dijkstra from `(origin, t0)` until every node marked
    /// in `s.target` (there are `n_targets`) is settled.
    fn dijkstra(&self, s: &mut Search, origin: usize, t0: f64, mut n_targets: usize) {
        s.set(origin, t0, NO_LINK);
        s.heap.push(Reverse((F(t0), origin as u32)));
        while let Some(Reverse((F(t), u))) = s.heap.pop() {
            let u = u as usize;
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
            for (l, v) in self.fwd.edges(u) {
                let nt = t + tt[l as usize];
                let v = v as usize;
                if nt < s.get(v) {
                    s.set(v, nt, l);
                    s.heap.push(Reverse((F(nt), v as u32)));
                }
            }
        }
    }

    #[inline]
    fn potential(&self, s: &mut Search, v: usize, target: usize) -> f64 {
        if s.pi_stamp[v] != s.cur {
            s.pi_stamp[v] = s.cur;
            s.pi[v] = self.landmarks.as_ref().unwrap().bound(self.n, v, target);
        }
        s.pi[v]
    }

    /// Time-dependent A* (ALT) from `(origin, t0)` to `target`.
    fn astar(&self, s: &mut Search, origin: usize, t0: f64, target: usize) {
        s.set(origin, t0, NO_LINK);
        let p = self.potential(s, origin, target);
        if p.is_infinite() {
            return;
        }
        s.heap.push(Reverse((F(t0 + p), origin as u32)));
        while let Some(Reverse((F(key), u))) = s.heap.pop() {
            let u = u as usize;
            let t = s.dist[u];
            if key > t + s.pi[u] {
                continue; // stale
            }
            if u == target {
                break;
            }
            let tt = self.column(t);
            for (l, v) in self.fwd.edges(u) {
                let nt = t + tt[l as usize];
                let v = v as usize;
                if nt < s.get(v) {
                    let p = self.potential(s, v, target);
                    if p.is_infinite() {
                        continue;
                    }
                    s.set(v, nt, l);
                    s.heap.push(Reverse((F(nt + p), v as u32)));
                }
            }
        }
    }

    fn answer(&self, s: &Search, query: usize, origin: usize, dest: usize, t0: f64) -> Answer {
        if s.get(dest).is_infinite() {
            return Answer { query, path: Vec::new(), travel_time: 0.0, distance: 0.0, found: false };
        }
        let mut path = Vec::new();
        let mut node = dest;
        while node != origin {
            let l = s.via[node];
            path.push(l);
            node = self.from_node[l as usize] as usize;
        }
        path.reverse();
        let mut distance = 0.0;
        for &l in &path {
            distance += self.length[l as usize];
        }
        Answer { query, path, travel_time: s.get(dest) - t0, distance, found: true }
    }

    /// Answer queries `group` (indices into the arrays), which share an
    /// origin and departure time.
    fn solve_group(&self, s: &mut Search, group: &[usize], o: &[u32], d: &[u32], t: &[f64],
                   alt: bool, out: &mut Vec<Answer>) {
        let origin = o[group[0]] as usize;
        let t0 = if self.is_static { 0.0 } else { t[group[0]] };
        if alt {
            for &q in group {
                let dest = d[q] as usize;
                if dest == origin {
                    out.push(Answer { query: q, path: Vec::new(), travel_time: 0.0, distance: 0.0, found: true });
                    continue;
                }
                s.reset();
                self.astar(s, origin, t0, dest);
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
            self.dijkstra(s, origin, t0, n_targets);
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

    fn solve(&self, o: &[u32], d: &[u32], t: &[f64], alt: bool, threads: usize) -> Vec<Answer> {
        let nq = o.len();
        // group by (origin, departure) -- or origin only for a static table
        let mut order: Vec<usize> = (0..nq).collect();
        let is_static = self.is_static;
        order.sort_by(|&a, &b| {
            o[a].cmp(&o[b])
                .then_with(|| if is_static { Ordering::Equal } else { t[a].total_cmp(&t[b]) })
                .then(a.cmp(&b))
        });
        let mut groups: Vec<&[usize]> = Vec::new();
        let mut start = 0;
        for i in 1..=nq {
            if i == nq || o[order[i]] != o[order[start]]
                || (!is_static && t[order[i]].to_bits() != t[order[start]].to_bits())
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
                self.solve_group(&mut s, groups[g], o, d, t, alt, &mut out);
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

#[pymethods]
impl Router {
    /// `offsets`/`links`: the mode's CSR adjacency (`Network.out_links`).
    /// `columns`: `(n_columns, n_links)` link travel times; `bin_col[b]` is the
    /// column of time bin `b`. `lower_bound`: per-link lower bound on any
    /// column (free-speed times); with `landmarks > 0` ALT landmarks are
    /// computed on it.
    #[new]
    #[pyo3(signature = (offsets, links, from_node, to_node, length, columns, bin_col, bin_size,
                        lower_bound=None, landmarks=0))]
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
    ) -> PyResult<Router> {
        let offsets = offsets.as_slice()?;
        let n = offsets.len().checked_sub(1).ok_or_else(|| PyValueError::new_err("empty offsets"))?;
        let n_links = from_node.len()?;
        let from_node = to_u32("from_node", from_node.as_slice()?, n)?;
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
        let to: Vec<u32> = link.iter().map(|&l| to_node[l as usize]).collect();
        let fwd = Csr { offsets, link, to };
        let lower = match lower_bound {
            Some(a) => Some(a.as_slice()?.to_vec()),
            None => None,
        };
        let landmarks = if landmarks > 0 {
            let lower = lower.ok_or_else(|| PyValueError::new_err("landmarks need lower_bound"))?;
            if lower.len() != n_links {
                return Err(PyValueError::new_err("lower_bound must have n_links entries"));
            }
            // reverse adjacency of the mode's links
            let mut counts = vec![0usize; n + 1];
            for &l in &fwd.link {
                counts[to_node[l as usize] as usize + 1] += 1;
            }
            for i in 0..n {
                counts[i + 1] += counts[i];
            }
            let mut fill = counts.clone();
            let mut rlink = vec![0u32; fwd.link.len()];
            let mut rto = vec![0u32; fwd.link.len()];
            for &l in &fwd.link {
                let v = to_node[l as usize] as usize;
                rlink[fill[v]] = l;
                rto[fill[v]] = from_node[l as usize];
                fill[v] += 1;
            }
            let bwd = Csr { offsets: counts, link: rlink, to: rto };
            let fwd_ref = &fwd;
            Some(Arc::new(py.allow_threads(|| Landmarks::build(fwd_ref, &bwd, n, &lower, landmarks))))
        } else {
            None
        };
        Ok(Router {
            n,
            fwd,
            from_node,
            length,
            n_links,
            columns,
            is_static: n_cols == 1,
            bin_col,
            bin_size,
            landmarks,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// The same graph and landmarks with other travel times (cheap: the
    /// landmarks are shared, not recomputed). The lower bound the landmarks
    /// were built on must still hold for the new times.
    fn with_times(&self, columns: PyReadonlyArray2<f64>, bin_col: PyReadonlyArray1<i64>,
                  bin_size: f64) -> PyResult<Router> {
        let (columns, n_cols, bin_col) = parse_columns(self.n_links, columns, bin_col, bin_size)?;
        Ok(Router {
            n: self.n,
            fwd: self.fwd.clone(),
            from_node: self.from_node.clone(),
            length: self.length.clone(),
            n_links: self.n_links,
            columns,
            is_static: n_cols == 1,
            bin_col,
            bin_size,
            landmarks: self.landmarks.clone(),
            pool: Mutex::new(Vec::new()),
        })
    }

    #[getter]
    fn landmark_nodes(&self) -> Vec<u32> {
        self.landmarks.as_ref().map_or_else(Vec::new, |l| l.nodes.clone())
    }

    #[getter]
    fn is_static(&self) -> bool {
        self.is_static
    }

    /// Route every query. Returns `(offsets, links, travel_time, distance,
    /// found)`: query `i`'s links are `links[offsets[i]:offsets[i + 1]]`
    /// (int32, so `links.tobytes()` slices straight into packed routes).
    #[pyo3(signature = (origins, destinations, departure_times, algorithm="dijkstra", threads=1))]
    #[allow(clippy::type_complexity)]
    fn route_many<'py>(
        &self,
        py: Python<'py>,
        origins: PyReadonlyArray1<i64>,
        destinations: PyReadonlyArray1<i64>,
        departure_times: PyReadonlyArray1<f64>,
        algorithm: &str,
        threads: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        let alt = match algorithm {
            "dijkstra" => false,
            "alt" if self.landmarks.is_some() => true,
            "alt" => return Err(PyValueError::new_err("algorithm 'alt' needs landmarks > 0")),
            other => return Err(PyValueError::new_err(format!("unknown algorithm {other:?}"))),
        };
        let o = to_u32("origins", origins.as_slice()?, self.n)?;
        let d = to_u32("destinations", destinations.as_slice()?, self.n)?;
        let t = departure_times.as_slice()?.to_vec();
        if d.len() != o.len() || t.len() != o.len() {
            return Err(PyValueError::new_err("query arrays must have the same length"));
        }
        let answers = py.allow_threads(|| self.solve(&o, &d, &t, alt, threads));
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
    #[pyo3(signature = (origin, destination, departure_time, algorithm="dijkstra"))]
    fn route<'py>(&self, py: Python<'py>, origin: usize, destination: usize,
                  departure_time: f64, algorithm: &str)
                  -> PyResult<Option<(Bound<'py, PyBytes>, f64, f64)>> {
        if origin >= self.n || destination >= self.n {
            return Err(PyValueError::new_err("node index out of range"));
        }
        let alt = match algorithm {
            "dijkstra" => false,
            "alt" if self.landmarks.is_some() => true,
            "alt" => return Err(PyValueError::new_err("algorithm 'alt' needs landmarks > 0")),
            other => return Err(PyValueError::new_err(format!("unknown algorithm {other:?}"))),
        };
        let mut s = self.take();
        let mut out = Vec::with_capacity(1);
        self.solve_group(&mut s, &[0], &[origin as u32], &[destination as u32],
                         &[departure_time], alt, &mut out);
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
    Ok(())
}
