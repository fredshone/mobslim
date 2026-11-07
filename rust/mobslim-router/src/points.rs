//! Nearest-point queries in the plane, for snapping activities to the link
//! with the nearest midpoint (`mobslim.activity_links.LinkSnapper`).
//!
//! A static, implicit k-d tree: the points are permuted so that every range
//! `[lo, hi)` has its median at `mid = (lo + hi) / 2`, with points not after
//! it on the split axis (x at even depth, y at odd) in `[lo, mid)` and points
//! not before it in `(mid, hi)`. Ties in distance go to the lowest point
//! index, so results do not depend on the tree's layout or the thread count.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Ranges this small are scanned rather than split.
const LEAF: usize = 8;

struct Tree {
    xy: Vec<[f64; 2]>,
    id: Vec<u32>,
}

impl Tree {
    fn new(points: &[[f64; 2]]) -> Tree {
        let mut order: Vec<(u32, [f64; 2])> =
            points.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
        Self::build(&mut order, 0);
        Tree {
            xy: order.iter().map(|&(_, p)| p).collect(),
            id: order.iter().map(|&(i, _)| i).collect(),
        }
    }

    fn build(items: &mut [(u32, [f64; 2])], depth: usize) {
        if items.len() <= LEAF {
            return;
        }
        let axis = depth & 1;
        let mid = items.len() / 2;
        items.select_nth_unstable_by(mid, |a, b| a.1[axis].total_cmp(&b.1[axis]));
        let (left, rest) = items.split_at_mut(mid);
        Self::build(left, depth + 1);
        Self::build(&mut rest[1..], depth + 1);
    }

    /// `(index, squared distance)` of the point nearest `q`.
    fn nearest(&self, q: [f64; 2]) -> (u32, f64) {
        let mut best = (u32::MAX, f64::INFINITY);
        self.search(0, self.xy.len(), 0, q, &mut best);
        best
    }

    #[inline]
    fn offer(&self, j: usize, q: [f64; 2], best: &mut (u32, f64)) {
        let p = self.xy[j];
        let d = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
        if d < best.1 || (d == best.1 && self.id[j] < best.0) {
            *best = (self.id[j], d);
        }
    }

    fn search(&self, lo: usize, hi: usize, depth: usize, q: [f64; 2], best: &mut (u32, f64)) {
        if hi - lo <= LEAF {
            for j in lo..hi {
                self.offer(j, q, best);
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let axis = depth & 1;
        let diff = q[axis] - self.xy[mid][axis];
        let (near, far) = if diff < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
        self.search(near.0, near.1, depth + 1, q, best);
        self.offer(mid, q, best);
        // `<=`: a point at exactly the best distance may still win on index
        if diff * diff <= best.1 {
            self.search(far.0, far.1, depth + 1, q, best);
        }
    }
}

fn points(name: &str, a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let v = a.as_array();
    if v.ncols() != 2 {
        return Err(PyValueError::new_err(format!("{name} must have shape (n, 2)")));
    }
    let out: Vec<[f64; 2]> = v.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    if out.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must be finite")));
    }
    Ok(out)
}

/// Exact nearest-point index over a fixed set of 2-D points.
#[pyclass(module = "mobslim_router")]
pub struct PointIndex {
    tree: Tree,
}

#[pymethods]
impl PointIndex {
    #[new]
    fn py_new(xy: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let pts = points("xy", &xy)?;
        if pts.is_empty() {
            return Err(PyValueError::new_err("PointIndex needs at least one point"));
        }
        if pts.len() > u32::MAX as usize {
            return Err(PyValueError::new_err("too many points"));
        }
        Ok(PointIndex { tree: Tree::new(&pts) })
    }

    fn __len__(&self) -> usize {
        self.tree.xy.len()
    }

    /// Index (into the points given at construction) of the point nearest
    /// each row of `xy`; ties go to the lowest index.
    #[pyo3(signature = (xy, threads=1))]
    fn nearest<'py>(&self, py: Python<'py>, xy: PyReadonlyArray2<'py, f64>, threads: usize)
                    -> PyResult<Bound<'py, PyArray1<i64>>> {
        let q = points("query xy", &xy)?;
        let out = py.allow_threads(|| {
            let mut out = vec![0i64; q.len()];
            let threads = threads.max(1).min(q.len() / 4096 + 1);
            if threads == 1 {
                for (o, &p) in out.iter_mut().zip(&q) {
                    *o = self.tree.nearest(p).0 as i64;
                }
            } else {
                let chunk = q.len().div_ceil(threads);
                std::thread::scope(|scope| {
                    for (os, qs) in out.chunks_mut(chunk).zip(q.chunks(chunk)) {
                        scope.spawn(move || {
                            for (o, &p) in os.iter_mut().zip(qs) {
                                *o = self.tree.nearest(p).0 as i64;
                            }
                        });
                    }
                });
            }
            out
        });
        Ok(out.into_pyarray(py))
    }
}

#[cfg(test)]
mod tests {
    use super::Tree;

    #[test]
    fn matches_brute_force_with_lowest_index_ties() {
        let mut s = 7u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f64 / 1000.0
        };
        // coarse coordinates: many exact duplicates and ties
        let pts: Vec<[f64; 2]> = (0..3000).map(|_| [(rnd() * 50.0).round(), (rnd() * 50.0).round()]).collect();
        let tree = Tree::new(&pts);
        for _ in 0..5000 {
            let q = [(rnd() * 60.0).round() / 2.0, (rnd() * 60.0).round() / 2.0];
            let mut want = (u32::MAX, f64::INFINITY);
            for (i, p) in pts.iter().enumerate() {
                let d = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                if d < want.1 {
                    want = (i as u32, d);
                }
            }
            assert_eq!(tree.nearest(q), want);
        }
    }
}
