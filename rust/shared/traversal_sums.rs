//! Link traversal sums per (link, entry-time bin), built while simulating:
//! the in-simulation form of `TraversalAggregate` (`mobslim.routing.traveltime`).
//!
//! Shared source, included by each mobsim kernel crate with
//! `#[path = "../../shared/traversal_sums.rs"] mod traversal_sums;`.
//!
//! A traversal is added when its vehicle leaves the link, to the bin of the
//! time it entered. That bin is almost always the current one or the one
//! before, so the cells are stored bin by bin (`cells[bin * n + link]`):
//! the cells being written at any time are one or two rows of the table,
//! which stay in cache, where a link-by-link table would be written all
//! over. The table is allocated zeroed (untouched pages cost nothing) and
//! each cell keeps its sums together, so an addition touches one cache line.
//!
//! Bins are any `TimeBins`: uniform slots of `slot` seconds, slot `s` in bin
//! `slot_bin[s]` (uniform bins: one slot each). A time's slot is
//! `np.floor_divide(enter, slot)`, clamped to the slots, so every traversal
//! lands in the same cell as in the Python aggregate; with whole-second
//! times the sums are exact, so they do not depend on the order of
//! additions either.

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[derive(Clone, Copy)]
#[repr(C)]
struct Cell {
    total: f64,
    sum_sq: f64,
    count: u64,
}

/// The cells used, as `SparseAggregate` takes them: flat indices
/// `link * n_bins + bin` into the `(n_links, n_bins)` table (unique, by
/// bin, then link), and their sums.
#[derive(Default)]
pub struct SparseSums {
    pub cells: Vec<i64>,
    pub total: Vec<f64>,
    pub count: Vec<i32>,
    pub sum_sq: Vec<f64>,
}

impl SparseSums {
    /// Appends `other` (whose cells must not occur in `self`).
    #[allow(dead_code)] // the partitioned kernels'
    pub fn extend(&mut self, other: SparseSums) {
        self.cells.extend(other.cells);
        self.total.extend(other.total);
        self.count.extend(other.count);
        self.sum_sq.extend(other.sum_sq);
    }

    /// `(cells, total, count, sum_sq)` as numpy arrays.
    pub fn into_py(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let items: Vec<Bound<'_, PyAny>> = vec![
            self.cells.into_pyarray(py).into_any(),
            self.total.into_pyarray(py).into_any(),
            self.count.into_pyarray(py).into_any(),
            self.sum_sq.into_pyarray(py).into_any(),
        ];
        Ok(PyTuple::new(py, items)?.into_any())
    }
}

pub struct TraversalSums {
    n: usize,            // links in the table
    n_bins: usize,       // 0: off (nothing is added)
    slot: f64,
    slot_bin: Vec<u32>,  // each slot's bin
    starts: Vec<f64>,    // each bin's start time; then +inf
    whole: bool,         // slot is a whole number (exact remainders)
    last: f64,           // last slot
    global: Option<Vec<u32>>, // table row -> network link (None: the same)
    cells: Vec<Cell>,
    lo: usize,           // bins touched: lo..hi
    hi: usize,
    // the last bin computed, and its time range (whole slots only): nearly
    // every traversal is in it or the one before, without a division
    cur: usize,
    cur_lo: f64,
    cur_hi: f64,
    prev_lo: f64, // the bin before's start
}

impl TraversalSums {
    /// No table: `add` does nothing.
    #[allow(dead_code)] // HermesSim's
    pub fn off() -> Self {
        TraversalSums::new(0, Vec::new(), 1.0, None)
    }

    /// A table of `n` links by the bins of slots of `slot` seconds, slot `s`
    /// in bin `slot_bin[s]` (runs of slots numbered from 0; empty: off).
    /// `global[i]`: the network link of row `i` (a partition's links), if
    /// not `i`.
    pub fn new(n: usize, slot_bin: Vec<u32>, slot: f64, global: Option<Vec<u32>>) -> Self {
        let on = slot > 0.0 && n > 0 && !slot_bin.is_empty();
        let n_bins = if on { *slot_bin.last().unwrap() as usize + 1 } else { 0 };
        let mut starts = vec![f64::INFINITY; n_bins + 1];
        for (s, &b) in slot_bin.iter().enumerate().rev() {
            if on {
                starts[b as usize] = s as f64 * slot;
            }
        }
        TraversalSums {
            n,
            n_bins,
            slot,
            last: slot_bin.len().saturating_sub(1) as f64,
            slot_bin,
            starts,
            whole: slot.fract() == 0.0 && slot < (1u64 << 32) as f64,
            global,
            cells: zeroed(n * n_bins),
            lo: usize::MAX,
            hi: 0,
            cur: 0,
            cur_lo: f64::INFINITY, // empty: the first traversal computes its bin
            cur_hi: f64::NEG_INFINITY,
            prev_lo: f64::INFINITY,
        }
    }

    #[inline]
    #[allow(dead_code)] // not DSim's
    pub fn is_on(&self) -> bool {
        self.n_bins > 0
    }

    /// Adds a traversal of row `link`, entered at `enter` and left at
    /// `leave` (seconds); one with `leave - enter` NaN is skipped.
    #[inline]
    pub fn add(&mut self, link: usize, enter: f64, leave: f64) {
        if self.n_bins == 0 {
            return;
        }
        let dt = leave - enter;
        if dt.is_nan() {
            return;
        }
        let b = if enter >= self.cur_lo && enter < self.cur_hi {
            self.cur
        } else if enter < self.cur_lo && enter >= self.prev_lo {
            self.cur - 1
        } else {
            self.bin_of(enter)
        };
        let c = &mut self.cells[b * self.n + link];
        c.total += dt;
        c.sum_sq += dt * dt;
        c.count += 1;
        self.lo = self.lo.min(b);
        self.hi = self.hi.max(b + 1);
    }

    /// The bin of `enter` (its slot's, clamped to the slots); with whole
    /// slots, remembered with its time range and the bin before's start
    /// (exact: the edges are whole numbers). The first bin reaches back to
    /// -inf, the last on to +inf, as the clamping does.
    #[inline(never)]
    fn bin_of(&mut self, enter: f64) -> usize {
        let s = floor_divide(enter, self.slot, self.whole).clamp(0.0, self.last) as usize;
        let b = self.slot_bin[s] as usize;
        if self.whole {
            let start = |c: usize| if c == 0 { f64::NEG_INFINITY } else { self.starts[c] };
            self.cur = b;
            self.cur_lo = start(b);
            self.cur_hi = if b + 1 == self.n_bins { f64::INFINITY } else { self.starts[b + 1] };
            self.prev_lo = if b == 0 { f64::INFINITY } else { start(b - 1) };
        }
        b
    }

    /// The cells used.
    pub fn into_sparse(self) -> SparseSums {
        let mut out = SparseSums::default();
        if self.n_bins == 0 || self.lo >= self.hi {
            return out;
        }
        for b in self.lo..self.hi {
            let row = &self.cells[b * self.n..(b + 1) * self.n];
            for (i, c) in row.iter().enumerate() {
                if c.count > 0 {
                    let link = self.global.as_ref().map_or(i, |g| g[i] as usize);
                    out.cells.push((link * self.n_bins + b) as i64);
                    out.total.push(c.total);
                    out.count.push(c.count.min(i32::MAX as u64) as i32);
                    out.sum_sq.push(c.sum_sq);
                }
            }
        }
        out
    }
}

/// `n` zeroed cells, allocated zeroed (pages are mapped on first write).
fn zeroed(n: usize) -> Vec<Cell> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<Cell>(n).expect("traversal table too large");
    // SAFETY: all-zero bits are a valid Cell (0.0, 0.0, 0); the buffer comes
    // from the global allocator with the layout a Vec<Cell> of capacity n uses.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout) as *mut Cell;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(p, n, n)
    }
}

/// The bin of time `a` for bins of `b > 0` seconds: `floor(a / b)` of the
/// exact quotient, as `np.floor_divide` gives it (the rounded quotient's
/// floor, corrected by the exact remainder; see
/// `mobslim_router::traversals::floor_divide`).
#[inline]
fn floor_divide(a: f64, b: f64, whole: bool) -> f64 {
    let q = (a / b).floor();
    let r = if whole { a - q * b } else { (-q).mul_add(b, a) };
    if r < 0.0 {
        q - 1.0
    } else if r >= b {
        q + 1.0
    } else {
        q
    }
}
