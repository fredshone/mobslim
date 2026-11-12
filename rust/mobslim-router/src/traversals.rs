//! Link traversal sums per (link, entry-time bin): the compiled
//! `TraversalAggregate.accumulate` (`mobslim.routing.traveltime`).
//!
//! One pass over the traversals, adding into the caller's arrays in place.
//! The numpy fallback makes about fifteen passes over arrays as long as the
//! traversals, plus three bin-sized `bincount` outputs per chunk; this reads
//! each input once and touches only the cells used, which matters most when
//! several simulations share the memory bandwidth (weaving's strands).

use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The bin of time `a` for bins of `b > 0` seconds: `floor(a / b)` of the
/// exact quotient, as `np.floor_divide` gives it (so every traversal lands
/// in the same bin as with the numpy fallback). The rounded quotient's floor
/// is corrected by the exact remainder: `a - q * b` when `b` is a whole
/// number (then `q * b` is exact, and so is the difference near a bin
/// edge), else with `mul_add`, which rounds once (but may be a slow
/// software call on CPUs without FMA).
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

/// Adds traversal `i` (link `links[i]`, entered at `enter[i]`, left at
/// `leave[i]`) to cell `(link, bin of enter)` of `total` (seconds), `count`
/// and `sum_sq`; traversals with `leave - enter` NaN are skipped. The bin
/// of a time is its slot's: slots of `slot` seconds, slot `s` in bin
/// `slot_bin[s]` (`TimeBins`; uniform bins: one slot each). Times before
/// the first slot or after the last use it. Sums are taken in input order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_traversals(
    py: Python<'_>,
    links: PyReadonlyArray1<i32>,
    enter: PyReadonlyArray1<f64>,
    leave: PyReadonlyArray1<f64>,
    slot: f64,
    slot_bin: PyReadonlyArray1<i64>,
    mut total: PyReadwriteArray2<f64>,
    mut count: PyReadwriteArray2<i64>,
    mut sum_sq: PyReadwriteArray2<f64>,
) -> PyResult<()> {
    let (links, enter, leave) = (links.as_slice()?, enter.as_slice()?, leave.as_slice()?);
    if enter.len() != links.len() || leave.len() != links.len() {
        return Err(PyValueError::new_err("links, enter and leave must have the same length"));
    }
    let shape = total.shape().to_vec();
    if count.shape() != shape.as_slice() || sum_sq.shape() != shape.as_slice() {
        return Err(PyValueError::new_err("total, count and sum_sq must have the same shape"));
    }
    let (n_links, n_bins) = (shape[0], shape[1]);
    let slot_bin = slot_bin.as_slice()?;
    if n_bins == 0 || slot_bin.is_empty() || !(slot > 0.0) {
        return Err(PyValueError::new_err("the table needs at least one bin, of positive size"));
    }
    if slot_bin.iter().any(|&b| b < 0 || b as usize >= n_bins) {
        return Err(PyValueError::new_err("slot_bin must be bins of the table"));
    }
    if links.iter().any(|&l| l < 0 || l as usize >= n_links) {
        return Err(PyValueError::new_err("link index out of range"));
    }
    let (total, count, sum_sq) = (total.as_slice_mut()?, count.as_slice_mut()?, sum_sq.as_slice_mut()?);
    let last = (slot_bin.len() - 1) as f64;
    let whole = slot.fract() == 0.0 && slot < (1u64 << 32) as f64;
    py.allow_threads(|| {
        for i in 0..links.len() {
            let dt = leave[i] - enter[i];
            if dt.is_nan() {
                continue;
            }
            let s = floor_divide(enter[i], slot, whole).clamp(0.0, last) as usize;
            let b = slot_bin[s] as usize;
            let k = links[i] as usize * n_bins + b;
            total[k] += dt;
            count[k] += 1;
            sum_sq[k] += dt * dt;
        }
    });
    Ok(())
}
