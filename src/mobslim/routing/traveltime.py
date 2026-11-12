"""Link travel times: free-speed, and time-binned tables learned from a run.

:func:`travel_time_table` turns link traversals (link, entry time, leave time)
into mean travel times binned by link *entry* time (MATSim default: 15 min
bins), with ``np.bincount``. The controller feeds it the mobsim's
trajectories (``Trajectories.travel_time_table``);
:class:`TravelTimeCollector` feeds it from events. The result is an immutable
:class:`TravelTimeTable`: a ``(n_links, n_bins)`` array that routers read,
and that is cheap to send to worker processes or a compiled router.
"""

import logging
from array import array
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from mobslim.events import EventHandler, LinkEnter, LinkLeave
from mobslim.network import Network

logger = logging.getLogger(__name__)

try:  # the compiled TraversalAggregate.accumulate (rust/mobslim-router); optional
    from mobslim_router import accumulate_traversals as _accumulate
except ImportError:  # pragma: no cover - depends on the build
    _accumulate = None

DEFAULT_BIN_SIZE = 900.0  # 15 minutes
DEFAULT_HORIZON = 30 * 3600.0
DEFAULT_OFF_PEAK_BIN_SIZE = 3600.0  # outside DEFAULT_PEAK (see TimeBins.off_peak)
DEFAULT_PEAK = (6 * 3600.0, 22 * 3600.0)
TRAVERSAL_CHUNK = 1 << 21  # traversals aggregated at a time (bounds the temporaries)


@dataclass(frozen=True, eq=False)
class TimeBins:
    """Travel-time bins of any widths: the horizon cut into uniform *slots*
    of ``slot`` seconds, slot ``s`` in bin ``slot_bin[s]`` (bins are runs of
    consecutive slots, numbered from 0).

    Wherever a ``bin_size`` is taken (tables, aggregates, mobsims, the
    controllers), a number means uniform bins of that many seconds and a
    ``TimeBins`` these bins. Times before 0 are in the first bin, times past
    the last slot in the last bin, as with uniform bins. A time's bin is
    its slot's (``floor(t / slot)``), so binning stays one division and a
    lookup, and the compiled router and mobsims take the slots as they are.

    :meth:`off_peak` gives the default: 15-minute bins from 06:00 to 22:00,
    hourly bins before and after (78 bins over 30 hours, against 120), as
    travel times change little outside the peaks.
    """

    slot: float
    slot_bin: np.ndarray  # int32 (n_slots,)

    def __post_init__(self):
        b = np.array(self.slot_bin, dtype=np.int32).reshape(-1)
        if not self.slot > 0:
            raise ValueError("TimeBins.slot must be positive.")
        if len(b) == 0 or b[0] != 0 or ((np.diff(b) != 0) & (np.diff(b) != 1)).any():
            raise ValueError("TimeBins.slot_bin must number runs of slots 0, 1, 2, ...")
        b.flags.writeable = False
        object.__setattr__(self, "slot", float(self.slot))
        object.__setattr__(self, "slot_bin", b)

    @classmethod
    def uniform(cls, bin_size: float, horizon: float = DEFAULT_HORIZON) -> "TimeBins":
        return cls(bin_size, np.arange(int(np.ceil(horizon / bin_size))))

    @classmethod
    def off_peak(cls, bin_size: float = DEFAULT_BIN_SIZE,
                 off_peak_bin_size: float = DEFAULT_OFF_PEAK_BIN_SIZE,
                 peak: tuple[float, float] = DEFAULT_PEAK,
                 horizon: float = DEFAULT_HORIZON) -> "TimeBins":
        """Bins of ``bin_size`` seconds from ``peak[0]`` to ``peak[1]``, and
        of ``off_peak_bin_size`` (a multiple of ``bin_size``) before and
        after, aligned to multiples of it (a bin is cut short at the peak's
        edges)."""
        ratio = off_peak_bin_size / bin_size
        if ratio < 1 or ratio != round(ratio):
            raise ValueError("off_peak_bin_size must be a multiple of bin_size.")
        t = np.arange(int(np.ceil(horizon / bin_size))) * bin_size
        in_peak = (t >= peak[0]) & (t < peak[1])
        coarse = np.floor_divide(t, off_peak_bin_size)
        new = np.ones(len(t), dtype=bool)  # a slot that starts a bin
        new[1:] = in_peak[1:] | (in_peak[1:] != in_peak[:-1]) | (coarse[1:] != coarse[:-1])
        return cls(bin_size, np.cumsum(new) - 1)

    @property
    def n_bins(self) -> int:
        return int(self.slot_bin[-1]) + 1

    @property
    def n_slots(self) -> int:
        return len(self.slot_bin)

    @property
    def horizon(self) -> float:
        return self.slot * self.n_slots

    @property
    def starts(self) -> np.ndarray:
        """Each bin's start time (seconds)."""
        return np.flatnonzero(np.diff(self.slot_bin, prepend=-1)) * self.slot

    def bin_of(self, times: np.ndarray) -> np.ndarray:
        """The bin of each time (``int64``; NaN times give bin 0)."""
        s = np.floor_divide(np.asarray(times, dtype=float), self.slot)
        s[np.isnan(s)] = 0
        np.clip(s, 0, self.n_slots - 1, out=s)
        return self.slot_bin[s.astype(np.int64)].astype(np.int64)

    def __eq__(self, other) -> bool:
        return (isinstance(other, TimeBins) and self.slot == other.slot
                and np.array_equal(self.slot_bin, other.slot_bin))

    def __hash__(self) -> int:
        return hash((self.slot, self.slot_bin.tobytes()))

    def __repr__(self) -> str:
        return f"TimeBins({self.n_bins} bins, slots of {self.slot:g} s up to {self.horizon:g} s)"


Bins = Union[float, TimeBins]  # seconds per bin (uniform), or TimeBins

UINT16_MAX = float(np.iinfo(np.uint16).max)  # 65,535 s: 18.2 hours


def compact_times(times: np.ndarray) -> np.ndarray:
    """Link travel times as whole seconds, rounded up, in two bytes each
    (``uint16``) when they all fit: the form of the learned tables.

    Rounding up keeps every time at or above the time given, so a learned
    table is never below free speed (the routers' landmark bounds stay
    valid). A table with a link time over 18.2 hours (or not finite) is
    kept in ``float32`` whole seconds instead, with a warning.
    """
    t = np.ceil(np.asarray(times, dtype=np.float64))
    if t.size and not (np.isfinite(t).all() and t.max() <= UINT16_MAX and t.min() >= 0):
        logger.warning("Link travel times beyond 0..%d s: kept as float32, not uint16.",
                       int(UINT16_MAX))
        return t.astype(np.float32)
    return t.astype(np.uint16)


def n_bins_of(bin_size: Bins, horizon: float = DEFAULT_HORIZON) -> int:
    """Bins of ``bin_size`` (TimeBins: its own; seconds: up to ``horizon``)."""
    if isinstance(bin_size, TimeBins):
        return bin_size.n_bins
    return int(np.ceil(horizon / bin_size))


def slots_of(bin_size: Bins, n_bins: int) -> tuple[float, np.ndarray]:
    """``(slot, slot_bin)`` of ``n_bins`` bins of ``bin_size``: the form
    the compiled kernels take (uniform bins: one slot per bin)."""
    if isinstance(bin_size, TimeBins):
        if bin_size.n_bins != n_bins:
            raise ValueError(f"{bin_size!r} does not have {n_bins} bins.")
        return bin_size.slot, bin_size.slot_bin
    return float(bin_size), np.arange(n_bins, dtype=np.int32)


class FreeSpeedTravelTime:
    """``length / freespeed``, independent of time."""

    is_static = True  # the same times at every time of day

    def __init__(self, network: Network):
        self._tt = tuple(network.free_speed_travel_times().tolist())

    def link_travel_time(self, link: int, time: float) -> float:
        return self._tt[link]

    def link_travel_times(self, time: float) -> tuple[float, ...]:
        return self._tt


@dataclass(frozen=True, eq=False)
class TravelTimeTable:
    """Immutable ``(n_links, n_bins)`` table of link travel times (seconds).

    Times before 0 use the first bin, times after the last bin use the last.

    Routers read one time bin at a time as a tuple of Python floats (fast to
    index from Python). Those tuples are built lazily, per bin, on first use,
    and identical bins (e.g. all the free-flow night bins) share one tuple,
    so memory grows with the number of *distinct* bins actually used, not
    with ``n_links * n_bins``. Pickling sends only ``times`` and
    ``bin_size``; the per-bin views are rebuilt in each process.
    """

    times: np.ndarray
    bin_size: Bins = DEFAULT_BIN_SIZE  # seconds per bin, or TimeBins
    _columns: Optional[tuple] = field(default=None, repr=False, compare=False)
    _static: Optional[bool] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.times.ndim != 2:
            raise ValueError("TravelTimeTable.times must be 2-D (n_links, n_bins).")
        if self.times.shape[1] < 1:
            raise ValueError("TravelTimeTable needs at least one time bin.")
        times = self.times.view()  # read-only view; the caller's array is untouched
        times.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "_columns", None)
        object.__setattr__(self, "_static", None)

    def __getstate__(self):
        return {"times": self.times, "bin_size": self.bin_size}

    def __setstate__(self, state):
        object.__setattr__(self, "times", state["times"])
        object.__setattr__(self, "bin_size", state["bin_size"])
        object.__setattr__(self, "_columns", None)
        object.__setattr__(self, "_static", None)

    @property
    def n_bins(self) -> int:
        return self.times.shape[1]

    @property
    def is_static(self) -> bool:
        """Whether every time bin holds the same times (e.g. free speed).

        Routes on a static table do not depend on the departure time, so
        routers may answer all queries from one origin with one search.
        """
        if self._static is None:
            t = self.times
            object.__setattr__(self, "_static", bool((t == t[:, :1]).all()))
        return self._static

    def time_bin(self, time: float) -> int:
        bins = self.bin_size
        if isinstance(bins, TimeBins):
            s = int(time // bins.slot)
            last = bins.n_slots - 1
            return int(bins.slot_bin[0 if s < 0 else (last if s > last else s)])
        b = int(time // bins)
        last = self.times.shape[1] - 1
        return 0 if b < 0 else (last if b > last else b)

    def link_travel_times(self, time: float) -> tuple[float, ...]:
        """Travel time of every link entered at ``time`` (cached per bin)."""
        cols = self._columns
        if cols is None:
            cols = ([None] * self.times.shape[1], {})  # per bin, by content
            object.__setattr__(self, "_columns", cols)
        b = self.time_bin(time)
        col = cols[0][b]
        if col is None:
            column = np.ascontiguousarray(self.times[:, b])
            key = column.tobytes()
            col = cols[1].get(key)
            if col is None:  # as floats, whatever the table's dtype
                col = cols[1][key] = tuple(column.astype(np.float64).tolist())
            cols[0][b] = col
        return col

    def link_travel_time(self, link: int, time: float) -> float:
        return self.link_travel_times(time)[link]

    @classmethod
    def free_speed(
        cls, network: Network, bin_size: Bins = DEFAULT_BIN_SIZE, horizon: float = DEFAULT_HORIZON
    ) -> "TravelTimeTable":
        n_bins = n_bins_of(bin_size, horizon)
        fs = network.free_speed_travel_times()
        return cls(np.repeat(fs[:, None], n_bins, axis=1), bin_size)

    def blend(self, other: "TravelTimeTable", weight: float) -> "TravelTimeTable":
        """``(1 - weight) * self + weight * other`` (for smoothed learning)."""
        if other.times.shape != self.times.shape or other.bin_size != self.bin_size:
            raise ValueError("Can only blend tables with the same shape and bin size.")
        return TravelTimeTable((1 - weight) * self.times + weight * other.times, self.bin_size)


class TravelTimeCollector(EventHandler):
    """Measures mean link travel time per (link, entry-time bin) from events.

    Observations are appended to compact arrays (12 bytes each) and
    aggregated in :meth:`table`, so memory grows with the number of link
    traversals, not with ``n_links * n_bins``. The controller computes the
    same table straight from the mobsim's trajectories
    (:func:`travel_time_table`); this handler is for event streams.
    """

    handles = (LinkEnter, LinkLeave)

    def __init__(
        self, network: Network, bin_size: Bins = DEFAULT_BIN_SIZE, horizon: float = DEFAULT_HORIZON
    ):
        self.network = network
        self.bin_size = bin_size
        self.horizon = horizon
        self.reset(0)

    def reset(self, iteration: int) -> None:
        self._link = array("i")
        self._enter = array("d")
        self._leave = array("d")
        self._entered: dict[int, tuple[int, float]] = {}

    def handle(self, event) -> None:
        if type(event) is LinkEnter:
            self._entered[event.person] = (event.link, event.time)
        else:
            link, t0 = self._entered.pop(event.person)
            self._link.append(link)
            self._enter.append(t0)
            self._leave.append(event.time)

    def table(self) -> TravelTimeTable:
        """Mean observed times; free-speed where nothing was observed."""
        return travel_time_table(
            self.network,
            np.frombuffer(self._link, dtype=np.int32) if self._link else np.empty(0, np.int32),
            np.frombuffer(self._enter) if self._enter else np.empty(0),
            np.frombuffer(self._leave) if self._leave else np.empty(0),
            self.bin_size,
            self.horizon,
        )


@dataclass(frozen=True, eq=False)
class TraversalAggregate:
    """Sums of link traversal times per (link, entry-time bin).

    ``total`` (seconds), ``sum_sq`` (seconds squared) and ``count`` are
    ``(n_links, n_bins)`` arrays. Aggregates of the same shape add
    (``a + b``), so traversals measured in separate simulations (the strands
    of a :class:`~mobslim.weave.Weave`) or arriving at different times pool
    into one estimate; :meth:`table` finalises them into a
    :class:`TravelTimeTable`.
    """

    total: np.ndarray
    count: np.ndarray
    sum_sq: np.ndarray
    bin_size: Bins = DEFAULT_BIN_SIZE  # seconds per bin, or TimeBins

    @classmethod
    def empty(cls, n_links: int, n_bins: int, bin_size: Bins = DEFAULT_BIN_SIZE) -> "TraversalAggregate":
        z = np.zeros((n_links, n_bins))
        return cls(z, np.zeros((n_links, n_bins), dtype=np.int64), z.copy(), bin_size)

    @classmethod
    def from_traversals(cls, n_links: int, links: np.ndarray, enter: np.ndarray,
                        leave: np.ndarray, bin_size: Bins = DEFAULT_BIN_SIZE,
                        horizon: float = DEFAULT_HORIZON) -> "TraversalAggregate":
        """``links[k]`` was entered at ``enter[k]`` and left at ``leave[k]``.
        Traversals that were never completed (``leave`` is NaN) are ignored."""
        agg = cls.empty(n_links, n_bins_of(bin_size, horizon), bin_size)
        for s in range(0, len(links), TRAVERSAL_CHUNK):
            e = s + TRAVERSAL_CHUNK
            agg.accumulate(links[s:e], enter[s:e], leave[s:e])
        return agg

    def accumulate(self, links: np.ndarray, enter: np.ndarray, leave: np.ndarray) -> None:
        """Add traversals in place (as :meth:`from_traversals`; incomplete
        ones, with ``enter`` or ``leave`` NaN, are ignored). In one compiled
        pass when ``mobslim_router`` is installed; else call it with chunks
        of at most about :data:`TRAVERSAL_CHUNK`: it uses about 25 bytes per
        traversal of temporaries, plus three bin-sized arrays."""
        slot, slot_bin = slots_of(self.bin_size, self.n_bins)
        if _accumulate is not None and self.count.dtype == np.int64 and all(
                a.flags.c_contiguous for a in (self.total, self.count, self.sum_sq)):
            _accumulate(np.ascontiguousarray(links, dtype=np.int32),
                        np.ascontiguousarray(enter, dtype=np.float64),
                        np.ascontiguousarray(leave, dtype=np.float64),
                        float(slot), np.ascontiguousarray(slot_bin, dtype=np.int64),
                        self.total, self.count, self.sum_sq)
            return
        n_bins = self.n_bins
        size = self.total.size
        dt = leave - enter
        skip = np.isnan(dt)
        b = np.floor_divide(enter, slot)
        b[skip] = 0
        np.clip(b, 0, len(slot_bin) - 1, out=b)
        k = links.astype(np.int64)
        k *= n_bins
        k += slot_bin[b.astype(np.int64)]
        k[skip] = size  # an extra bin, dropped below
        dt[skip] = 0.0
        self.total.reshape(-1)[:] += np.bincount(k, weights=dt, minlength=size + 1)[:size]
        self.count.reshape(-1)[:] += np.bincount(k, minlength=size + 1)[:size]
        dt *= dt
        self.sum_sq.reshape(-1)[:] += np.bincount(k, weights=dt, minlength=size + 1)[:size]

    @property
    def n_bins(self) -> int:
        return self.total.shape[1]

    def __add__(self, other: "TraversalAggregate") -> "TraversalAggregate":
        _check_same(self, other)
        return TraversalAggregate(self.total + other.total, self.count + other.count,
                                  self.sum_sq + other.sum_sq, self.bin_size)

    def add_into(self, dense: "TraversalAggregate") -> None:
        """Add this aggregate to ``dense``, in place."""
        _check_same(dense, self)
        np.add(dense.total, self.total, out=dense.total)
        np.add(dense.count, self.count, out=dense.count)
        np.add(dense.sum_sq, self.sum_sq, out=dense.sum_sq)

    def to_sparse(self) -> "SparseAggregate":
        """Only the cells with observations (see :class:`SparseAggregate`)."""
        cells = np.flatnonzero(self.count)
        return SparseAggregate(cells, self.total.reshape(-1)[cells],
                               self.count.reshape(-1)[cells].astype(np.int32),
                               self.sum_sq.reshape(-1)[cells], self.total.shape, self.bin_size)

    def mean(self) -> np.ndarray:
        """Mean travel time per bin (NaN where nothing was observed)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.total / self.count

    def variance(self) -> np.ndarray:
        """Sample variance per bin (NaN with fewer than two observations)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            n = self.count
            return np.where(n > 1, (self.sum_sq - self.total ** 2 / n) / (n - 1), np.nan)

    def table(self, network: Network, compact: bool = True) -> "TravelTimeTable":
        """Mean observed times; free speed where nothing was observed or the
        mean is below free speed. ``compact``: in whole seconds, rounded up,
        two bytes each (see :func:`compact_times`); else as computed."""
        fs = network.free_speed_travel_times()[:, None]
        times = np.where(self.count > 0, np.maximum(self.mean(), fs), fs)
        return TravelTimeTable(compact_times(times) if compact else times, self.bin_size)


@dataclass(frozen=True, eq=False)
class SparseAggregate:
    """A :class:`TraversalAggregate` holding only its cells with
    observations: ``cells`` (unique flat indices into the
    ``(n_links, n_bins)`` arrays) and their ``total``, ``count`` and
    ``sum_sq``. Few cells are used (departures span a few hours of the
    horizon), so this is the form to send between processes.
    """

    cells: np.ndarray  # int64[k]
    total: np.ndarray  # float64[k]
    count: np.ndarray  # int32[k]
    sum_sq: np.ndarray  # float64[k]
    shape: tuple
    bin_size: Bins = DEFAULT_BIN_SIZE  # seconds per bin, or TimeBins

    @property
    def n_bins(self) -> int:
        return self.shape[1]

    def add_into(self, dense: TraversalAggregate) -> None:
        """Add this aggregate to ``dense``, in place (the cells are unique,
        so fancy-index addition is exact)."""
        _check_same(dense, self)
        dense.total.reshape(-1)[self.cells] += self.total
        dense.count.reshape(-1)[self.cells] += self.count
        dense.sum_sq.reshape(-1)[self.cells] += self.sum_sq

    def to_dense(self) -> TraversalAggregate:
        dense = TraversalAggregate.empty(*self.shape, self.bin_size)
        self.add_into(dense)
        return dense


def _check_same(dense: TraversalAggregate, other) -> None:
    shape = other.shape if isinstance(other, SparseAggregate) else other.total.shape
    if shape != dense.total.shape or other.bin_size != dense.bin_size:
        raise ValueError("Can only add aggregates with the same shape and bin size.")


def sum_aggregates(aggregates) -> TraversalAggregate:
    """The sum of dense and sparse aggregates, added in order into the
    first dense one, in place (no copies); all sparse: a new dense one."""
    aggregates = list(aggregates)
    first = next((i for i, a in enumerate(aggregates) if isinstance(a, TraversalAggregate)), None)
    total = aggregates[0].to_dense() if first is None else aggregates[first]
    for i, a in enumerate(aggregates):
        if i != first and not (first is None and i == 0):
            a.add_into(total)
    return total


class TravelTimeEstimator:
    """Travel times learned across iterations from traversal aggregates.

    Each :meth:`observe` finalises the aggregate into a table and smooths it
    into the estimate: ``tt <- (1 - smoothing) * tt + smoothing * observed``.
    ``smoothing = 1`` (the default) keeps only the latest observation, which
    is :class:`~mobslim.controller.Controller`'s behaviour. The estimate
    starts at free speed.
    """

    def __init__(self, network: Network, bin_size: Bins = DEFAULT_BIN_SIZE,
                 horizon: float = DEFAULT_HORIZON, smoothing: float = 1.0):
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1].")
        self.network = network
        self.smoothing = smoothing
        self._table = TravelTimeTable.free_speed(network, bin_size, horizon)
        self._estimate = self._table  # smoothed, not rounded (smoothing < 1)

    def observe(self, aggregate: TraversalAggregate) -> TravelTimeTable:
        """The new estimate, as a compact table (whole seconds, rounded up;
        see :func:`compact_times`). Smoothing works on the unrounded times,
        so rounding never accumulates."""
        if self.smoothing == 1:
            self._table = aggregate.table(self.network)
        else:
            observed = aggregate.table(self.network, compact=False)
            self._estimate = self._estimate.blend(observed, self.smoothing)
            self._table = TravelTimeTable(compact_times(self._estimate.times),
                                          self._estimate.bin_size)
        return self._table

    def table(self) -> TravelTimeTable:
        return self._table


def travel_time_table(
    network: Network,
    links: np.ndarray,
    enter: np.ndarray,
    leave: np.ndarray,
    bin_size: Bins = DEFAULT_BIN_SIZE,
    horizon: float = DEFAULT_HORIZON,
) -> TravelTimeTable:
    """Mean link travel time per (link, entry-time bin) from traversal arrays.

    ``links[k]`` was entered at ``enter[k]`` and left at ``leave[k]``.
    Traversals that were never completed (``leave`` is NaN) are ignored.
    Bins without observations, and means below free speed, use free speed.
    (:meth:`TraversalAggregate.from_traversals`, then
    :meth:`TraversalAggregate.table`.)
    """
    return TraversalAggregate.from_traversals(network.n_links, links, enter, leave,
                                              bin_size, horizon).table(network)
