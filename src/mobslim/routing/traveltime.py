"""Link travel times: free-speed, and time-binned tables learned from a run.

:func:`travel_time_table` turns link traversals (link, entry time, leave time)
into mean travel times binned by link *entry* time (MATSim default: 15 min
bins), with ``np.bincount``. The controller feeds it the mobsim's
trajectories (``Trajectories.travel_time_table``);
:class:`TravelTimeCollector` feeds it from events. The result is an immutable
:class:`TravelTimeTable`: a ``(n_links, n_bins)`` array that routers read,
and that is cheap to send to worker processes or a compiled router.
"""

from array import array
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mobslim.events import EventHandler, LinkEnter, LinkLeave
from mobslim.network import Network

DEFAULT_BIN_SIZE = 900.0  # 15 minutes
DEFAULT_HORIZON = 30 * 3600.0


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
    bin_size: float = DEFAULT_BIN_SIZE
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
        b = int(time // self.bin_size)
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
            if col is None:
                col = cols[1][key] = tuple(column.tolist())
            cols[0][b] = col
        return col

    def link_travel_time(self, link: int, time: float) -> float:
        return self.link_travel_times(time)[link]

    @classmethod
    def free_speed(
        cls, network: Network, bin_size: float = DEFAULT_BIN_SIZE, horizon: float = DEFAULT_HORIZON
    ) -> "TravelTimeTable":
        n_bins = int(np.ceil(horizon / bin_size))
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
        self, network: Network, bin_size: float = DEFAULT_BIN_SIZE, horizon: float = DEFAULT_HORIZON
    ):
        self.network = network
        self.bin_size = bin_size
        self.n_bins = int(np.ceil(horizon / bin_size))
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
            self.n_bins * self.bin_size,
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
    bin_size: float = DEFAULT_BIN_SIZE

    @classmethod
    def empty(cls, n_links: int, n_bins: int, bin_size: float = DEFAULT_BIN_SIZE) -> "TraversalAggregate":
        z = np.zeros((n_links, n_bins))
        return cls(z, np.zeros((n_links, n_bins), dtype=np.int64), z.copy(), bin_size)

    @classmethod
    def from_traversals(cls, n_links: int, links: np.ndarray, enter: np.ndarray,
                        leave: np.ndarray, bin_size: float = DEFAULT_BIN_SIZE,
                        horizon: float = DEFAULT_HORIZON) -> "TraversalAggregate":
        """``links[k]`` was entered at ``enter[k]`` and left at ``leave[k]``.
        Traversals that were never completed (``leave`` is NaN) are ignored."""
        n_bins = int(np.ceil(horizon / bin_size))
        done = ~np.isnan(leave)
        links, enter, leave = links[done], enter[done], leave[done]
        b = np.clip((enter // bin_size).astype(np.int64), 0, n_bins - 1)
        k = links.astype(np.int64) * n_bins + b
        size = n_links * n_bins
        dt = leave - enter
        total = np.bincount(k, weights=dt, minlength=size).reshape(-1, n_bins)
        sum_sq = np.bincount(k, weights=dt * dt, minlength=size).reshape(-1, n_bins)
        count = np.bincount(k, minlength=size).reshape(-1, n_bins)
        return cls(total, count, sum_sq, bin_size)

    @property
    def n_bins(self) -> int:
        return self.total.shape[1]

    def __add__(self, other: "TraversalAggregate") -> "TraversalAggregate":
        if other.total.shape != self.total.shape or other.bin_size != self.bin_size:
            raise ValueError("Can only add aggregates with the same shape and bin size.")
        return TraversalAggregate(self.total + other.total, self.count + other.count,
                                  self.sum_sq + other.sum_sq, self.bin_size)

    def mean(self) -> np.ndarray:
        """Mean travel time per bin (NaN where nothing was observed)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.total / self.count

    def variance(self) -> np.ndarray:
        """Sample variance per bin (NaN with fewer than two observations)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            n = self.count
            return np.where(n > 1, (self.sum_sq - self.total ** 2 / n) / (n - 1), np.nan)

    def table(self, network: Network) -> "TravelTimeTable":
        """Mean observed times; free speed where nothing was observed or the
        mean is below free speed."""
        fs = network.free_speed_travel_times()[:, None]
        times = np.where(self.count > 0, np.maximum(self.mean(), fs), fs)
        return TravelTimeTable(times, self.bin_size)


class TravelTimeEstimator:
    """Travel times learned across iterations from traversal aggregates.

    Each :meth:`observe` finalises the aggregate into a table and smooths it
    into the estimate: ``tt <- (1 - smoothing) * tt + smoothing * observed``.
    ``smoothing = 1`` (the default) keeps only the latest observation, which
    is :class:`~mobslim.controller.Controller`'s behaviour. The estimate
    starts at free speed.
    """

    def __init__(self, network: Network, bin_size: float = DEFAULT_BIN_SIZE,
                 horizon: float = DEFAULT_HORIZON, smoothing: float = 1.0):
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1].")
        self.network = network
        self.smoothing = smoothing
        self._table = TravelTimeTable.free_speed(network, bin_size, horizon)

    def observe(self, aggregate: TraversalAggregate) -> TravelTimeTable:
        observed = aggregate.table(self.network)
        self._table = observed if self.smoothing == 1 else self._table.blend(observed, self.smoothing)
        return self._table

    def table(self) -> TravelTimeTable:
        return self._table


def travel_time_table(
    network: Network,
    links: np.ndarray,
    enter: np.ndarray,
    leave: np.ndarray,
    bin_size: float = DEFAULT_BIN_SIZE,
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
