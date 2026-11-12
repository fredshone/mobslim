"""Compiled routers (the optional ``mobslim_router`` Rust extension).

:class:`RustRouter` has the :class:`~mobslim.routing.base.Router` interface
plus a parallel :meth:`~RustRouter.route_many`. Algorithms:

* ``"auto"`` (default): ``"alt"`` with :data:`AUTO_LANDMARKS` landmarks
  when no link time is below free speed (always so for learned travel
  times), else ``"dijkstra"``. ALT with 4 landmarks routes 2-5x faster than
  Dijkstra on the showcase networks.
* ``"dijkstra"``: time-dependent Dijkstra, *identical* routes to
  :class:`~mobslim.routing.dijkstra.DijkstraRouter`.
* ``"alt"``: A* with landmarks. Landmark lower bounds are computed once per
  network and mode on free-speed times and reused for every later table,
  which is valid because learned link times are never below free speed
  (:func:`~mobslim.routing.traveltime.travel_time_table` clamps them).
  Same least travel times as Dijkstra; where several routes tie, ALT may
  pick a different one.

:func:`default_router` is the router factory the controller uses: the Rust
router when it is installed and the travel times are a
:class:`TravelTimeTable` or :class:`FreeSpeedTravelTime`, else the Python one.

Use ``functools.partial(RustRouter, algorithm="dijkstra", threads=4)`` as a
factory to change the options (a ``partial`` pickles, so it works with
:class:`~mobslim.replanning.runners.ProcessPoolRunner`).
"""

import os
from contextlib import contextmanager
from typing import Optional, Sequence

import numpy as np

from mobslim.network import Network
from mobslim.population import Route
from mobslim.routing.base import NoRouteError, Router, TravelTime
from mobslim.routing.dijkstra import DijkstraRouter
from mobslim.routing.traveltime import (
    FreeSpeedTravelTime,
    TimeBins,
    TravelTimeTable,
)

try:  # rust/mobslim-router; optional
    import mobslim_router as _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None

ALGORITHMS = ("dijkstra", "alt", "auto")
DEFAULT_LANDMARKS = 16
AUTO_LANDMARKS = 4  # "auto": fastest of 4, 8, 16 on the 40 and 80 grids and Sioux Falls
THREADS_ENV = "MOBSLIM_ROUTER_THREADS"


def rust_router_available() -> bool:
    """Whether the compiled router (``mobslim_router``) is installed."""
    return _rust is not None


def default_threads() -> int:
    """``$MOBSLIM_ROUTER_THREADS`` if set, else the number of CPUs."""
    env = os.environ.get(THREADS_ENV)
    return max(1, int(env)) if env else (os.cpu_count() or 1)


@contextmanager
def router_threads(threads: Optional[int]):
    """Routers made inside default to ``threads`` threads (``None``: no
    change); ``$MOBSLIM_ROUTER_THREADS``, if set, still wins."""
    if threads is None or THREADS_ENV in os.environ:
        yield
        return
    os.environ[THREADS_ENV] = str(threads)
    try:
        yield
    finally:
        os.environ.pop(THREADS_ENV, None)


def travel_time_columns(network: Network, travel_time: TravelTime):
    """``(columns, bin_col, bin_size)``: the distinct time-bin columns of the
    travel times as an ``(n_columns, n_links)`` array, and the column of each
    uniform bin of ``bin_size`` seconds (with uneven bins, a ``TimeBins``:
    each slot's bin's column)."""
    if isinstance(travel_time, TravelTimeTable):
        t = travel_time.times
        index: dict[bytes, int] = {}
        bin_col = np.empty(t.shape[1], dtype=np.int64)
        cols = []
        for b in range(t.shape[1]):
            col = np.ascontiguousarray(t[:, b], dtype=np.float64)
            c = index.setdefault(col.tobytes(), len(cols))
            if c == len(cols):
                cols.append(col)
            bin_col[b] = c
        bins = travel_time.bin_size
        if isinstance(bins, TimeBins):
            return np.stack(cols), bin_col[bins.slot_bin], bins.slot
        return np.stack(cols), bin_col, float(bins)
    if isinstance(travel_time, FreeSpeedTravelTime):
        fs = np.ascontiguousarray(network.free_speed_travel_times(), dtype=np.float64)
        return fs[None, :], np.zeros(1, dtype=np.int64), 1.0
    raise TypeError(
        "RustRouter needs a TravelTimeTable or FreeSpeedTravelTime, "
        f"not {type(travel_time).__name__}."
    )


def _base_router(network: Network, mode: str, landmarks: int):
    """The compiled graph for ``(network, mode)`` with free-speed times and
    ``landmarks`` ALT landmarks, built once and cached on the network."""
    key = ("rust_router", mode, landmarks)
    cache = network._cache
    if key not in cache:
        offsets, links = network.out_links(mode)
        fs = np.ascontiguousarray(network.free_speed_travel_times(), dtype=np.float64)
        cache[key] = _rust.Router(
            np.asarray(offsets, dtype=np.int64),
            np.asarray(links, dtype=np.int64),
            np.asarray(network.from_node, dtype=np.int64),
            np.asarray(network.to_node, dtype=np.int64),
            np.ascontiguousarray(network.length, dtype=np.float64),
            fs[None, :],
            np.zeros(1, dtype=np.int64),
            1.0,
            lower_bound=fs,
            landmarks=landmarks,
        )
    return cache[key]


class RustRouter:
    def __init__(
        self,
        network: Network,
        mode: str,
        travel_time: TravelTime,
        algorithm: str = "auto",
        landmarks: int = DEFAULT_LANDMARKS,
        threads: Optional[int] = None,
    ):
        if _rust is None:
            raise ImportError(
                "RustRouter needs the mobslim_router extension "
                "(rust/mobslim-router; `uv sync` builds it when cargo is installed)."
            )
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, not {algorithm!r}.")
        self.mode = mode
        self.threads = threads or default_threads()
        columns, bin_col, bin_size = travel_time_columns(network, travel_time)
        if algorithm == "auto":  # ALT when its lower bound holds, else Dijkstra
            valid = (columns >= network.free_speed_travel_times()[None, :]).all()
            algorithm = "alt" if valid else "dijkstra"
            landmarks = AUTO_LANDMARKS if landmarks == DEFAULT_LANDMARKS else landmarks
        self.algorithm = algorithm
        if algorithm == "alt":
            if not (columns >= network.free_speed_travel_times()[None, :]).all():
                raise ValueError(
                    "algorithm='alt' needs every link time >= its free-speed time "
                    "(the landmark lower bound)."
                )
        else:
            landmarks = 0
        self._router = _base_router(network, mode, landmarks).with_times(columns, bin_col, bin_size)
        self.is_static = self._router.is_static

    @property
    def landmark_nodes(self) -> list[int]:
        return list(self._router.landmark_nodes)

    def _no_route(self, origin, destination):
        return NoRouteError(
            f"No '{self.mode}' route from node {origin} to node {destination}."
        )

    def route(self, origin: int, destination: int, departure_time: float) -> Route:
        found = self._router.route(origin, destination, departure_time, self.algorithm)
        if found is None:
            raise self._no_route(origin, destination)
        data, travel_time, distance = found
        return Route.packed(data, travel_time, distance)

    def route_many(
        self,
        origins: Sequence[int],
        destinations: Sequence[int],
        departure_times: Sequence[float],
    ) -> list[Route]:
        """Route every query, in parallel over ``threads`` threads."""
        o = np.asarray(origins, dtype=np.int64)
        d = np.asarray(destinations, dtype=np.int64)
        t = np.asarray(departure_times, dtype=np.float64)
        offsets, links, tt, dist, found = self._router.route_many(
            o, d, t, self.algorithm, self.threads
        )
        if not found.all():
            q = int(np.flatnonzero(~found)[0])
            raise self._no_route(int(o[q]), int(d[q]))
        # one buffer for all routes; each route keeps its slice as packed int32
        data, packed = links.tobytes(), Route.packed
        bounds = (offsets * 4).tolist()
        return [
            packed(data[a:b], x, y)
            for a, b, x, y in zip(bounds, bounds[1:], tt.tolist(), dist.tolist())
        ]


def default_router(network: Network, mode: str, travel_time: TravelTime) -> Router:
    """Router factory: :class:`RustRouter` when installed and the travel
    times are a table, else :class:`DijkstraRouter`. Both give shortest
    routes; the compiled one uses ALT (A* with landmarks, about twice as fast)
    whenever no link is faster than free speed, which holds for learned
    travel times."""
    if _rust is not None and isinstance(travel_time, (TravelTimeTable, FreeSpeedTravelTime)):
        return RustRouter(network, mode, travel_time)
    return DijkstraRouter(network, mode, travel_time)
