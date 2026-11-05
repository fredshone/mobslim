"""RAPTOR transit routing, least generalised cost (as MATSim's SwissRailRaptor).

RAPTOR (*Round-bAsed Public Transit Optimized Router*, Delling, Pajor and
Werneck, 2012) works in rounds: round *k* finds journeys with *k* rides by
scanning, once each, every transit route through a stop improved in round
*k - 1*, then relaxes footpaths from the stops reached. Like SwissRailRaptor
this implementation keeps **one scalar label, generalised cost**, per stop and
per route stop, improved round by round. Arrival time only decides which
trips can be caught. Costs are linear in the scoring utilities:

* waiting: ``-waiting`` per hour (also the time between a transfer walk and
  the earliest boarding, e.g. a minimal transfer time at the same stop),
* in vehicle: ``-travel`` per hour,
* walking (access, egress, transfer): ``-walk`` per hour,
* every transfer: ``-line_switch``.

A label is replaced only by a strictly cheaper one, so ties keep the
journey found first, which has no more rides. The answer is the cheapest
over the egress stops, compared with walking the whole way (which wins
ties). Boardings and transfers that already cost more than the best answer
are pruned, and the search stops ``max_transfers_after_first_arrival``
rounds after it first reaches the destination.

The router works on the **schedule**, not on experienced delays. A ride names
route stops, not a departure: the mobsim decides which vehicle it is.

:class:`RaptorData` holds the derived tables (stop times per route, stop to
route-stop index, footpaths); it is built once per router and is plain
arrays and lists, rebuilt in each worker process.
"""

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from mobslim.network import Network
from mobslim.population import TransitPassengerRoute
from mobslim.transit.core import Transit

_I32 = np.dtype("=i4")

try:  # rust/mobslim-router; optional
    import mobslim_router as _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None


@dataclass(frozen=True)
class RaptorWeights:
    """The utilities a transit search minimises the negative of [util/h;
    ``line_switch`` util per transfer]. Hashable, so queries with equal
    weights share a search."""

    travel: float = -6.0  # in-vehicle
    waiting: float = -6.0
    walk: float = -12.0  # access, egress, transfers
    line_switch: float = -1.0

    @classmethod
    def from_scoring(cls, params, mode: str = "pt", walk_mode: Optional[str] = None) -> "RaptorWeights":
        """As MATSim's SwissRailRaptor derives them (``RaptorUtils.createParameters``,
        ``TransitRouterConfig``): time also costs the opportunity of not
        performing, so travel, waiting and walking are each the scoring
        utility minus ``performing``; a transfer costs ``utility_of_line_switch``.
        ``walk_mode`` defaults to ``params.walk_mode``."""
        perf = params.performing
        walk_mode = params.walk_mode if walk_mode is None else walk_mode
        return cls(travel=params.mode(mode).marginal_utility_of_traveling - perf,
                   waiting=params.marginal_utility_of_waiting_pt - perf,
                   walk=params.mode(walk_mode).marginal_utility_of_traveling - perf,
                   line_switch=params.utility_of_line_switch)

    def costs(self) -> tuple[float, float, float, float]:
        """``(travel, wait, walk)`` cost per second and cost per transfer."""
        return (-self.travel / 3600.0, -self.waiting / 3600.0, -self.walk / 3600.0,
                -self.line_switch)


class PersonRaptorWeights:
    """Per-person transit weights: ``sets`` (distinct :class:`RaptorWeights`)
    and ``codes[i]``, person ``i``'s index into ``sets`` (``-1``: the
    router's own weights). Compact and picklable for worker processes."""

    __slots__ = ("sets", "codes")

    def __init__(self, sets: Sequence[RaptorWeights], codes):
        self.sets = tuple(sets)
        self.codes = np.asarray(codes, dtype=np.int32)
        self.codes.flags.writeable = False
        if len(self.codes) and (self.codes.min() < -1 or self.codes.max() >= len(self.sets)):
            raise ValueError("PersonRaptorWeights: code out of range.")

    @classmethod
    def from_sequence(cls, weights: Sequence[Optional[RaptorWeights]]) -> "PersonRaptorWeights":
        """From one :class:`RaptorWeights` (or ``None``) per person."""
        index: dict = {}
        codes = [-1 if w is None else index.setdefault(w, len(index)) for w in weights]
        return cls(tuple(index), codes)

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def any(self) -> bool:
        """Whether anyone has weights of their own."""
        return bool(len(self.sets)) and bool((self.codes >= 0).any())

    def of_code(self, code: int) -> Optional[RaptorWeights]:
        return None if code < 0 else self.sets[code]

    def of(self, person_index: int) -> Optional[RaptorWeights]:
        return self.of_code(int(self.codes[person_index]))

    def __getstate__(self):
        return (self.sets, self.codes)

    def __setstate__(self, state):
        self.sets, self.codes = state

    def __repr__(self) -> str:
        return f"PersonRaptorWeights(persons={len(self)}, sets={len(self.sets)})"


@dataclass(frozen=True)
class RaptorParams:
    search_radius: float = 1000.0  # access and egress search radius [m]
    extension_radius: float = 200.0  # added to the nearest stop's distance if none is in range [m]
    max_transfer_distance: float = 100.0  # MATSim's maxBeelineWalkConnectionDistance [m]
    min_transfer_time: float = 0.0  # [s], also for same-stop transfers
    max_transfers: int = 20
    max_transfers_after_first_arrival: int = 2
    travel: float = -6.0  # in-vehicle [util/h]
    waiting: float = -6.0  # [util/h]
    walk: float = -12.0  # [util/h]
    line_switch: float = -1.0  # per transfer [util]

    @property
    def weights(self) -> RaptorWeights:
        return RaptorWeights(self.travel, self.waiting, self.walk, self.line_switch)

    @classmethod
    def from_scoring(cls, params, mode: str = "pt", walk_mode: Optional[str] = None,
                     **kw) -> "RaptorParams":
        """Utilities from a :class:`~mobslim.scoring.ScoringParams` (see
        :meth:`RaptorWeights.from_scoring`). ``kw`` sets the other fields
        (search radii, transfers)."""
        w = RaptorWeights.from_scoring(params, mode, walk_mode)
        return cls(travel=w.travel, waiting=w.waiting, walk=w.walk,
                   line_switch=w.line_switch, **kw)


class RaptorData:
    """Derived tables for RAPTOR, from a :class:`Transit` and a :class:`RaptorParams`.

    * per route ``r``: ``route_stops[r]`` (stop of each position),
      ``arr[r][trip][i]`` / ``dep[r][trip][i]`` (stop times, trips sorted by
      departure), ``dep_cols[r][i]`` (departure times at position ``i``
      over trips, sorted, for a binary search);
    * ``stop_rs[s]``: ``(route, position)`` pairs at stop ``s``;
    * ``transfers[s]``: ``(to_stop, walk_time, transfer_time, distance)``,
      where ``transfer_time = max(walk_time, minimal transfer time)``.
    """

    def __init__(self, transit: Transit, params: RaptorParams):
        s = transit.schedule
        walk = transit.params.walk
        self.walk_speed = walk.speed
        self.walk_factor = walk.beeline_distance_factor
        self.stop_xy = np.ascontiguousarray(s.stop_xy, dtype=float)
        self.rs_offsets = s.route_stop_offsets.tolist()
        self.rs_distance = s.rs_distance.tolist()
        rs_stop = s.rs_stop.tolist()
        rs_arr, rs_dep = s.rs_arrival.tolist(), s.rs_departure.tolist()
        dep_time, dep_off = s.dep_time.tolist(), s.route_dep_offsets.tolist()
        self.route_stops, self.arr, self.dep, self.dep_cols = [], [], [], []
        self.stop_rs = [[] for _ in range(s.n_stops)]
        for r in range(s.n_routes):
            k0, k1 = self.rs_offsets[r], self.rs_offsets[r + 1]
            stops = rs_stop[k0:k1]
            self.route_stops.append(stops)
            for i, st in enumerate(stops):
                self.stop_rs[st].append((r, i))
            trips = dep_time[dep_off[r]:dep_off[r + 1]]
            self.arr.append([[t + rs_arr[k] for k in range(k0, k1)] for t in trips])
            dep = [[t + rs_dep[k] for k in range(k0, k1)] for t in trips]
            self.dep.append(dep)
            self.dep_cols.append([[row[i] for row in dep] for i in range(k1 - k0)])
        self.transfers = transfer_table(s, walk, params)

    def walk_time(self, distance: float) -> float:
        return distance * self.walk_factor / self.walk_speed

    def nearby_stops(self, x: float, y: float, radius: float, extension: float):
        """``[(stop, beeline distance)]`` within ``radius``, or within the
        nearest stop's distance + ``extension`` if none is in range."""
        if not len(self.stop_xy):
            return []
        dx, dy = self.stop_xy[:, 0] - x, self.stop_xy[:, 1] - y
        d = np.sqrt(dx * dx + dy * dy)
        hit = np.flatnonzero(d <= radius)
        if not len(hit):
            hit = np.flatnonzero(d <= d.min() + extension)
        return list(zip(hit.tolist(), d[hit].tolist()))


def transfer_table(s, walk, params: RaptorParams) -> list:
    """Footpaths per stop: ``[(to_stop, walk_time, transfer_time, distance)]``.

    Every stop has itself (walk 0, the stop's minimal transfer time), then
    the stops within ``max_transfer_distance`` and those with a minimal
    transfer time to it, ordered by target stop. ``transfer_time =
    max(walk_time, minimal transfer time)``; distances include the walk's
    beeline factor.
    """
    n = s.n_stops
    xy = np.ascontiguousarray(s.stop_xy, dtype=float)
    mtt = {(a, b): t for a, b, t in zip(s.mtt_from.tolist(), s.mtt_to.tolist(), s.mtt_time.tolist())}
    pairs: dict = {}
    cell = max(params.max_transfer_distance, 1e-9)
    grid: dict = {}
    keys = np.floor(xy / cell).astype(np.int64).tolist() if n else []
    for i, (cx, cy) in enumerate(keys):
        grid.setdefault((cx, cy), []).append(i)
    max_d = params.max_transfer_distance
    for i, (cx, cy) in enumerate(keys):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    if j == i:
                        continue
                    d = math.hypot(*(xy[i] - xy[j]).tolist())
                    if d <= max_d:
                        pairs[(i, j)] = d
    for (a, b) in mtt:
        if a != b and (a, b) not in pairs:
            pairs[(a, b)] = math.hypot(*(xy[a] - xy[b]).tolist())
    out = [[] for _ in range(n)]
    for i in range(n):
        out[i].append((i, 0.0, mtt.get((i, i), params.min_transfer_time), 0.0))
    factor, speed = walk.beeline_distance_factor, walk.speed
    for (a, b), d in sorted(pairs.items()):
        w = d * factor / speed
        out[a].append((b, w, max(w, mtt.get((a, b), params.min_transfer_time)), d * factor))
    return out


# label: (kind, cost, time, prev, info). kinds: access, transfer, ride
_ACCESS, _TRANSFER, _RIDE = 0, 1, 2


class RaptorRouter:
    """Least-cost transit router (the Python reference; see :class:`RustRaptorRouter`) with the :class:`~mobslim.routing.base.Router`
    interface (between nodes) plus ``route_xy`` / ``route_many_xy`` (between
    coordinates, which is what ``route_plans`` uses for activities).

    ``route_many_xy(..., weights=)`` takes one :class:`RaptorWeights` (or
    ``None``: the params's) per query, for person-specific routing; the
    tables are shared, only the search differs.
    """

    def __init__(self, network: Network, transit: Transit, params: RaptorParams = RaptorParams()):
        self.network = network
        self.transit = transit
        self.params = params
        self.data = RaptorData(transit, params)
        self._costs = params.weights.costs()

    # -- interface ----------------------------------------------------------
    def route(self, origin: int, destination: int, departure_time: float) -> TransitPassengerRoute:
        xy = self.network.node_xy
        return self.route_xy(tuple(xy[origin].tolist()), tuple(xy[destination].tolist()), departure_time)

    def route_many(self, origins: Sequence[int], destinations: Sequence[int],
                   departure_times: Sequence[float]) -> list[TransitPassengerRoute]:
        xy = self.network.node_xy.tolist()
        return self.route_many_xy([tuple(xy[o]) for o in origins],
                                  [tuple(xy[d]) for d in destinations], departure_times)

    def route_xy(self, origin_xy, destination_xy, departure_time: float) -> TransitPassengerRoute:
        return self._search(tuple(origin_xy), float(departure_time), [tuple(destination_xy)])[0]

    def route_many_xy(self, origins_xy, destinations_xy, departure_times,
                      weights: Optional[Sequence[Optional[RaptorWeights]]] = None) -> list[TransitPassengerRoute]:
        """``[route_xy(o, d, t) for ...]``: one search per distinct ``(origin,
        time, weights)``. ``weights``: per query (``None``: the params's)."""
        groups: dict = {}
        if weights is None:
            weights = [None] * len(origins_xy)
        for q, (o, t, w) in enumerate(zip(origins_xy, departure_times, weights)):
            groups.setdefault((tuple(o), float(t), w), []).append(q)
        out: list = [None] * len(origins_xy)
        for (o, t, w), queries in groups.items():
            costs = self._costs if w is None else w.costs()
            routes = self._search(o, t, [tuple(destinations_xy[q]) for q in queries], costs)
            for q, route in zip(queries, routes):
                out[q] = route
        return out

    # -- search ---------------------------------------------------------------
    def _search(self, origin, t0: float, dests: list, costs=None) -> list[TransitPassengerRoute]:
        data, cfg = self.data, self.params
        c_travel, c_wait, c_walk, c_switch = self._costs if costs is None else costs
        factor = data.walk_factor
        n_dest = len(dests)

        # direct walks: the initial answer (and pruning bound) per destination
        best_cost, best = [], []  # best[j]: None = walk, else (ride label, egress stop)
        walks = []
        for dx, dy in dests:
            ex, ey = dx - origin[0], dy - origin[1]
            dist = math.sqrt(ex * ex + ey * ey) * factor
            wt = dist / data.walk_speed
            walks.append((wt, dist))
            best_cost.append(wt * c_walk)
            best.append(None)
        # egress: stop -> [(dest, walk time, walk cost, walk distance)]
        egress: dict = {}
        for j, (dx, dy) in enumerate(dests):
            for st, d in data.nearby_stops(dx, dy, cfg.search_radius, cfg.extension_radius):
                wt = data.walk_time(d)
                egress.setdefault(st, []).append((j, wt, wt * c_walk, d * factor))
        active = [True] * n_dest
        first_round = [None] * n_dest

        def bound() -> float:
            costs = [c for c, a in zip(best_cost, active) if a]
            return max(costs) if costs else -math.inf

        stop_label: dict = {}  # stop -> boardable label
        marked = set()
        for st, d in data.nearby_stops(origin[0], origin[1], cfg.search_radius, cfg.extension_radius):
            wt = data.walk_time(d)
            stop_label[st] = (_ACCESS, wt * c_walk, t0 + wt, None, (st, d * factor))
            marked.add(st)
        rs_best: dict = {}  # route stop -> best arrival cost
        arr_label: dict = {}  # stop -> best vehicle arrival label
        rs_offsets, stop_rs = data.rs_offsets, data.stop_rs
        B = bound()

        for rnd in range(1, cfg.max_transfers + 2):
            if not marked or B == -math.inf:
                break
            scan: dict = {}
            for st in marked:
                for r, i in stop_rs[st]:
                    if i < scan.get(r, math.inf):
                        scan[r] = i
            improved = set()
            for r in sorted(scan):
                stops, arr, dep, cols = data.route_stops[r], data.arr[r], data.dep[r], data.dep_cols[r]
                k0 = rs_offsets[r]
                board = None  # (label, position, trip, cost at departure, departure time)
                for i in range(scan[r], len(stops)):
                    st = stops[i]
                    k = k0 + i
                    if board is not None:
                        blab, bpos, trip, bcost, bdep = board
                        ta = arr[trip][i]
                        c = bcost + (ta - bdep) * c_travel
                        if c < B and c < rs_best.get(k, math.inf):
                            rs_best[k] = c
                            elem = (_RIDE, c, ta, blab, (k0 + bpos, k))
                            cur = arr_label.get(st)
                            if cur is None or c < cur[1]:
                                arr_label[st] = elem
                                improved.add(st)
                            for j, wt, wc, wd in egress.get(st, ()):
                                if active[j] and c + wc < best_cost[j]:
                                    best_cost[j] = c + wc
                                    best[j] = (elem, wt, wd)
                                    if first_round[j] is None:
                                        first_round[j] = rnd
                                    B = bound()
                    if st in marked:
                        lab = stop_label[st]
                        col = cols[i]
                        idx = bisect_left(col, lab[2])
                        if idx < len(col):
                            td = col[idx]
                            nc = lab[1] + (td - lab[2]) * c_wait
                            if nc < B and (board is None or
                                           nc < board[3] + (dep[board[2]][i] - board[4]) * c_travel):
                                board = (lab, i, idx, nc, td)
            # footpaths (transfers) from stops reached by vehicle this round
            new_marked = set()
            for st in sorted(improved):
                e = arr_label[st]
                for to, wt, tt, wd in data.transfers[st]:
                    c = e[1] + wt * c_walk + (tt - wt) * c_wait + c_switch
                    if c >= B:
                        continue
                    cur = stop_label.get(to)
                    if cur is None or c < cur[1]:
                        stop_label[to] = (_TRANSFER, c, e[2] + tt, e, (st, to, wd))
                        new_marked.add(to)
            marked = new_marked
            for j in range(n_dest):
                if active[j] and first_round[j] is not None and \
                        rnd >= first_round[j] + cfg.max_transfers_after_first_arrival:
                    active[j] = False
            B = bound()

        return [self._result(best[j], walks[j], t0) for j in range(n_dest)]

    def _result(self, found, walk, t0: float) -> TransitPassengerRoute:
        if found is None:
            return TransitPassengerRoute.packed(b"", walk[0], walk[1])
        elem, egress_time, egress_dist = found
        rides, walks = [], []  # walks: access, then one per transfer
        e = elem
        while e is not None:
            if e[0] == _RIDE:
                rides.append(e[4])
            elif e[0] == _TRANSFER:
                walks.append(e[4][2])
            else:
                walks.append(e[4][1])
            e = e[3]
        rides.reverse()
        walks.reverse()
        dist_rs = self.data.rs_distance
        distance = walks[0]
        for n, (b, a) in enumerate(rides):
            distance += dist_rs[a] - dist_rs[b]
            if n + 1 < len(rides):
                distance += walks[n + 1]
        distance += egress_dist
        data = np.array(rides, dtype=_I32).tobytes()
        return TransitPassengerRoute.packed(data, elem[2] + egress_time - t0, distance)


# ---------------------------------------------------------------- compiled


def rust_raptor_available() -> bool:
    """Whether the compiled RAPTOR (``mobslim_router.Raptor``) is installed."""
    return _rust is not None and hasattr(_rust, "Raptor")


def _kernel_tables(transit: Transit, params: RaptorParams):
    """The compiled router, built from the schedule (see ``rust/mobslim-router/src/raptor.rs``)."""
    s = transit.schedule
    walk = transit.params.walk
    i8, f8 = np.int64, np.float64
    rso, dep_off = s.route_stop_offsets.astype(i8), s.route_dep_offsets.astype(i8)
    rs_arr, rs_dep = s.rs_arrival.astype(f8), s.rs_departure.astype(f8)
    dep_time = s.dep_time.astype(f8)
    arr, dep, cols, n_trips = [], [], [], []
    for r in range(s.n_routes):
        k0, k1 = rso[r], rso[r + 1]
        trips = dep_time[dep_off[r]:dep_off[r + 1]]
        a = trips[:, None] + rs_arr[k0:k1][None, :]
        d = trips[:, None] + rs_dep[k0:k1][None, :]
        arr.append(a.ravel())
        dep.append(d.ravel())
        cols.append(d.T.ravel())
        n_trips.append(len(trips))

    def cat(parts):
        return np.ascontiguousarray(np.concatenate(parts) if parts else np.zeros(0), dtype=f8)

    rs_stop = s.rs_stop.astype(i8)
    order = np.argsort(rs_stop, kind="stable")  # by stop, then route and position
    rs_route = s.rs_route.astype(i8)
    stop_rs_offsets = np.concatenate([[0], np.cumsum(np.bincount(rs_stop, minlength=s.n_stops))])
    transfers = transfer_table(s, walk, params)
    flat = [t for row in transfers for t in row]
    tr_offsets = np.concatenate([[0], np.cumsum([len(row) for row in transfers])]).astype(i8)
    tr = np.array(flat, dtype=f8).reshape(-1, 4)
    xy = np.ascontiguousarray(s.stop_xy, dtype=f8).reshape(-1, 2)
    return _rust.Raptor(
        np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1]), rso,
        rs_stop, s.rs_distance.astype(f8), np.array(n_trips, dtype=i8), cat(arr), cat(dep),
        cat(cols), stop_rs_offsets.astype(i8), rs_route[order],
        (order - rso[rs_route[order]]).astype(i8), tr_offsets, tr[:, 0].astype(i8),
        np.ascontiguousarray(tr[:, 1]), np.ascontiguousarray(tr[:, 2]),
        np.ascontiguousarray(tr[:, 3]),
        walk_speed=float(walk.speed), walk_factor=float(walk.beeline_distance_factor),
        radius=float(params.search_radius), extension=float(params.extension_radius),
        max_transfers=int(params.max_transfers),
        after_first=int(params.max_transfers_after_first_arrival),
        **dict(zip(("c_travel", "c_wait", "c_walk", "c_switch"), params.weights.costs())),
    )


class RustRaptorRouter:
    """:class:`RaptorRouter` on the compiled kernel (``rust/mobslim-router``).

    Same interface and *identical* routes; ``route_many`` / ``route_many_xy``
    release the GIL and run over ``threads`` threads (default
    ``$MOBSLIM_ROUTER_THREADS`` or the number of CPUs).
    """

    def __init__(self, network: Network, transit: Transit, params: RaptorParams = RaptorParams(),
                 threads: Optional[int] = None):
        if not rust_raptor_available():
            raise ImportError(
                "RustRaptorRouter needs the mobslim_router extension "
                "(rust/mobslim-router; `uv sync` builds it when cargo is installed)."
            )
        from mobslim.routing.compiled import default_threads

        self.network = network
        self.transit = transit
        self.params = params
        self.threads = threads or default_threads()
        self._kernel = _kernel_tables(transit, params)

    def route(self, origin: int, destination: int, departure_time: float) -> TransitPassengerRoute:
        return self.route_many([origin], [destination], [departure_time])[0]

    def route_many(self, origins: Sequence[int], destinations: Sequence[int],
                   departure_times: Sequence[float]) -> list[TransitPassengerRoute]:
        xy = self.network.node_xy
        return self.route_many_xy(xy[np.asarray(origins, dtype=np.int64)],
                                  xy[np.asarray(destinations, dtype=np.int64)], departure_times)

    def route_xy(self, origin_xy, destination_xy, departure_time: float) -> TransitPassengerRoute:
        return self.route_many_xy([origin_xy], [destination_xy], [departure_time])[0]

    def route_many_xy(self, origins_xy, destinations_xy, departure_times,
                      weights: Optional[Sequence[Optional[RaptorWeights]]] = None) -> list[TransitPassengerRoute]:
        o = np.asarray(origins_xy, dtype=np.float64).reshape(-1, 2)
        d = np.asarray(destinations_xy, dtype=np.float64).reshape(-1, 2)
        t = np.ascontiguousarray(departure_times, dtype=np.float64)
        costs = None
        if weights is not None and any(w is not None for w in weights):
            own = self.params.weights.costs()
            costs = np.array([own if w is None else w.costs() for w in weights],
                             dtype=np.float64).reshape(-1, 4)
        offsets, rides, tt, dist = self._kernel.route_many(
            np.ascontiguousarray(o[:, 0]), np.ascontiguousarray(o[:, 1]), t,
            np.ascontiguousarray(d[:, 0]), np.ascontiguousarray(d[:, 1]), threads=self.threads,
            costs=costs)
        data = np.asarray(rides, dtype=_I32).tobytes()
        off = (np.asarray(offsets) * 8).tolist()  # 2 int32 per ride
        return [TransitPassengerRoute.packed(data[off[q]:off[q + 1]], tt_q, d_q)
                for q, (tt_q, d_q) in enumerate(zip(np.asarray(tt).tolist(),
                                                     np.asarray(dist).tolist()))]
