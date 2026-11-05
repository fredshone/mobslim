"""The transit schedule as tables: stops, lines, routes, route stops, departures.

Everything is struct-of-arrays with dense int32 indices and CSR slices, like
:class:`~mobslim.network.Network` and :class:`~mobslim.planbatch.PlanBatch`:

=============  ==============  ==============================================
table          row             columns
=============  ==============  ==============================================
stops          stop ``s``      ``stop_ids``, ``stop_xy``, ``stop_link`` (-1 =
                               none), ``stop_blocking``
lines          line ``l``      ``line_ids``
routes         route ``r``     ``route_ids`` (``(line_id, route_id)``: route
                               ids are unique only per line), ``route_line``,
                               ``route_mode`` (code into ``modes``), and CSR
                               offsets into route stops, route links and
                               departures
route stops    ``k`` (global)  ``rs_stop``, ``rs_arrival``, ``rs_departure``
                               (offsets from the departure), ``rs_await``,
                               ``rs_link_pos`` (index in the route's links of
                               the stop's link), ``rs_distance`` (metres along
                               the route to the end of that link)
route links                    ``route_links``
departures     trip ``d``      ``dep_ids`` (``(line, route, id)``),
                               ``dep_time`` (sorted within each route),
                               ``dep_vehicle`` (into ``vehicle_ids``)
transfers                      ``mtt_from``, ``mtt_to``, ``mtt_time``
                               (MATSim's ``minimalTransferTimes``)
=============  ==============  ==============================================

Stop times are never stored: :meth:`TransitSchedule.stop_times` derives the
``(n_departures, n_stops)`` matrix of a route as ``dep_time + offset``.
Vehicles are referenced by id (``vehicle_ids``) and resolved against a
:class:`~mobslim.transit.vehicles.TransitVehicles` when the two are put
together in a :class:`~mobslim.transit.Transit`.

Arrays are read-only. Derived views are cached and dropped when pickling.
"""

from dataclasses import dataclass, field, fields
from typing import Hashable, Iterable, Sequence

import numpy as np

from mobslim.ids import IdMap

_ARRAYS = (
    "stop_xy", "stop_link", "stop_blocking", "route_line", "route_mode",
    "route_stop_offsets", "route_link_offsets", "route_dep_offsets",
    "rs_stop", "rs_arrival", "rs_departure", "rs_await", "rs_link_pos", "rs_distance",
    "route_links", "dep_time", "dep_vehicle", "mtt_from", "mtt_to", "mtt_time",
)


@dataclass(frozen=True, eq=False)
class TransitSchedule:
    stop_ids: IdMap
    stop_xy: np.ndarray  # (n_stops, 2) f8
    stop_link: np.ndarray  # (n_stops,) i4, -1 = none
    stop_blocking: np.ndarray  # (n_stops,) bool
    line_ids: IdMap
    route_ids: IdMap  # of (line_id, route_id)
    route_line: np.ndarray  # (n_routes,) i4
    route_mode: np.ndarray  # (n_routes,) i4, code into modes
    modes: tuple[str, ...]
    route_stop_offsets: np.ndarray  # (n_routes + 1,) i8
    route_link_offsets: np.ndarray  # (n_routes + 1,) i8
    route_dep_offsets: np.ndarray  # (n_routes + 1,) i8
    rs_stop: np.ndarray  # (n_route_stops,) i4
    rs_arrival: np.ndarray  # f8 offset [s]
    rs_departure: np.ndarray  # f8 offset [s]
    rs_await: np.ndarray  # bool
    rs_link_pos: np.ndarray  # i4, index into the route's links
    rs_distance: np.ndarray  # f8 [m] along the route to the end of the stop's link
    route_links: np.ndarray  # (n_route_links,) i4
    dep_ids: IdMap  # of (line_id, route_id, departure_id)
    dep_time: np.ndarray  # (n_departures,) f8
    dep_vehicle: np.ndarray  # (n_departures,) i4, into vehicle_ids
    vehicle_ids: IdMap
    mtt_from: np.ndarray  # i4 stop
    mtt_to: np.ndarray  # i4 stop
    mtt_time: np.ndarray  # f8 [s]
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for name in _ARRAYS:
            getattr(self, name).flags.writeable = False

    def __getstate__(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "_cache"}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_cache", {})

    # -- sizes ----------------------------------------------------------------
    @property
    def n_stops(self) -> int:
        return len(self.stop_ids)

    @property
    def n_lines(self) -> int:
        return len(self.line_ids)

    @property
    def n_routes(self) -> int:
        return len(self.route_line)

    @property
    def n_route_stops(self) -> int:
        return len(self.rs_stop)

    @property
    def n_departures(self) -> int:
        return len(self.dep_time)

    # -- derived indices (cached) ---------------------------------------------
    def _cached(self, key, make):
        value = self._cache.get(key)
        if value is None:
            value = self._cache[key] = make()
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        return value

    @property
    def rs_route(self) -> np.ndarray:
        """Route of each route stop."""
        return self._cached("rs_route", lambda: np.repeat(
            np.arange(self.n_routes, dtype=np.int32), np.diff(self.route_stop_offsets)))

    @property
    def rs_index_in_route(self) -> np.ndarray:
        """Position of each route stop within its route (0 = first)."""
        return self._cached("rs_index", lambda: (
            np.arange(self.n_route_stops) - self.route_stop_offsets[self.rs_route]).astype(np.int32))

    @property
    def dep_route(self) -> np.ndarray:
        """Route of each departure."""
        return self._cached("dep_route", lambda: np.repeat(
            np.arange(self.n_routes, dtype=np.int32), np.diff(self.route_dep_offsets)))

    @property
    def stop_route_stops(self) -> tuple[np.ndarray, np.ndarray]:
        """CSR ``(offsets, route_stops)``: the route stops at each stop."""
        def make():
            order = np.argsort(self.rs_stop, kind="stable")
            counts = np.bincount(self.rs_stop, minlength=self.n_stops)
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            offsets.flags.writeable = False
            order = order.astype(np.int32)
            order.flags.writeable = False
            return offsets, order
        return self._cached("stop_rs", make)

    def stop_times(self, route: int, kind: str = "arrival") -> np.ndarray:
        """``(n_departures_r, n_stops_r)`` matrix ``dep_time + offset`` of
        ``route`` (``kind`` is ``"arrival"`` or ``"departure"``). Not stored."""
        offsets = {"arrival": self.rs_arrival, "departure": self.rs_departure}[kind]
        k0, k1 = self.route_stop_offsets[route], self.route_stop_offsets[route + 1]
        d0, d1 = self.route_dep_offsets[route], self.route_dep_offsets[route + 1]
        return self.dep_time[d0:d1, None] + offsets[None, k0:k1]

    # -- lookups ----------------------------------------------------------------
    def route_index(self, line: Hashable, route: Hashable) -> int:
        return self.route_ids.index((line, route))

    def route_stop(self, line: Hashable, route: Hashable, stop: Hashable) -> int:
        """Global index of the (first) route stop of ``route`` at ``stop``."""
        r = self.route_index(line, route)
        s = self.stop_ids.index(stop)
        k0, k1 = self.route_stop_offsets[r], self.route_stop_offsets[r + 1]
        hits = np.flatnonzero(self.rs_stop[k0:k1] == s)
        if not len(hits):
            raise KeyError(f"Route {line}/{route} does not serve stop {stop!r}.")
        return int(k0 + hits[0])

    def route_stops_of(self, route: int) -> dict[str, np.ndarray]:
        """Column slices of ``route``'s route stops: ``stop``, ``arrival``,
        ``departure``, ``await``, ``link_pos`` and ``distance``."""
        k0, k1 = self.route_stop_offsets[route], self.route_stop_offsets[route + 1]
        return {"stop": self.rs_stop[k0:k1], "arrival": self.rs_arrival[k0:k1],
                "departure": self.rs_departure[k0:k1], "await": self.rs_await[k0:k1],
                "link_pos": self.rs_link_pos[k0:k1], "distance": self.rs_distance[k0:k1]}

    def route_links_of(self, route: int) -> np.ndarray:
        return self.route_links[self.route_link_offsets[route]:self.route_link_offsets[route + 1]]

    def route_mode_name(self, route: int) -> str:
        return self.modes[self.route_mode[route]]

    def describe_rides(self, rides: Iterable[Sequence[int]]) -> list[tuple]:
        """``[(line_id, route_id, board_stop_id, alight_stop_id), ...]``."""
        out = []
        for board, alight in rides:
            line, route = self.route_ids.id(int(self.rs_route[board]))
            out.append((line, route, self.stop_ids.id(int(self.rs_stop[board])),
                        self.stop_ids.id(int(self.rs_stop[alight]))))
        return out

    def __repr__(self) -> str:
        return (f"TransitSchedule(stops={self.n_stops}, lines={self.n_lines}, "
                f"routes={self.n_routes}, departures={self.n_departures})")


def build_schedule(network, stops: Sequence[tuple], routes: Sequence[dict],
                   transfers: Sequence[tuple] = ()) -> TransitSchedule:
    """Assemble and check a :class:`TransitSchedule`.

    ``stops``: ``(id, x, y, link_id or None, blocking)``. ``routes``: dicts
    with ``line``, ``route``, ``mode``, ``links`` (link ids), ``stops``
    (``(stop_id, arrival, departure, await)``; a missing offset is ``None``
    and copies the other) and ``departures`` (``(id, time, vehicle_id)``).
    ``transfers``: ``(from_stop, to_stop, time)``.

    Raises ``ValueError`` if a route's links are not connected, a stop is
    not on its route's links in route order, offsets decrease, or a stop is
    unknown.
    """
    stop_ids = IdMap()
    stop_xy, stop_link, stop_blocking = [], [], []
    for sid, x, y, link, blocking in stops:
        if sid in stop_ids:
            raise ValueError(f"Duplicate stop id {sid!r}.")
        stop_ids.add(sid)
        stop_xy.append((float(x), float(y)))
        stop_link.append(-1 if link is None else _link(network, link, f"stop {sid!r}"))
        stop_blocking.append(bool(blocking))
    stop_link_arr = np.array(stop_link, dtype=np.int32)

    line_ids, route_ids, vehicle_ids, dep_ids = IdMap(), IdMap(), IdMap(), IdMap()
    mode_code: dict[str, int] = {}
    route_line, route_mode = [], []
    rs_offsets, rl_offsets, dep_offsets = [0], [0], [0]
    rs_stop, rs_arr, rs_dep, rs_await, rs_pos, rs_dist = [], [], [], [], [], []
    route_links, dep_time, dep_vehicle = [], [], []
    to_node, from_node, length = network.to_node, network.from_node, network.length
    for spec in routes:
        line, rid = spec["line"], spec["route"]
        name = f"{line}/{rid}"
        if (line, rid) in route_ids:
            raise ValueError(f"Duplicate transit route {name}.")
        route_ids.add((line, rid))
        route_line.append(line_ids.add(line))
        route_mode.append(mode_code.setdefault(spec["mode"], len(mode_code)))
        links = [_link(network, lk, f"route {name}") for lk in spec["links"]]
        for a, b in zip(links, links[1:]):
            if to_node[a] != from_node[b]:
                raise ValueError(
                    f"Route {name}: links {network.link_ids.id(a)!r} and "
                    f"{network.link_ids.id(b)!r} are not connected.")
        cum = np.cumsum(length[links]) if links else np.zeros(0)
        pos, last_dep = 0, -np.inf
        if not spec["stops"]:
            raise ValueError(f"Route {name} has no stops.")
        for sid, arr, dep, wait in spec["stops"]:
            if sid not in stop_ids:
                raise ValueError(f"Route {name}: unknown stop {sid!r}.")
            s = stop_ids.index(sid)
            if arr is None and dep is None:
                raise ValueError(f"Route {name}: stop {sid!r} has neither an arrival nor a departure offset.")
            arr = dep if arr is None else arr
            dep = arr if dep is None else dep
            if arr > dep or arr < last_dep:
                raise ValueError(f"Route {name}: offsets decrease at stop {sid!r}.")
            last_dep = dep
            link = int(stop_link_arr[s])
            if link < 0:
                raise ValueError(f"Route {name}: stop {sid!r} has no link.")
            while pos < len(links) and links[pos] != link:
                pos += 1
            if pos == len(links):
                raise ValueError(
                    f"Route {name}: stop {sid!r} (on link {network.link_ids.id(link)!r}) "
                    "is not on the route's links, or not in route order.")
            rs_stop.append(s)
            rs_arr.append(float(arr))
            rs_dep.append(float(dep))
            rs_await.append(bool(wait))
            rs_pos.append(pos)
            rs_dist.append(float(cum[pos]))
        route_links.extend(links)
        deps = sorted(spec["departures"], key=lambda d: d[1])  # stable: ties keep file order
        for did, time, vid in deps:
            key = (line, rid, did)
            if key in dep_ids:
                raise ValueError(f"Route {name}: duplicate departure id {did!r}.")
            dep_ids.add(key)
            dep_time.append(float(time))
            dep_vehicle.append(vehicle_ids.add(vid))
        rs_offsets.append(len(rs_stop))
        rl_offsets.append(len(route_links))
        dep_offsets.append(len(dep_time))

    mtt_from, mtt_to, mtt_time = [], [], []
    for a, b, time in transfers:
        for sid in (a, b):
            if sid not in stop_ids:
                raise ValueError(f"Minimal transfer time: unknown stop {sid!r}.")
        mtt_from.append(stop_ids.index(a))
        mtt_to.append(stop_ids.index(b))
        mtt_time.append(float(time))

    i4, i8 = np.int32, np.int64
    return TransitSchedule(
        stop_ids=stop_ids,
        stop_xy=np.array(stop_xy, dtype=float).reshape(-1, 2),
        stop_link=stop_link_arr,
        stop_blocking=np.array(stop_blocking, dtype=bool),
        line_ids=line_ids,
        route_ids=route_ids,
        route_line=np.array(route_line, dtype=i4),
        route_mode=np.array(route_mode, dtype=i4),
        modes=tuple(mode_code),
        route_stop_offsets=np.array(rs_offsets, dtype=i8),
        route_link_offsets=np.array(rl_offsets, dtype=i8),
        route_dep_offsets=np.array(dep_offsets, dtype=i8),
        rs_stop=np.array(rs_stop, dtype=i4),
        rs_arrival=np.array(rs_arr, dtype=float),
        rs_departure=np.array(rs_dep, dtype=float),
        rs_await=np.array(rs_await, dtype=bool),
        rs_link_pos=np.array(rs_pos, dtype=i4),
        rs_distance=np.array(rs_dist, dtype=float),
        route_links=np.array(route_links, dtype=i4),
        dep_ids=dep_ids,
        dep_time=np.array(dep_time, dtype=float),
        dep_vehicle=np.array(dep_vehicle, dtype=i4),
        vehicle_ids=vehicle_ids,
        mtt_from=np.array(mtt_from, dtype=i4),
        mtt_to=np.array(mtt_to, dtype=i4),
        mtt_time=np.array(mtt_time, dtype=float),
    )


def _link(network, link_id, what) -> int:
    try:
        return network.link_index(link_id)
    except KeyError:
        raise ValueError(f"{what}: unknown link {link_id!r}.") from None
