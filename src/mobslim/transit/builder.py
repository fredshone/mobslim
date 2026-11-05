"""``TransitBuilder``: schedule and vehicles in a few lines (tests, synthetic scenarios).

::

    b = TransitBuilder(network)
    b.vehicle_type("bus", seats=40, standing=30, access_time=1.0, egress_time=1.0,
                   doors="serial", pce=1.0)
    b.stop("S0", x, y, link="a-0", blocking=False)
    b.route("L", "r", mode="bus", links=["a-0", "0-1", ...],
            stops=[("S0", 0, 0), ("S1", 120, 120, False), ...])  # (stop, arr, dep[, await=True])
    b.departures("L", "r", times=[...], vehicle_type="bus")      # one vehicle per departure
    b.departures("L", "r", times=[...], vehicles=["v1", ...])    # or explicit vehicles
    b.headway("L", "r", start=6 * H, end=10 * H, every=600, vehicle_type="bus")
    transit = b.build(TransitConfig(...))
"""

from typing import Hashable, Optional, Sequence

import numpy as np

from mobslim.network import Network
from mobslim.transit.core import Transit, TransitConfig
from mobslim.transit.schedule import build_schedule
from mobslim.transit.vehicles import build_vehicles


class TransitBuilder:
    def __init__(self, network: Network):
        self.network = network
        self._types: dict = {}
        self._vehicles: dict = {}
        self._stops: dict = {}
        self._routes: dict = {}
        self._transfers: list = []
        self._auto = 0

    def vehicle_type(self, type_id: Hashable, *, seats: int, standing: int = 0,
                     access_time: float = 1.0, egress_time: float = 1.0, doors: str = "serial",
                     pce: float = 1.0, length: float = 7.5, max_speed: float = np.inf):
        if type_id in self._types:
            raise ValueError(f"Duplicate vehicle type {type_id!r}.")
        self._types[type_id] = dict(id=type_id, seats=seats, standing=standing,
                                    access_time=access_time, egress_time=egress_time,
                                    doors=doors, pce=pce, length=length, max_speed=max_speed)
        return self

    def vehicle(self, vehicle_id: Hashable, type_id: Hashable):
        if vehicle_id in self._vehicles:
            raise ValueError(f"Duplicate vehicle {vehicle_id!r}.")
        self._vehicles[vehicle_id] = type_id
        return self

    def stop(self, stop_id: Hashable, x: float, y: float, *, link: Optional[Hashable] = None,
             blocking: bool = False):
        if stop_id in self._stops:
            raise ValueError(f"Duplicate stop {stop_id!r}.")
        self._stops[stop_id] = (stop_id, x, y, link, blocking)
        return self

    def route(self, line: Hashable, route: Hashable, *, mode: str, links: Sequence[Hashable],
              stops: Sequence[tuple]):
        """``stops``: ``(stop_id, arrival, departure[, await_departure=True])``."""
        if (line, route) in self._routes:
            raise ValueError(f"Duplicate route {line}/{route}.")
        rows = []
        for st in stops:
            sid, arr, dep = st[:3]
            rows.append((sid, arr, dep, st[3] if len(st) > 3 else True))
        self._routes[(line, route)] = dict(line=line, route=route, mode=mode, links=list(links),
                                           stops=rows, departures=[])
        return self

    def departures(self, line: Hashable, route: Hashable, *, times: Sequence[float],
                   vehicle_type: Optional[Hashable] = None,
                   vehicles: Optional[Sequence[Hashable]] = None):
        """Departures at ``times``: each gets a new vehicle of ``vehicle_type``,
        or uses ``vehicles`` (one id per departure, declared with :meth:`vehicle`)."""
        spec = self._routes[(line, route)]
        times = list(times)
        if (vehicle_type is None) == (vehicles is None):
            raise ValueError("departures: give either vehicle_type or vehicles.")
        if vehicles is None:
            if vehicle_type not in self._types:
                raise ValueError(f"Unknown vehicle type {vehicle_type!r}.")
            vehicles = []
            for _ in times:
                vid = f"{line}_{route}_{self._auto}"
                self._auto += 1
                self.vehicle(vid, vehicle_type)
                vehicles.append(vid)
        elif len(vehicles) != len(times):
            raise ValueError("departures: need one vehicle per departure time.")
        for t, vid in zip(times, vehicles):
            spec["departures"].append((f"{len(spec['departures'])}", float(t), vid))
        return self

    def headway(self, line: Hashable, route: Hashable, *, start: float, end: float,
                every: float, vehicle_type: Hashable):
        """Departures from ``start`` to ``end`` (inclusive) every ``every`` seconds."""
        n = int(np.floor((end - start) / every + 1e-9)) + 1
        return self.departures(line, route, times=[start + k * every for k in range(n)],
                               vehicle_type=vehicle_type)

    def minimal_transfer_time(self, from_stop: Hashable, to_stop: Hashable, time: float):
        self._transfers.append((from_stop, to_stop, time))
        return self

    def line_along(self, line: Hashable, route: Hashable, nodes: Sequence[Hashable], *,
                   mode: str = "bus", link_mode: Optional[str] = None, dwell: float = 0.0,
                   await_departure: bool = True, blocking: bool = False):
        """A route through ``nodes`` (node ids), shortest path per hop on
        ``link_mode`` links (default ``mode``) at free speed, with a stop at
        each node on the link that ends there. The first stop is on a link
        into ``nodes[0]``, which becomes the route's first (not traversed)
        link. Offsets are free-speed times plus ``dwell`` per stop. Stops are
        shared by every line that serves the same link (id ``"node@link"``).
        Returns the stop ids."""
        from mobslim.routing.dijkstra import DijkstraRouter
        from mobslim.routing.traveltime import FreeSpeedTravelTime

        net = self.network
        link_mode = link_mode or mode
        router = DijkstraRouter(net, link_mode, FreeSpeedTravelTime(net))
        idx = [net.node_index(n) for n in nodes]
        mask = net.mode_mask(link_mode)
        into = np.flatnonzero(mask & (net.to_node == idx[0]))
        if not len(into):
            raise ValueError(f"line_along: no '{link_mode}' link into node {nodes[0]!r}.")
        links = [int(into[0])]
        stop_links = [links[0]]
        for a, b in zip(idx, idx[1:]):
            hop = router.route(a, b, 0.0).links
            if not hop:
                raise ValueError("line_along: consecutive nodes must differ.")
            links.extend(hop)
            stop_links.append(hop[-1])
        fs = net.free_speed_travel_times()
        stops, t, pos = [], 0.0, 0
        for k, (node, link) in enumerate(zip(nodes, stop_links)):
            if k:
                while links[pos] != link:
                    pos += 1
                    t += float(fs[links[pos]])
            sid = f"{node}@{net.link_ids.id(link)}"
            if sid not in self._stops:
                x, y = net.node_xy[idx[k]].tolist()
                self.stop(sid, x, y, link=net.link_ids.id(link), blocking=blocking)
            stops.append((sid, t, t + dwell, await_departure))
            t += dwell
        self.route(line, route, mode=mode, links=[net.link_ids.id(k) for k in links], stops=stops)
        return [s[0] for s in stops]

    def build(self, config: TransitConfig = TransitConfig()) -> Transit:
        schedule = build_schedule(self.network, list(self._stops.values()),
                                  list(self._routes.values()), self._transfers)
        vehicles = build_vehicles(list(self._types.values()), list(self._vehicles.items()))
        return Transit(schedule, vehicles, config)
