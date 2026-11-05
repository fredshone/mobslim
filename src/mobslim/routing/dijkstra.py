"""Time-dependent Dijkstra over the network's per-mode adjacency.

The cost of a link is its travel time when entered at the label's arrival
time, so routes respond to time-of-day congestion in a
:class:`~mobslim.routing.traveltime.TravelTimeTable`.

All out-links of a settled node are entered at the same time, so the router
asks the travel-time model for the whole link vector once per settled node
(:meth:`TravelTime.link_travel_times`) instead of once per link.

:meth:`DijkstraRouter.route_many` answers many queries at once. Queries that
share an origin and departure time are answered by one search that runs until
every destination is settled. A settled label is final and the heap order is
the same, so that gives exactly the routes one search per query would. On a
*static* travel-time model (``travel_time.is_static``, e.g. free speed) the
departure time does not matter: every search starts at time 0 and all queries
from one origin share a search.

This is a reference implementation in pure Python. The compiled
:class:`~mobslim.routing.compiled.RustRouter` returns identical routes.
Anything with the same ``route(origin, destination, departure_time) -> Route``
signature can be plugged in via a router factory.
"""

import heapq
import math
from typing import Sequence

import numpy as np

from mobslim.network import Network
from mobslim.population import Route
from mobslim.routing.base import NoRouteError, TravelTime

_I32 = np.dtype("=i4")  # Route's packed link format
_EMPTY = Route.packed(b"", 0.0, 0.0)  # origin == destination


class DijkstraRouter:
    def __init__(self, network: Network, mode: str, travel_time: TravelTime):
        self.mode = mode
        offsets, links = network.out_links(mode)
        to = network.to_node.tolist()
        # adjacency per node as ((link, to_node), ...): one tuple unpack per edge
        self._adj = tuple(
            tuple((links[k], to[links[k]]) for k in range(offsets[u], offsets[u + 1]))
            for u in range(network.n_nodes)
        )
        self._from = network.from_node.tolist()
        self._length = network.length.tolist()
        self._times_at = travel_time.link_travel_times
        self.is_static = bool(getattr(travel_time, "is_static", False))

    def _search(self, origin: int, t0: float, targets: set) -> tuple[dict, dict]:
        """Labels and predecessor links until every node in ``targets`` is
        settled (``targets`` is consumed)."""
        adj, times_at = self._adj, self._times_at
        pop, push, inf = heapq.heappop, heapq.heappush, math.inf
        best = {origin: t0}
        get = best.get
        via: dict[int, int] = {}  # node -> link used to reach it
        heap = [(t0, origin)]
        while heap:
            t, u = pop(heap)
            if t > best[u]:
                continue
            if u in targets:
                targets.discard(u)
                if not targets:
                    break
            tt = times_at(t)
            for link, v in adj[u]:
                nt = t + tt[link]
                if nt < get(v, inf):
                    best[v] = nt
                    via[v] = link
                    push(heap, (nt, v))
        return best, via

    def _route(self, best: dict, via: dict, origin: int, destination: int, t0: float) -> Route:
        if destination not in best:
            raise NoRouteError(
                f"No '{self.mode}' route from node {origin} to node {destination}."
            )
        path = []
        node = destination
        while node != origin:
            link = via[node]
            path.append(link)
            node = self._from[link]
        path.reverse()
        length, distance = self._length, 0.0
        for k in path:  # plain left-to-right sum (not sum(): it compensates since 3.12)
            distance += length[k]
        data = np.array(path, dtype=_I32).tobytes()
        return Route.packed(data, best[destination] - t0, distance)

    def route(self, origin: int, destination: int, departure_time: float) -> Route:
        if origin == destination:
            return _EMPTY
        t0 = 0.0 if self.is_static else departure_time
        best, via = self._search(origin, t0, {destination})
        return self._route(best, via, origin, destination, t0)

    def route_many(
        self,
        origins: Sequence[int],
        destinations: Sequence[int],
        departure_times: Sequence[float],
    ) -> list[Route]:
        """``[route(o, d, t) for ...]``, with one search per distinct
        ``(origin, departure_time)`` (per origin on a static model)."""
        groups: dict[tuple, list[int]] = {}
        static = self.is_static
        for q, (o, t) in enumerate(zip(origins, departure_times)):
            groups.setdefault((o, 0.0) if static else (o, t), []).append(q)
        out: list = [None] * len(origins)
        for (o, t0), queries in groups.items():
            targets = {destinations[q] for q in queries} - {o}
            if targets:
                best, via = self._search(o, t0, set(targets))
            for q in queries:
                d = destinations[q]
                out[q] = _EMPTY if d == o else self._route(best, via, o, d, t0)
        return out
