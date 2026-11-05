"""Routing interfaces.

* :class:`TravelTime` answers "how long does link ``l`` take if entered at
  time ``t``?". Implementations: free speed, or a binned table learned from
  events (:mod:`mobslim.routing.traveltime`).
* :class:`Router` finds a least-cost :class:`~mobslim.population.Route`
  between two nodes for one mode.
* A *router factory* builds a router for a mode from a network and travel
  times. Replanning holds a factory, not a router, so routers can be rebuilt
  inside worker processes (and swapped for compiled implementations).

Routers may also offer ``route_many(origins, destinations, departure_times)
-> list[Route]`` to answer a batch at once (sharing searches, or in
parallel); :func:`route_many` falls back to one ``route`` call per query.

Routers go from node to node. :func:`route_many_to_links` makes a leg end on
its activity's link (``Activity.link``), as MATSim's routes do, with any
router: it routes to the link's from-node and appends the link.
"""

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from mobslim.network import Network
from mobslim.population import Route


class NoRouteError(RuntimeError):
    pass


@runtime_checkable
class TravelTime(Protocol):
    """Link travel times, possibly time-dependent.

    ``link_travel_times(time)`` returns the travel time of *every* link when
    entered at ``time``, indexable by link index. Routers call it once per
    settled node rather than once per link, so implementations should return a
    cached sequence (not build a new one per call) and the caller must not
    mutate it. ``link_travel_time`` is the single-link convenience form.
    """

    def link_travel_time(self, link: int, time: float) -> float: ...

    def link_travel_times(self, time: float) -> Sequence[float]: ...


@runtime_checkable
class Router(Protocol):
    def route(self, origin: int, destination: int, departure_time: float) -> Route: ...


class RouterFactory(Protocol):
    def __call__(self, network: Network, mode: str, travel_time: TravelTime) -> Router: ...


def route_many(
    router: Router,
    origins: Sequence[int],
    destinations: Sequence[int],
    departure_times: Sequence[float],
) -> list[Route]:
    """Route a batch of queries with ``router.route_many`` if it has one."""
    many = getattr(router, "route_many", None)
    if many is not None:
        return many(origins, destinations, departure_times)
    return [router.route(o, d, t) for o, d, t in zip(origins, destinations, departure_times)]


_I32 = np.dtype("=i4")


def arrival_link(network: Network, mode: str, origin: int, origin_link, destination: int,
                 destination_link):
    """The link a ``mode`` leg must end by driving through, or ``None``.

    That is the destination activity's link, unless there is none, the leg
    starts on it (MATSim: an empty route), the origin has no link and is at
    the destination node already, or the link does not allow ``mode``.
    """
    if destination_link is None or destination_link == origin_link:
        return None
    if origin_link is None and origin == destination:
        return None
    if not network.mode_mask(mode)[destination_link]:
        return None
    return destination_link


def route_many_to_links(
    router: Router,
    network: Network,
    mode: str,
    origins: Sequence[int],
    origin_links: Sequence,
    destinations: Sequence[int],
    destination_links: Sequence,
    departure_times: Sequence[float],
) -> list[Route]:
    """Like :func:`route_many`, but a leg whose destination activity is on a
    link (``destination_links[q]`` not ``None``) ends on that link, as in
    MATSim: it is routed to the link's from-node, then through the link.

    Activity links are optional (``None``); see :func:`arrival_link` for when
    a link is not used. The appended link adds its length and its free-speed
    time (as MATSim's ``NetworkRoutingModule`` estimates the arrival link)
    to the route's distance and travel time.
    """
    from_node = network.from_node
    arrivals = [arrival_link(network, mode, o, ol, d, dl)
                for o, ol, d, dl in zip(origins, origin_links, destinations, destination_links)]
    targets = [d if a is None else int(from_node[a]) for d, a in zip(destinations, arrivals)]
    routes = route_many(router, origins, targets, departure_times)
    if not any(a is not None for a in arrivals):
        return routes
    fstt, length = network.free_speed_travel_times(), network.length
    out = []
    for route, a in zip(routes, arrivals):
        if a is None:
            out.append(route)
        else:
            out.append(Route.packed(route.data + np.array([a], dtype=_I32).tobytes(),
                                    route.travel_time + float(fstt[a]),
                                    route.distance + float(length[a])))
    return out
