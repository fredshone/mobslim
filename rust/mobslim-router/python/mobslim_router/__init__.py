"""Compiled routers for :mod:`mobslim.routing` (see ``mobslim.routing.compiled``).

``Router`` holds one mode's adjacency and travel-time columns and answers
queries with time-dependent Dijkstra (identical to
``mobslim.routing.DijkstraRouter``) or ALT. ``Raptor`` is the least-cost
transit router (identical to ``mobslim.transit.RaptorRouter``). ``PointIndex``
answers exact nearest-point queries (``mobslim.activity_links``).
``accumulate_traversals`` adds link traversals into travel-time sums
(``mobslim.routing.traveltime.TraversalAggregate.accumulate``).
"""

from mobslim_router._kernel import PointIndex, Raptor, Router, accumulate_traversals

__all__ = ["PointIndex", "Raptor", "Router", "accumulate_traversals"]
