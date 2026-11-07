"""Compiled routers for :mod:`mobslim.routing` (see ``mobslim.routing.compiled``).

``Router`` holds one mode's adjacency and travel-time columns and answers
queries with time-dependent Dijkstra (identical to
``mobslim.routing.DijkstraRouter``) or ALT. ``Raptor`` is the least-cost
transit router (identical to ``mobslim.transit.RaptorRouter``). ``PointIndex``
answers exact nearest-point queries (``mobslim.activity_links``).
"""

from mobslim_router._kernel import PointIndex, Raptor, Router

__all__ = ["PointIndex", "Raptor", "Router"]
