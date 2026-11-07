"""Assigning activities to network links (MATSim's ``XY2Links``).

An activity is a *location* (its coordinates), not a network node. Before
simulation every activity with coordinates but no link is assigned the link
whose midpoint (the average of its two nodes) is nearest, among links that
allow one of the assignment ``modes``. Its node is then the link's to-node,
as in MATSim: a network leg leaves from the end of the link and arrives by
driving through it, while teleported legs and pt walks still use the
activity's own coordinates. Activities that already have a link (e.g. from
a MATSim plans file) keep it, and so do activities placed at a node without
coordinates (e.g. ``PlanBuilder.act(type, node)``), unless
``snap_node_activities``.

Assignment can prefer some links over others with ordered *tiers*: each
tier allows only the links matched by none of its ``exclude`` filters, up to
a ``max_distance``. An activity takes the nearest link of the first tier that
has one within its distance; the last, implicit tier is every eligible link,
at any distance. For example, to keep activities off motorways unless no
other link is within 1 km::

    ActivityLinkParams.avoiding(LinkFilter("type", values=("motorway", "trunk")),
                                unless_beyond=1000.0)

Filters read link attributes (:meth:`~mobslim.network.Network.link_attribute`),
which a MATSim network loads lazily, only when a filter asks for them.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from mobslim.network import DEFAULT_MODE, Network

try:  # rust/mobslim-router; optional
    import mobslim_router as _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None

if TYPE_CHECKING:
    from mobslim.scenario import Scenario


@dataclass(frozen=True)
class LinkFilter:
    """Links whose ``attribute`` is one of ``values`` (compared as strings),
    or numerically within ``[min, max]`` (either bound optional). Links
    without the attribute never match, nor does an attribute the network
    lacks altogether."""

    attribute: str
    values: tuple = ()
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(str(v) for v in self.values))
        if not self.values and self.min is None and self.max is None:
            raise ValueError(f"LinkFilter({self.attribute!r}) needs values, min or max.")

    def mask(self, network: Network) -> np.ndarray:
        out = np.zeros(network.n_links, dtype=bool)
        if not network.has_link_attribute(self.attribute):
            return out
        col = network.link_attribute(self.attribute).tolist()
        if self.values:
            wanted = set(self.values)
            out |= np.array([v is not None and str(v) in wanted for v in col], dtype=bool)
        if self.min is not None or self.max is not None:
            num = np.array([_number(v) for v in col], dtype=float)
            ok = ~np.isnan(num)
            if self.min is not None:
                ok &= num >= self.min
            if self.max is not None:
                ok &= num <= self.max
            out |= ok
        return out


def _number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


@dataclass(frozen=True)
class SnapTier:
    """Links matched by none of ``exclude``, within ``max_distance`` [m]."""

    exclude: tuple[LinkFilter, ...] = ()
    max_distance: float = math.inf

    def __post_init__(self):
        object.__setattr__(self, "exclude", tuple(self.exclude))
        if not self.max_distance >= 0:
            raise ValueError("SnapTier.max_distance must be >= 0.")


@dataclass(frozen=True)
class ActivityLinkParams:
    """How activities are assigned to links.

    ``modes``: a link is eligible if it allows one of them (MATSim snaps to
    the car network); ``None``, or modes no link allows: every link.
    ``tiers``: preferred subsets tried in order before the implicit last
    tier (every eligible link, any distance). ``snap_node_activities``: also
    assign activities placed at a node without coordinates (from the node's).
    """

    modes: Optional[tuple[str, ...]] = (DEFAULT_MODE,)
    tiers: tuple[SnapTier, ...] = ()
    snap_node_activities: bool = False

    def __post_init__(self):
        if self.modes is not None:
            object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "tiers", tuple(self.tiers))

    @classmethod
    def avoiding(cls, *filters: LinkFilter, unless_beyond: float = 1000.0,
                 **kwargs) -> "ActivityLinkParams":
        """Never assign to links matched by ``filters`` unless no other
        eligible link is within ``unless_beyond`` metres."""
        return cls(tiers=(SnapTier(filters, unless_beyond),), **kwargs)

    def eligible(self, network: Network) -> np.ndarray:
        """Bool per link: may an activity be assigned to it at all?"""
        mask = np.zeros(network.n_links, dtype=bool)
        for m in self.modes or ():
            if m in network.modes:
                mask |= network.mode_mask(m)
        return mask if mask.any() else np.ones(network.n_links, dtype=bool)


def link_midpoints(network: Network) -> np.ndarray:
    """``(n_links, 2)``: the average of each link's two node locations."""
    xy = network.node_xy
    return (xy[network.from_node] + xy[network.to_node]) / 2


SNAP_BACKENDS = ("rust", "quadtree")


def rust_snapper_available() -> bool:
    """Whether the compiled point index (``mobslim_router.PointIndex``) is installed."""
    return _rust is not None and hasattr(_rust, "PointIndex")


class _QuadTreeIndex:
    """Nearest point with ``fastquadtree`` (one query at a time)."""

    def __init__(self, xy: np.ndarray, bounds):
        from fastquadtree import QuadTree

        # f64: real-world (e.g. UTM) coordinates lose metre precision in f32
        self.qt = QuadTree(bounds, capacity=16, dtype="f64")
        self.qt.insert_many_np(xy)  # ids 0..k-1, in order

    def nearest(self, xy: np.ndarray) -> np.ndarray:
        nn = self.qt.nearest_neighbor
        return np.fromiter((nn(p)[0] for p in map(tuple, xy.tolist())),
                           dtype=np.int64, count=len(xy))


class LinkSnapper:
    """Nearest link midpoint, per tier, with an exact point index per tier:
    the compiled ``mobslim_router.PointIndex`` (a k-d tree, queried in
    parallel) if installed, else a ``fastquadtree``. Both find a nearest
    midpoint exactly; between equally near links the k-d tree takes the
    lowest link index, the quadtree any one of them.

    ``backend``: ``"rust"``, ``"quadtree"`` or ``None`` (the fastest
    installed). ``threads`` (Rust only): default ``$MOBSLIM_ROUTER_THREADS``,
    else the number of CPUs. Prefer :func:`link_snapper`, which caches one
    per network and params.
    """

    def __init__(self, network: Network, params: ActivityLinkParams = ActivityLinkParams(),
                 backend: Optional[str] = None, threads: Optional[int] = None):
        if backend is None:
            backend = "rust" if rust_snapper_available() else "quadtree"
        if backend not in SNAP_BACKENDS:
            raise ValueError(f"backend must be one of {SNAP_BACKENDS}, not {backend!r}.")
        if backend == "rust" and not rust_snapper_available():
            raise ImportError("The rust backend needs the mobslim_router extension "
                              "(uv sync --group rust).")
        if threads is None:
            from mobslim.routing.compiled import default_threads
            threads = default_threads()
        self.backend, self.threads = backend, max(1, int(threads))
        mid = link_midpoints(network)
        eligible = params.eligible(network)
        masks = []
        for tier in params.tiers:
            mask = eligible.copy()
            for f in tier.exclude:
                mask &= ~f.mask(network)
            masks.append((mask, tier.max_distance))
        masks.append((eligible, math.inf))
        bounds = (*(mid.min(axis=0) - 1), *(mid.max(axis=0) + 1)) if len(mid) else (0, 0, 1, 1)
        # (link indices, their midpoints, point index over them, max distance)
        self.tiers = []
        for mask, max_distance in masks:
            links = np.flatnonzero(mask)
            if not len(links):
                continue
            pts = np.ascontiguousarray(mid[links], dtype=np.float64)
            index = _rust.PointIndex(pts) if backend == "rust" else _QuadTreeIndex(pts, bounds)
            self.tiers.append((links, pts, index, max_distance))

    def _nearest(self, index, xy: np.ndarray) -> np.ndarray:
        if self.backend == "rust":
            return index.nearest(xy, threads=self.threads)
        return index.nearest(xy)

    def snap(self, xy) -> np.ndarray:
        """Link index per point of ``xy`` (``(n, 2)``)."""
        if not self.tiers:
            raise ValueError("No link to assign activities to (the network has no links).")
        xy = np.ascontiguousarray(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        if not np.isfinite(xy).all():
            raise ValueError("Cannot assign a link to an activity without finite coordinates.")
        out = np.empty(len(xy), dtype=np.int64)
        pending = np.arange(len(xy))  # points no tier has taken yet
        for links, pts, index, max_distance in self.tiers:
            if not len(pending):
                break
            q = xy[pending]
            k = self._nearest(index, q)
            if math.isinf(max_distance):
                out[pending] = links[k]
                break
            ok = np.hypot(pts[k, 0] - q[:, 0], pts[k, 1] - q[:, 1]) <= max_distance
            out[pending[ok]] = links[k[ok]]
            pending = pending[~ok]
        return out


def link_snapper(network: Network, params: ActivityLinkParams = ActivityLinkParams()
                 ) -> LinkSnapper:
    """The :class:`LinkSnapper` for ``network`` and ``params``, cached on the network."""
    key = ("snapper", params)
    if key not in network._cache:
        network._cache[key] = LinkSnapper(network, params)
    return network._cache[key]


def assign_activity_links(scenario: "Scenario",
                          params: ActivityLinkParams = ActivityLinkParams()) -> "Scenario":
    """``scenario`` with every held plan's activities that have coordinates
    but no link assigned one (with ``params.snap_node_activities``, those
    placed at a node too). An assigned activity's node becomes its link's
    to-node; its coordinates are kept.

    Network routes of legs to or from an assigned activity are dropped and
    recomputed by :func:`~mobslim.replanning.prepare.route_scenario`. Scores
    and selection are kept. Returns ``scenario`` itself if nothing changes.
    """
    pop, net = scenario.population, scenario.network
    live = pop.live_plan_rows()
    batch = pop.plans.take(live)
    need = batch.act_link < 0
    if not params.snap_node_activities:
        need &= ~np.isnan(batch.act_xy[:, 0])
    if not need.any():
        return scenario
    xy = batch.resolved_act_xy(net).copy()
    link = link_snapper(net, params).snap(xy[need])
    act_link = batch.act_link.copy()
    act_link[need] = link
    act_node = batch.act_node.copy()
    act_node[need] = net.to_node[link]
    batch = dataclasses.replace(batch, act_link=act_link, act_node=act_node, act_xy=xy)
    batch = _unroute(batch, need, _network_modes(scenario))
    return scenario.with_population(pop.with_live_batch(live, batch))


def _network_modes(scenario: "Scenario") -> frozenset:
    modes = scenario.network.modes
    if scenario.transit is not None:
        modes = modes - set(scenario.transit.params.passenger_modes)
    return modes


def _unroute(batch, changed: np.ndarray, network_modes):
    """``batch`` with the network routes of legs next to a ``changed``
    activity removed (the legs marked un-routed)."""
    if not batch.n_legs:
        return batch
    origin = batch.leg_origin_act()
    drop = (changed[origin] | changed[origin + 1]) & batch.network_legs(network_modes)
    drop &= batch.leg_routed & ~batch.leg_pt_route
    if not drop.any():
        return batch
    keep = ~drop[batch.leg_of_link()]
    counts = np.where(drop, 0, np.diff(batch.route_offsets))
    offsets = np.zeros(batch.n_legs + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    nan = np.where(drop, np.nan, 1.0)
    return dataclasses.replace(
        batch, route_offsets=offsets, route_links=batch.route_links[keep],
        leg_routed=batch.leg_routed & ~drop,
        leg_route_time=batch.leg_route_time * nan,
        leg_route_distance=batch.leg_route_distance * nan)
