"""Plans encoded as flat arrays: the format a mobsim executes.

A :class:`PlanBatch` is a columnar ("struct of arrays") encoding of a list of
plans, usually every person's selected plan in person-index order. Plan
``p``'s activities, legs and route links are contiguous slices of flat
arrays, located by offset arrays (CSR layout)::

    activities of plan p   act_offsets[p] : act_offsets[p + 1]
    legs of plan p         leg_offsets[p] : leg_offsets[p + 1]   (one fewer)
    links of leg l         route_offsets[l] : route_offsets[l + 1]
    rides of leg l         ride_offsets[l] : ride_offsets[l + 1]  (pt legs)

A pt leg's rides (its :class:`~mobslim.population.TransitPassengerRoute`) are pairs of
global route stops, ``ride_board_stop[r]`` and ``ride_alight_stop[r]``.
Network legs have link slices and no rides; pt legs have rides and no links.
``act_xy`` holds each activity's coordinates (``NaN`` where it has none and
no network was given to fill in its node's), ``act_link`` its link (``-1``
where it has none).

The leg between global activities ``a`` and ``a + 1`` of plan ``p`` has the
global index ``a - p``. Activity types and modes are small integer codes into
``act_types`` and ``modes``. Missing times are ``NaN``.

This is the boundary format for a compiled mobsim: plain numeric arrays, no
Python objects. The Python :class:`~mobslim.mobsim.qsim.QSim` runs on it too,
and the mobsim's output (:class:`~mobslim.trajectories.Trajectories`) is
aligned with it: one value per activity, per leg, or per route link.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from mobslim.population import Plan

_I32 = np.dtype("=i4")


@dataclass(frozen=True, eq=False)
class PlanBatch:
    act_offsets: np.ndarray  # int64 (n_plans + 1,)
    act_type: np.ndarray  # int32 (n_acts,), code into act_types
    act_node: np.ndarray  # int32 (n_acts,)
    act_end_time: np.ndarray  # float64 (n_acts,), NaN = none
    act_duration: np.ndarray  # float64 (n_acts,), NaN = none
    leg_mode: np.ndarray  # int32 (n_legs,), code into modes
    leg_travel_time: np.ndarray  # float64 (n_legs,), fixed teleport time, NaN = none
    leg_routed: np.ndarray  # bool (n_legs,): the leg has a route (possibly empty)
    route_offsets: np.ndarray  # int64 (n_legs + 1,)
    route_links: np.ndarray  # int32 (n_route_links,)
    act_types: tuple[str, ...]
    modes: tuple[str, ...]
    act_xy: np.ndarray = None  # float64 (n_acts, 2), NaN = none
    leg_pt_route: np.ndarray = None  # bool (n_legs,): the route is a TransitPassengerRoute
    ride_offsets: np.ndarray = None  # int64 (n_legs + 1,)
    ride_board_stop: np.ndarray = None  # int32 (n_rides,), global route stop
    ride_alight_stop: np.ndarray = None  # int32 (n_rides,), global route stop
    act_link: np.ndarray = None  # int32 (n_acts,), -1 = none
    leg_route_time: np.ndarray = None  # float64 (n_legs,), the route's expected time, NaN = none
    leg_route_distance: np.ndarray = None  # float64 (n_legs,), the route's distance, NaN = none

    _ARRAYS = ("act_offsets", "act_type", "act_node", "act_end_time", "act_duration",
               "leg_mode", "leg_travel_time", "leg_routed", "route_offsets", "route_links",
               "act_xy", "leg_pt_route", "ride_offsets", "ride_board_stop", "ride_alight_stop", "act_link",
               "leg_route_time", "leg_route_distance")

    def __post_init__(self):
        n_acts, n_legs = len(self.act_type), len(self.leg_mode)
        defaults = {
            "act_xy": lambda: np.full((n_acts, 2), np.nan),
            "leg_pt_route": lambda: np.zeros(n_legs, dtype=bool),
            "ride_offsets": lambda: np.zeros(n_legs + 1, dtype=np.int64),
            "ride_board_stop": lambda: np.zeros(0, dtype=np.int32),
            "ride_alight_stop": lambda: np.zeros(0, dtype=np.int32),
            "act_link": lambda: np.full(n_acts, -1, dtype=np.int32),
            "leg_route_time": lambda: np.full(n_legs, np.nan),
            "leg_route_distance": lambda: np.full(n_legs, np.nan),
        }
        for name, make in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, make())
        for name in self._ARRAYS:
            getattr(self, name).flags.writeable = False

    # -- sizes and derived indices ------------------------------------------
    @property
    def n_plans(self) -> int:
        return len(self.act_offsets) - 1

    @property
    def n_acts(self) -> int:
        return len(self.act_type)

    @property
    def n_legs(self) -> int:
        return len(self.leg_mode)

    @property
    def leg_offsets(self) -> np.ndarray:
        """``(n_plans + 1,)``: legs of plan ``p`` are ``leg_offsets[p]:leg_offsets[p+1]``."""
        return self.act_offsets - np.arange(self.n_plans + 1)

    def _cached(self, name: str, make) -> np.ndarray:
        """A derived index array, built once (the batch is immutable) and read-only."""
        value = self.__dict__.get(name)
        if value is None:
            value = make()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        return value

    def plan_of_act(self) -> np.ndarray:
        return self._cached("_plan_of_act", lambda: np.repeat(np.arange(self.n_plans),
                                                              np.diff(self.act_offsets)))

    def plan_of_leg(self) -> np.ndarray:
        return self._cached("_plan_of_leg", lambda: np.repeat(np.arange(self.n_plans),
                                                              np.diff(self.leg_offsets)))

    @property
    def n_rides(self) -> int:
        return len(self.ride_board_stop)

    def leg_of_link(self) -> np.ndarray:
        return self._cached("_leg_of_link", lambda: np.repeat(np.arange(self.n_legs),
                                                              np.diff(self.route_offsets)))

    def leg_of_ride(self) -> np.ndarray:
        return self._cached("_leg_of_ride", lambda: np.repeat(np.arange(self.n_legs),
                                                              np.diff(self.ride_offsets)))

    def resolved_act_xy(self, network) -> np.ndarray:
        """``act_xy`` with each missing coordinate taken from the activity's node."""
        xy = self.act_xy
        missing = np.isnan(xy[:, 0])
        if not missing.any():
            return xy
        out = xy.copy()
        out[missing] = network.node_xy[self.act_node[missing]]
        out.flags.writeable = False
        return out

    def mode_legs(self, modes) -> np.ndarray:
        """Bool per leg: is its mode in ``modes``?"""
        is_in = np.array([m in modes for m in self.modes], dtype=bool)
        return is_in[self.leg_mode] if self.n_legs else np.zeros(0, dtype=bool)

    def leg_origin_act(self) -> np.ndarray:
        """Global index of the activity each leg starts from (``a = l + p``)."""
        return np.arange(self.n_legs) + self.plan_of_leg()

    def network_legs(self, network_modes) -> np.ndarray:
        """Bool per leg: is its mode a network mode?"""
        is_net = np.array([m in network_modes for m in self.modes], dtype=bool)
        return is_net[self.leg_mode] if self.n_legs else np.zeros(0, dtype=bool)

    # -- building -----------------------------------------------------------
    @classmethod
    def from_plans(cls, plans: Sequence["Plan"], network=None) -> "PlanBatch":
        """Encode plans in order. Activity types and modes get codes in order
        of first appearance. With ``network``, activities without coordinates
        get their node's in ``act_xy`` (else ``NaN``)."""
        from mobslim.population import TransitPassengerRoute

        b = PlanBatchBuilder()
        nan = math.nan
        act, leg, end = b.act, b.leg, b.end_plan
        pt_route = TransitPassengerRoute
        for plan in plans:
            elements = plan.elements
            for i, e in enumerate(elements):
                if i & 1:
                    route = e.route
                    tt = nan if e.travel_time is None else e.travel_time
                    if route is None:
                        leg(e.mode, tt)
                    else:
                        leg(e.mode, tt, route.data, route.__class__ is pt_route,
                            route.travel_time, route.distance)
                else:
                    act(e.type, e.node, nan if e.end_time is None else e.end_time,
                        nan if e.duration is None else e.duration,
                        nan if e.x is None else e.x, nan if e.y is None else e.y,
                        -1 if e.link is None else e.link)
            end()
        return b.build(network)

    # -- checks -------------------------------------------------------------
    def invalid_legs(self, network, network_modes=None, transit=None) -> np.ndarray:
        """Indices of network- and transit-mode legs without a valid route.

        A valid network route is a :class:`~mobslim.population.Route` that is
        connected, starts at the leg's origin node, ends at its destination
        node, and only uses links that allow the leg's mode. If the
        destination activity is on a link that the route must use (see
        :func:`~mobslim.routing.base.arrival_link`), the route ends with it;
        an activity's link must end at the activity's node.
        ``network_modes`` defaults to every link mode.

        With ``transit`` (a :class:`~mobslim.transit.Transit`), legs of its
        passenger modes need a :class:`~mobslim.population.TransitPassengerRoute` whose
        rides board and alight on the same transit route, alight after they
        board, and name existing route stops. Vectorised over all legs.
        Teleported legs are not checked here.
        """
        if network_modes is None:
            network_modes = network.modes
        bad_pt = np.zeros(self.n_legs, dtype=bool)
        if transit is not None:
            is_pt = self.mode_legs(transit.params.passenger_modes)
            bad_pt = is_pt & ~self.leg_pt_route
            bad_pt |= is_pt & (np.diff(self.route_offsets) > 0)
            if self.n_rides:
                s = transit.schedule
                board = self.ride_board_stop.astype(np.int64)
                alight = self.ride_alight_stop.astype(np.int64)
                exists = (board < s.n_route_stops) & (alight < s.n_route_stops)
                b, a = np.where(exists, board, 0), np.where(exists, alight, 0)
                rs_route = s.rs_route
                broken = ~exists | (rs_route[b] != rs_route[a]) | (a <= b)
                leg = self.leg_of_ride()
                bad_pt |= is_pt & (np.bincount(leg[broken], minlength=self.n_legs) > 0)
        is_net = self.network_legs(network_modes)
        origin_act = self.leg_origin_act()
        origin = self.act_node[origin_act].astype(np.int64)
        dest = self.act_node[origin_act + 1].astype(np.int64)
        bad = is_net & (~self.leg_routed | self.leg_pt_route)
        bad |= bad_pt

        links = self.route_links.astype(np.int64)
        if len(links):
            leg = self.leg_of_link()
            starts = self.route_offsets[:-1]
            ends = self.route_offsets[1:]
            nonempty = ends > starts
            # node each link must start from: the previous link's to-node, or
            # the leg's origin for the first link of a leg
            expected = np.empty_like(links)
            expected[1:] = network.to_node[links[:-1]]
            expected[starts[nonempty]] = origin[nonempty]
            allowed = np.zeros(len(links), dtype=bool)
            for code, mode in enumerate(self.modes):
                if mode in network_modes:
                    on = self.leg_mode[leg] == code
                    allowed[on] = network.mode_mask(mode)[links[on]]
            broken = (network.from_node[links] != expected) | ~allowed
            bad |= is_net & (np.bincount(leg[broken], minlength=self.n_legs) > 0)
            last_to = np.full(self.n_legs, -1, dtype=np.int64)
            last_to[nonempty] = network.to_node[links[ends[nonempty] - 1]]
            bad |= is_net & nonempty & (last_to != dest)
            bad |= is_net & self.leg_routed & ~nonempty & (origin != dest)
            last = np.full(self.n_legs, -1, dtype=np.int64)
            last[nonempty] = links[ends[nonempty] - 1]
        else:
            bad |= is_net & self.leg_routed & (origin != dest)
            last = np.full(self.n_legs, -1, dtype=np.int64)
        # an activity's link must end at its node
        act_link = self.act_link.astype(np.int64)
        off = (act_link >= 0) & (network.to_node[np.maximum(act_link, 0)] != self.act_node)
        bad |= is_net & (off[origin_act] | off[origin_act + 1])
        # legs that must end by driving through the destination activity's link
        origin_link = self.act_link[origin_act].astype(np.int64)
        dest_link = self.act_link[origin_act + 1].astype(np.int64)
        must = is_net & (dest_link >= 0) & (dest_link != origin_link)
        must &= ~((origin_link < 0) & (origin == dest))
        if must.any():
            for code, mode in enumerate(self.modes):
                on = must & (self.leg_mode == code)
                if on.any():
                    must[on] = network.mode_mask(mode)[dest_link[on]]
            bad |= must & self.leg_routed & (last != dest_link)
        return np.flatnonzero(bad)


class PlanBatchBuilder:
    """Builds a :class:`PlanBatch` element by element, without plan objects.

    Call :meth:`act` and :meth:`leg` alternately (starting and ending with an
    activity) and :meth:`end_plan` after each plan. Missing values are
    ``NaN`` (times, coordinates) or ``-1`` (link). A routed leg passes its
    packed ``route`` (native int32 bytes: links, or ``(board, alight)``
    pairs if ``pt``) and the route's expected time and distance.
    """

    def __init__(self):
        self._type_code: dict[str, int] = {}
        self._mode_code: dict[str, int] = {}
        self.act_offsets = [0]
        self.a_type, self.a_node, self.a_end, self.a_dur = [], [], [], []
        self.a_x, self.a_y, self.a_link = [], [], []
        self.l_mode, self.l_tt, self.l_routed, self.l_pt = [], [], [], []
        self.l_rtt, self.l_rdist = [], []
        self.route_offsets, self._route_chunks, self._n_links = [0], [], 0
        self.ride_offsets, self._ride_chunks, self._n_rides = [0], [], 0

    def act(self, type: str, node: int, end_time: float = math.nan, duration: float = math.nan,
            x: float = math.nan, y: float = math.nan, link: int = -1) -> None:
        code = self._type_code.get(type)
        if code is None:
            code = self._type_code[type] = len(self._type_code)
        self.a_type.append(code)
        self.a_node.append(node)
        self.a_end.append(end_time)
        self.a_dur.append(duration)
        self.a_x.append(x)
        self.a_y.append(y)
        self.a_link.append(link)

    def leg(self, mode: str, travel_time: float = math.nan, route: bytes = None, pt: bool = False,
            route_time: float = math.nan, route_distance: float = math.nan) -> None:
        code = self._mode_code.get(mode)
        if code is None:
            code = self._mode_code[mode] = len(self._mode_code)
        self.l_mode.append(code)
        self.l_tt.append(travel_time)
        if route is None:
            self.l_routed.append(False)
            self.l_pt.append(False)
            self.l_rtt.append(math.nan)
            self.l_rdist.append(math.nan)
        else:
            self.l_routed.append(True)
            self.l_pt.append(pt)
            self.l_rtt.append(route_time)
            self.l_rdist.append(route_distance)
            if pt:
                self._ride_chunks.append(route)
                self._n_rides += len(route) >> 3
            else:
                self._route_chunks.append(route)
                self._n_links += len(route) >> 2
        self.route_offsets.append(self._n_links)
        self.ride_offsets.append(self._n_rides)

    def end_plan(self) -> None:
        self.act_offsets.append(len(self.a_type))

    @property
    def n_plans(self) -> int:
        return len(self.act_offsets) - 1

    def build(self, network=None) -> PlanBatch:
        rides = np.frombuffer(b"".join(self._ride_chunks), dtype=_I32).reshape(-1, 2)
        act_xy = np.column_stack([np.array(self.a_x, dtype=float), np.array(self.a_y, dtype=float)])
        act_node = np.array(self.a_node, dtype=np.int32)
        if network is not None and len(act_xy):
            missing = np.isnan(act_xy[:, 0])
            act_xy[missing] = network.node_xy[act_node[missing]]
        return PlanBatch(
            act_offsets=np.array(self.act_offsets, dtype=np.int64),
            act_type=np.array(self.a_type, dtype=np.int32),
            act_node=act_node,
            act_end_time=np.array(self.a_end, dtype=float),
            act_duration=np.array(self.a_dur, dtype=float),
            leg_mode=np.array(self.l_mode, dtype=np.int32),
            leg_travel_time=np.array(self.l_tt, dtype=float),
            leg_routed=np.array(self.l_routed, dtype=bool),
            route_offsets=np.array(self.route_offsets, dtype=np.int64),
            route_links=np.frombuffer(b"".join(self._route_chunks), dtype=_I32).copy(),
            act_types=tuple(self._type_code),
            modes=tuple(self._mode_code),
            act_xy=act_xy.reshape(-1, 2),
            leg_pt_route=np.array(self.l_pt, dtype=bool),
            ride_offsets=np.array(self.ride_offsets, dtype=np.int64),
            ride_board_stop=rides[:, 0].copy(),
            ride_alight_stop=rides[:, 1].copy(),
            act_link=np.array(self.a_link, dtype=np.int32),
            leg_route_time=np.array(self.l_rtt, dtype=float),
            leg_route_distance=np.array(self.l_rdist, dtype=float),
        )
