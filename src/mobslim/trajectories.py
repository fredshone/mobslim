"""Trajectories: what happened in a mobsim run, as arrays.

:class:`Trajectories` is the mobsim's primary output. It records times
aligned with the executed :class:`~mobslim.planbatch.PlanBatch`: one value
per activity, per leg and per route link. ``NaN`` means "did not happen"
(e.g. an activity never reached because the person got stuck).

===============  ========  ================================================
act_start        per act   activity start (``NaN`` for each plan's first)
act_end          per act   activity end (``NaN`` for the last, or if stuck)
leg_departure    per leg   departure time
leg_arrival      per leg   arrival time (``NaN`` if stuck on the way)
leg_distance     per leg   entered link lengths, or the teleport distance
link_enter       per link  time the route link was entered
stuck            per plan  the person did not finish their plan
ride_wait_start  per ride  pt: reached the board stop (``NaN`` = never)
ride_board       per ride  pt: boarded
ride_alight      per ride  pt: alighted
ride_trip        per ride  pt: departure boarded (-1 = none)
===============  ========  ================================================

With transit, ``transit`` (:class:`TransitTrajectories`) holds the
vehicles' times at their stops (and, in network mode, their link entries).
Boardings, loads and denied boardings are derived from these arrays.

Everything the iteration needs is derived from this table with numpy: link
travel times (:meth:`Trajectories.travel_time_table`), scores
(:meth:`~mobslim.scoring.CharyparNagelScorer.score_trajectories`) and
metrics. :class:`~mobslim.experience.ExperiencedPlan` objects are built only
on request (:meth:`Trajectories.experienced_plans`).

The table holds the same information as the event stream at about 8 bytes
per link traversal. A link is left when the next one is entered, or at
arrival for a leg's last link, so leave times are not stored.

Mobsims that only produce events can build the same table with a
:class:`TrajectoryCollector`.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mobslim.events import (
    ActivityEnd,
    ActivityStart,
    AgentWaitingForPt,
    EventHandler,
    LinkEnter,
    PersonArrival,
    PersonDeparture,
    PersonEntersVehicle,
    PersonLeavesVehicle,
    PersonStuck,
    Teleportation,
    TransitVehicleLinkEnter,
    VehicleArrivesAtFacility,
    VehicleDepartsAtFacility,
)
from mobslim.experience import (
    ExperiencedActivity,
    ExperiencedLeg,
    ExperiencedPlan,
    ExperiencedRide,
)
from mobslim.network import Network
from mobslim.planbatch import PlanBatch
from mobslim.routing.traveltime import (
    DEFAULT_BIN_SIZE,
    DEFAULT_HORIZON,
    TravelTimeTable,
    travel_time_table,
)


@dataclass(frozen=True, eq=False)
class TransitTrajectories:
    """Transit vehicles' trajectories, one entry per (trip, route stop) and
    per (trip, route link). Trip ``d``'s stops are
    ``trip_stop_offsets[d]:trip_stop_offsets[d + 1]`` (its route's stops in
    order), its links ``trip_link_offsets[d]:...`` (``link_enter`` is
    ``NaN`` in schedule mode, where vehicles are off the network; the first
    link's entry is the departure time)."""

    transit: object  # mobslim.transit.Transit
    stop_arrival: np.ndarray
    stop_departure: np.ndarray
    link_enter: np.ndarray

    def __post_init__(self):
        s = self.transit.schedule
        r = s.dep_route
        n_stops = np.diff(s.route_stop_offsets)[r] if s.n_departures else np.zeros(0, np.int64)
        n_links = np.diff(s.route_link_offsets)[r] if s.n_departures else np.zeros(0, np.int64)
        object.__setattr__(self, "trip_stop_offsets", np.concatenate([[0], np.cumsum(n_stops)]).astype(np.int64))
        object.__setattr__(self, "trip_link_offsets", np.concatenate([[0], np.cumsum(n_links)]).astype(np.int64))
        for name in ("stop_arrival", "stop_departure", "link_enter"):
            a = np.asarray(getattr(self, name), dtype=float)
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    @property
    def schedule(self):
        return self.transit.schedule

    def trip_stop_route_stop(self) -> np.ndarray:
        """Global route stop of each (trip, route stop) entry."""
        s = self.schedule
        trip = np.repeat(np.arange(s.n_departures), np.diff(self.trip_stop_offsets))
        within = np.arange(len(trip)) - self.trip_stop_offsets[trip]
        return s.route_stop_offsets[s.dep_route[trip]] + within

    def trip_stop_index(self, trip: np.ndarray, route_stop: np.ndarray) -> np.ndarray:
        """Entry of ``(trip, route_stop)`` pairs in the per-trip-stop arrays."""
        s = self.schedule
        return self.trip_stop_offsets[trip] + (route_stop - s.route_stop_offsets[s.dep_route[trip]])

    def link_traversals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(links, enter, leave)`` of transit vehicles' completed link
        traversals, with dwell at stops on the link taken off the leave time.
        A trip's first link (not traversed) and last link (no leave time
        recorded) are left out."""
        s = self.schedule
        lo = self.trip_link_offsets
        n = len(self.link_enter)
        trip = np.repeat(np.arange(s.n_departures), np.diff(lo))
        within = np.arange(n) - lo[trip]
        links = s.route_links[s.route_link_offsets[s.dep_route[trip]] + within]
        leave = np.full(n, np.nan)
        leave[:-1] = self.link_enter[1:]
        last = lo[1:][np.diff(lo) > 0] - 1
        leave[last] = np.nan
        rs = self.trip_stop_route_stop()
        stop_trip = np.repeat(np.arange(s.n_departures), np.diff(self.trip_stop_offsets))
        dwell = np.nan_to_num(self.stop_departure - self.stop_arrival)
        dwell = np.bincount(lo[stop_trip] + s.rs_link_pos[rs], weights=dwell, minlength=n)
        ok = (within > 0) & ~np.isnan(self.link_enter) & ~np.isnan(leave)
        return links[ok], self.link_enter[ok], (leave - dwell)[ok]

    def departure_delay(self) -> np.ndarray:
        """Actual minus scheduled departure per (trip, route stop) (``NaN`` if not departed)."""
        s = self.schedule
        trip = np.repeat(np.arange(s.n_departures), np.diff(self.trip_stop_offsets))
        return self.stop_departure - (s.dep_time[trip] + s.rs_departure[self.trip_stop_route_stop()])


@dataclass(frozen=True, eq=False)
class Trajectories:
    plans: PlanBatch
    act_start: np.ndarray  # float64 (n_acts,)
    act_end: np.ndarray  # float64 (n_acts,)
    leg_departure: np.ndarray  # float64 (n_legs,)
    leg_arrival: np.ndarray  # float64 (n_legs,)
    leg_distance: np.ndarray  # float64 (n_legs,)
    link_enter: np.ndarray  # float64 (n_route_links,)
    stuck: np.ndarray  # bool (n_plans,)
    ride_wait_start: Optional[np.ndarray] = None  # float64 (n_rides,)
    ride_board: Optional[np.ndarray] = None  # float64 (n_rides,)
    ride_alight: Optional[np.ndarray] = None  # float64 (n_rides,)
    ride_trip: Optional[np.ndarray] = None  # int32 (n_rides,), -1 = none
    transit: Optional[TransitTrajectories] = None
    transit_modes: frozenset = frozenset()
    _experienced: Optional[list] = field(default=None, repr=False)

    def __post_init__(self):
        p = self.plans
        n_rides = p.n_rides
        for name in ("ride_wait_start", "ride_board", "ride_alight"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(n_rides, np.nan))
        if self.ride_trip is None:
            object.__setattr__(self, "ride_trip", np.full(n_rides, -1, dtype=np.int32))
        sizes = {"act_start": p.n_acts, "act_end": p.n_acts, "leg_departure": p.n_legs,
                 "leg_arrival": p.n_legs, "leg_distance": p.n_legs,
                 "link_enter": len(p.route_links), "stuck": p.n_plans,
                 "ride_wait_start": n_rides, "ride_board": n_rides, "ride_alight": n_rides,
                 "ride_trip": n_rides}
        for name, n in sizes.items():
            a = getattr(self, name)
            if a.shape != (n,):
                raise ValueError(f"Trajectories.{name} must have shape ({n},), got {a.shape}.")
            a.flags.writeable = False

    @classmethod
    def build(cls, plans: PlanBatch, network: Network, *, act_start, act_end, leg_departure,
              leg_arrival, link_enter, stuck, teleport_distance,
              network_modes=None, transit=None, ride_wait_start=None, ride_board=None,
              ride_alight=None, ride_trip=None, stop_arrival=None, stop_departure=None,
              transit_link_enter=None, act_xy=None) -> "Trajectories":
        """Assemble from recorded values (lists or arrays).

        ``teleport_distance`` is per leg; it is used for teleported legs, and
        network legs get the summed length of the links they entered. With
        ``transit``, pt legs get the distance of the walks they finished and
        the rides they completed (``rs_distance`` of alight minus board).
        """
        link_enter = np.asarray(link_enter, dtype=float)
        distance = np.asarray(teleport_distance, dtype=float).copy()
        is_net = plans.network_legs(network.modes if network_modes is None else network_modes)
        if len(link_enter):
            entered = ~np.isnan(link_enter)
            lengths = np.where(entered, network.length[plans.route_links], 0.0)
            by_leg = np.bincount(plans.leg_of_link(), weights=lengths, minlength=plans.n_legs)
            distance[is_net] = by_leg[is_net]
        else:
            distance[is_net] = 0.0
        leg_arrival = np.asarray(leg_arrival, dtype=float)
        transit_tr, transit_modes, rides = None, frozenset(), {}
        if transit is not None:
            from mobslim.transit.walks import pt_walks

            transit_modes = frozenset(transit.params.passenger_modes)
            rides = dict(
                ride_wait_start=np.asarray(ride_wait_start, dtype=float),
                ride_board=np.asarray(ride_board, dtype=float),
                ride_alight=np.asarray(ride_alight, dtype=float),
                ride_trip=np.asarray(ride_trip, dtype=np.int32),
            )
            transit_tr = TransitTrajectories(transit, stop_arrival, stop_departure, transit_link_enter)
            if act_xy is None:
                act_xy = plans.resolved_act_xy(network)
            walks = pt_walks(plans, act_xy, transit)
            is_pt = plans.mode_legs(transit_modes)
            leg_of_ride = plans.leg_of_ride()
            per_ride = np.where(~np.isnan(rides["ride_wait_start"]), walks.before_distance, 0.0)
            per_ride += _ride_distance(transit, plans, rides["ride_alight"])
            pt = np.bincount(leg_of_ride, weights=per_ride, minlength=plans.n_legs).astype(float)
            pt += np.where(~np.isnan(leg_arrival), walks.after_distance, 0.0)
            distance[is_pt] = pt[is_pt]
        return cls(
            plans,
            np.asarray(act_start, dtype=float),
            np.asarray(act_end, dtype=float),
            np.asarray(leg_departure, dtype=float),
            leg_arrival,
            distance,
            link_enter,
            np.asarray(stuck, dtype=bool),
            transit=transit_tr,
            transit_modes=transit_modes,
            **rides,
        )

    # -- transit views -----------------------------------------------------
    def ride_distance(self) -> np.ndarray:
        """In-vehicle distance per ride (0 if not completed)."""
        if self.transit is None:
            return np.zeros(self.plans.n_rides)
        return _ride_distance(self.transit.transit, self.plans, self.ride_alight)

    def boardings(self) -> np.ndarray:
        """Boardings per (trip, route stop) entry of ``transit``."""
        return self._count_at(self.ride_board, self.plans.ride_board_stop)

    def alightings(self) -> np.ndarray:
        return self._count_at(self.ride_alight, self.plans.ride_alight_stop)

    def _count_at(self, times, route_stops) -> np.ndarray:
        tt = self.transit
        n = len(tt.stop_arrival)
        done = ~np.isnan(times) & (self.ride_trip >= 0)
        idx = tt.trip_stop_index(self.ride_trip[done].astype(np.int64),
                                 route_stops[done].astype(np.int64))
        return np.bincount(idx, minlength=n)

    def pt_load(self) -> np.ndarray:
        """Passengers on board leaving each (trip, route stop) entry."""
        tt = self.transit
        net = np.cumsum(self.boardings() - self.alightings())
        start = tt.trip_stop_offsets[:-1]
        counts = np.diff(tt.trip_stop_offsets)
        before = np.concatenate([[0], net])[start]
        return net - np.repeat(before, counts)

    def denied_boardings(self) -> np.ndarray:
        """Per ride: departures of its route from its board stop that left
        while the passenger waited (``wait_start <= departure < board``)."""
        out = np.zeros(self.plans.n_rides, dtype=np.int64)
        tt = self.transit
        if tt is None or not self.plans.n_rides:
            return out
        rs_of_entry = tt.trip_stop_route_stop()
        waited = np.flatnonzero(~np.isnan(self.ride_wait_start))
        board_rs = self.plans.ride_board_stop[waited]
        for k in np.unique(board_rs).tolist():
            times = np.sort(tt.stop_departure[rs_of_entry == k])
            times = times[~np.isnan(times)]
            rides = waited[board_rs == k]
            board = np.where(np.isnan(self.ride_board[rides]), np.inf, self.ride_board[rides])
            out[rides] = (np.searchsorted(times, board, side="left")
                          - np.searchsorted(times, self.ride_wait_start[rides], side="left"))
        return out

    # -- derived views -----------------------------------------------------
    def link_leave(self) -> np.ndarray:
        """Leave time per route link: the next link's entry, or the leg's
        arrival for its last link. ``NaN`` if the link was not left."""
        enter = self.link_enter
        leave = np.full_like(enter, np.nan)
        if len(enter):
            leave[:-1] = enter[1:]
            ends = self.plans.route_offsets[1:]
            last = ends > self.plans.route_offsets[:-1]
            leave[ends[last] - 1] = self.leg_arrival[last]
            leave[np.isnan(enter)] = np.nan
        return leave

    def travel_time_table(
        self, network: Network, bin_size: float = DEFAULT_BIN_SIZE, horizon: float = DEFAULT_HORIZON
    ) -> TravelTimeTable:
        """Mean link travel times per entry-time bin (see :func:`travel_time_table`)."""
        entered = ~np.isnan(self.link_enter)
        links = self.plans.route_links[entered]
        enter = self.link_enter[entered]
        leave = self.link_leave()[entered]
        tt = self.transit
        if tt is not None and tt.transit.params.learn_from_transit:
            v_links, v_enter, v_leave = tt.link_traversals()
            links = np.concatenate([links, v_links])
            enter = np.concatenate([enter, v_enter])
            leave = np.concatenate([leave, v_leave])
        return travel_time_table(network, links, enter, leave, bin_size, horizon)

    def experienced_plans(self) -> list[ExperiencedPlan]:
        """One :class:`ExperiencedPlan` per person (built once, then cached)."""
        if self._experienced is None:
            object.__setattr__(self, "_experienced", self._build_experienced())
        return self._experienced

    def _build_experienced(self) -> list[ExperiencedPlan]:
        b = self.plans
        types = [b.act_types[c] for c in b.act_type.tolist()]
        modes = [b.modes[c] for c in b.leg_mode.tolist()]
        nodes = b.act_node.tolist()
        start, end = self.act_start.tolist(), self.act_end.tolist()
        dep, arr, dist = (self.leg_departure.tolist(), self.leg_arrival.tolist(),
                          self.leg_distance.tolist())
        enter, links, roff = self.link_enter.tolist(), b.route_links.tolist(), b.route_offsets.tolist()
        act_off, stuck = b.act_offsets.tolist(), self.stuck.tolist()
        transit_legs = b.mode_legs(self.transit_modes).tolist()
        ro = b.ride_offsets.tolist()
        rb, ra = b.ride_board_stop.tolist(), b.ride_alight_stop.tolist()
        rw, rbt, rat = (_none_list(self.ride_wait_start), _none_list(self.ride_board),
                        _none_list(self.ride_alight))
        rtrip, rdist = self.ride_trip.tolist(), self.ride_distance().tolist()
        out = []
        for p in range(b.n_plans):
            a0, a1 = act_off[p], act_off[p + 1]
            elements = []
            for a in range(a0, a1):
                s = start[a]
                if a > a0 and s != s:
                    break  # never reached
                e = end[a]
                elements.append(ExperiencedActivity(types[a], nodes[a], None if s != s else s,
                                                    None if e != e else e))
                if a == a1 - 1:
                    break
                leg = a - p
                d = dep[leg]
                if d != d:
                    break  # never departed
                r0, r1 = roff[leg], roff[leg + 1]
                r = r0
                while r < r1 and enter[r] == enter[r]:  # entered links are a prefix
                    r += 1
                t = arr[leg]
                rides = None
                if transit_legs[leg]:
                    rides = tuple(
                        ExperiencedRide(rb[k], ra[k], rtrip[k], rw[k], rbt[k], rat[k], rdist[k])
                        for k in range(ro[leg], ro[leg + 1]) if rw[k] is not None)
                elements.append(ExperiencedLeg(modes[leg], d, None if t != t else t, dist[leg],
                                               tuple(links[r0:r]), rides))
            out.append(ExperiencedPlan(tuple(elements), stuck[p]))
        return out


_VEHICLE_EVENTS = frozenset((VehicleArrivesAtFacility, VehicleDepartsAtFacility,
                             TransitVehicleLinkEnter))


class TrajectoryCollector(EventHandler):
    """Builds :class:`Trajectories` from events, for mobsims that only emit events.

    It assumes the events follow the executed plans: persons end and start
    their planned activities in order and enter the links of their planned
    routes in order. Without ``transit`` it records the riders' rides but
    ignores the vehicles' events.
    """

    handles = (ActivityStart, ActivityEnd, PersonDeparture, PersonArrival,
               LinkEnter, Teleportation, PersonStuck, AgentWaitingForPt,
               PersonEntersVehicle, PersonLeavesVehicle, VehicleArrivesAtFacility,
               VehicleDepartsAtFacility, TransitVehicleLinkEnter)

    def __init__(self, network: Network, plans: PlanBatch, transit=None):
        self.network = network
        self.plans = plans
        self.transit = transit
        self.reset(0)

    def reset(self, iteration: int) -> None:
        b, nan = self.plans, math.nan
        self._act = b.act_offsets[:-1].tolist()  # current activity per person
        self._leg = [0] * b.n_plans
        self._pos = [0] * b.n_plans  # next route link per person
        self._route_offsets = b.route_offsets.tolist()
        self._act_start = [nan] * b.n_acts
        self._act_end = [nan] * b.n_acts
        self._dep = [nan] * b.n_legs
        self._arr = [nan] * b.n_legs
        self._dist = [0.0] * b.n_legs
        self._enter = [nan] * len(b.route_links)
        self._stuck = [False] * b.n_plans
        self._ride = [0] * b.n_plans  # current ride per person
        self._ride_offsets = b.ride_offsets.tolist()
        self._rw = [nan] * b.n_rides
        self._rb = [nan] * b.n_rides
        self._ra = [nan] * b.n_rides
        self._rt = [-1] * b.n_rides
        self._result = None
        if self.transit is not None:
            empty = TransitTrajectories(self.transit, [], [], [])  # for the offsets
            self._tso = empty.trip_stop_offsets.tolist()
            self._tlo = empty.trip_link_offsets.tolist()
            s = self.transit.schedule
            self._trip_route = s.dep_route.tolist()
            self._rso = s.route_stop_offsets.tolist()
            self._stop_arr = [nan] * self._tso[-1]
            self._stop_dep = [nan] * self._tso[-1]
            self._veh_enter = [nan] * self._tlo[-1]
            self._trip_rs = [0] * s.n_departures  # next stop entry per trip
            self._trip_link = [0] * s.n_departures

    def _vehicle_event(self, t, e) -> None:
        d = e.trip
        if t is VehicleArrivesAtFacility:
            self._stop_arr[self._tso[d] + self._trip_rs[d]] = e.time
        elif t is VehicleDepartsAtFacility:
            self._stop_dep[self._tso[d] + self._trip_rs[d]] = e.time
            self._trip_rs[d] += 1
        else:  # TransitVehicleLinkEnter
            self._veh_enter[self._tlo[d] + self._trip_link[d]] = e.time
            self._trip_link[d] += 1

    def handle(self, e) -> None:
        t = type(e)
        if t in _VEHICLE_EVENTS:
            if self.transit is not None:  # without the schedule: riders only
                self._vehicle_event(t, e)
            return
        p = e.person
        if t is AgentWaitingForPt:
            self._rw[self._ride[p]] = e.time
        elif t is PersonEntersVehicle:
            self._rb[self._ride[p]] = e.time
            self._rt[self._ride[p]] = e.trip
        elif t is PersonLeavesVehicle:
            self._ra[self._ride[p]] = e.time
            self._ride[p] += 1
        elif t is LinkEnter:
            pos = self._pos[p]
            self._enter[pos] = e.time
            self._pos[p] = pos + 1
        elif t is ActivityEnd:
            self._act_end[self._act[p]] = e.time
        elif t is PersonDeparture:
            leg = self._act[p] - p
            self._leg[p] = leg
            self._dep[leg] = e.time
            self._pos[p] = self._route_offsets[leg]
            self._ride[p] = self._ride_offsets[leg]
        elif t is PersonArrival:
            self._arr[self._leg[p]] = e.time
        elif t is ActivityStart:
            a = self._act[p] + 1
            self._act[p] = a
            self._act_start[a] = e.time
        elif t is Teleportation:
            self._dist[self._leg[p]] = e.distance
        elif t is PersonStuck:
            self._stuck[p] = True

    def finish(self) -> None:
        self._result = Trajectories.build(
            self.plans, self.network, act_start=self._act_start, act_end=self._act_end,
            leg_departure=self._dep, leg_arrival=self._arr, link_enter=self._enter,
            stuck=self._stuck, teleport_distance=self._dist,
            **self._transit_kwargs(),
        )

    def _transit_kwargs(self) -> dict:
        if self.transit is None:
            return {}
        modes = self.transit.params.passenger_modes
        return dict(network_modes=self.network.modes - frozenset(modes), transit=self.transit,
                    ride_wait_start=self._rw, ride_board=self._rb, ride_alight=self._ra,
                    ride_trip=self._rt, stop_arrival=self._stop_arr,
                    stop_departure=self._stop_dep, transit_link_enter=self._veh_enter)

    def trajectories(self) -> Trajectories:
        if self._result is None:
            raise RuntimeError("TrajectoryCollector.finish() has not been called.")
        return self._result


def _ride_distance(transit, plans: PlanBatch, ride_alight: np.ndarray) -> np.ndarray:
    d = transit.schedule.rs_distance
    full = d[plans.ride_alight_stop] - d[plans.ride_board_stop] if plans.n_rides else np.zeros(0)
    return np.where(np.isnan(ride_alight), 0.0, full)


def _none_list(a: np.ndarray) -> list:
    return [None if v != v else v for v in a.tolist()]
