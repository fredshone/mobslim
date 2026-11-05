"""HermesSim: a whole-second queue simulation with a compiled kernel.

Modelled on MATSim's Hermes mobsim (``org.matsim.core.mobsim.hermes``):

* **Time is whole seconds**, and the agenda is a set of **per-second
  buckets** instead of a priority queue.
* The kernel runs on **flat arrays** (the :class:`~mobslim.planbatch.PlanBatch`
  encoding plus per-link parameters) and returns arrays, so it is written
  once in Python (this module: the readable spec and the fallback) and once
  in Rust (the ``mobslim_hermes`` extension, the fast path). Both backends
  must give identical results; ``tests/test_hermes.py`` checks that they do.
* **Events are recorded compactly** while the kernel runs and turned into
  event objects afterwards, only for the types some handler wants, like
  Hermes's ``EventArray``.

The queue model is the QSim's (and MATSim's), and so is the handling of
blocked vehicles. Unlike Hermes, blocked vehicles are **not polled**: a full
link keeps a FIFO of the upstream links and departures waiting for it and
wakes one per freed slot, so run time does not grow with congestion.

How time is discretised:

* **Free-speed time** per link is ``round(length / freespeed)`` seconds
  (half up). This is unbiased, but shifts each link by up to 0.5 s.
* **Flow capacity** is kept exact over time. Each link keeps its next free
  flow slot as a float, and a vehicle leaves at the first whole second at or
  after it, so several vehicles can leave in one second if the capacity is
  above 1 veh/s. When the head left at its first possible second, the next slot
  counts from the unrounded time, so fractional headways do not drift (a
  1.8 s headway gives exits at 10, 12, 14, 16, 18, 21, ...).
* **Activity ends**, ``end_time`` or ``start + duration``, are rounded up to
  the next whole second. **Teleport times** are rounded half up.
* **Stuck time** counts per vehicle, as in the QSim, and is forced at the
  first whole second at or after ``blocked_since + stuck_time``.
  ``stuck_time=inf`` is supported: an unresolvable gridlock ends the run
  with the blocked persons reported as stuck.

With integer free-speed times, headways and activity times, HermesSim makes
the same moves in the same order as the QSim, so the trajectories are
identical. Same-second entries are processed in push order, like the QSim's
``(time, sequence)`` heap.

**Transit** (``scenario.transit``) follows the QSim's rules
(:mod:`mobslim.mobsim._transit`) on the same clock: trips start and serve
stops no earlier than the timetable's times rounded up; walks, door and
per-passenger times round half up. Transit vehicles are vehicles
``n_plans + d`` in the kernel and take their PCE of storage and flow. Stop
departures are "late" entries: they run after every other entry of their
second, so passengers arriving in that second still board.
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mobslim.events import (
    ActivityEnd,
    ActivityStart,
    AgentWaitingForPt,
    EventsManager,
    LinkEnter,
    LinkLeave,
    PersonArrival,
    PersonDeparture,
    PersonEntersVehicle,
    PersonLeavesVehicle,
    PersonStuck,
    Teleportation,
    TransitDriverStarts,
    TransitVehicleLinkEnter,
    TransitVehicleLinkLeave,
    VehicleArrivesAtFacility,
    VehicleDepartsAtFacility,
)
from mobslim.mobsim._common import EPS, link_parameters, validate
from mobslim.planbatch import PlanBatch
from mobslim.routing.teleport import teleport_many
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories
from mobslim.transit.walks import pt_walks

try:  # the compiled kernel (rust/mobslim-hermes); optional
    import mobslim_hermes as _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None

BACKENDS = ("auto", "rust", "python")

# recorded event kinds; the kernel's event mask has bit ``1 << kind`` set for
# each kind some handler wants
EV_ACT_START, EV_ACT_END, EV_DEPARTURE, EV_ARRIVAL = 0, 1, 2, 3
EV_LINK_ENTER, EV_LINK_LEAVE, EV_TELEPORT, EV_STUCK = 4, 5, 6, 7
# transit: "person" holds the trip for vehicle events; ref is a route stop,
# trip or link
EV_DRIVER_START, EV_VEH_ARRIVE, EV_VEH_DEPART, EV_WAITING = 8, 9, 10, 11
EV_ENTER_VEH, EV_LEAVE_VEH, EV_VEH_LINK_ENTER, EV_VEH_LINK_LEAVE = 12, 13, 14, 15
N_EVENT_KINDS = 16
EVENT_KIND = {
    ActivityStart: EV_ACT_START, ActivityEnd: EV_ACT_END,
    PersonDeparture: EV_DEPARTURE, PersonArrival: EV_ARRIVAL,
    LinkEnter: EV_LINK_ENTER, LinkLeave: EV_LINK_LEAVE,
    Teleportation: EV_TELEPORT, PersonStuck: EV_STUCK,
    TransitDriverStarts: EV_DRIVER_START, VehicleArrivesAtFacility: EV_VEH_ARRIVE,
    VehicleDepartsAtFacility: EV_VEH_DEPART, AgentWaitingForPt: EV_WAITING,
    PersonEntersVehicle: EV_ENTER_VEH, PersonLeavesVehicle: EV_LEAVE_VEH,
    TransitVehicleLinkEnter: EV_VEH_LINK_ENTER, TransitVehicleLinkLeave: EV_VEH_LINK_LEAVE,
}

# packed event code: kind << 58 | person << 29 | ref (activity, leg, link, ...)
_KIND_SHIFT, _PERSON_SHIFT = 58, 29
_MASK = (1 << 29) - 1
MAX_INDEX = 1 << 29  # persons, trips, activities, legs and links must be below this

# agenda entry kinds
_ACT_END, _TRY_LEAVE, _FORCE_LEAVE, _TRY_DEPART, _FORCE_DEPART, _TELEPORT_ARRIVE = range(6)
_TRIP_START, _STOP_ARRIVE, _STOP_DEPART, _DOOR_OPEN, _ALIGHT_DONE = range(6, 11)
_BOARD_DONE, _DOOR_CLOSE, _PAX_AT_STOP, _PAX_ARRIVE = range(11, 15)

# doors of a trip in network mode
_OPENING, _OPEN, _CLOSING, _CLOSED = 1, 2, 3, 4


def rust_available() -> bool:
    """Whether the compiled kernel (``mobslim_hermes``) is installed."""
    return _rust is not None


@dataclass(frozen=True)
class HermesSimConfig:
    flow_capacity_factor: float = 1.0
    storage_capacity_factor: float = 1.0
    stuck_time: float = 10.0
    effective_cell_size: float = 7.5  # meters per vehicle (PCU)
    end_time: Optional[float] = None  # None: run until every plan is done
    validate_routes: bool = True
    # "rust": the compiled kernel (ImportError if missing); "python": the
    # reference kernel in this module; "auto": rust if installed
    backend: str = "auto"

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {self.backend!r}.")
        if not self.stuck_time > 0:
            raise ValueError("stuck_time must be positive (inf disables it).")


class HermesSim:
    """Whole-second queue mobsim. Stateless between runs; safe to reuse."""

    def __init__(self, config: HermesSimConfig = HermesSimConfig()):
        self.config = config
        if config.backend == "rust" and _rust is None:
            raise ImportError(
                "HermesSim backend='rust' needs the mobslim_hermes extension "
                "(rust/mobslim-hermes; `uv sync` builds it when cargo is installed)."
            )

    @property
    def backend(self) -> str:
        """The kernel that runs: ``"rust"`` or ``"python"``."""
        if self.config.backend == "auto":
            return "rust" if _rust is not None else "python"
        return self.config.backend

    def run(self, scenario: Scenario, events: EventsManager) -> Trajectories:
        plans = scenario.population.selected_batch()
        return self.run_batch(scenario, plans, events)

    def run_batch(self, scenario: Scenario, plans: PlanBatch, events: EventsManager) -> Trajectories:
        """Run already-encoded plans (``plans`` row ``i`` = person index ``i``)."""
        if plans.n_plans != len(scenario.population):
            raise ValueError("PlanBatch must have one plan per person.")
        cfg = self.config
        if cfg.validate_routes:
            validate(scenario, plans)
        inputs, tele_distance = kernel_inputs(scenario, plans, cfg, _event_mask(events))
        kernel = _rust.run if self.backend == "rust" else run_kernel
        (act_start, act_end, dep, arr, enter, stuck, ev_time, ev_code, stuck_leg, stuck_link,
         last_time, ride_wait, ride_board, ride_alight, ride_trip, stop_arr,
         stop_dep) = kernel(**inputs)
        arr, enter = np.asarray(arr), np.asarray(enter)
        n_plan_links = len(plans.route_links)
        teleported = ~plans.network_legs(scenario.modes.network_modes)
        distance = np.where(teleported & ~np.isnan(arr), tele_distance, 0.0)
        transit = {}
        if scenario.transit is not None:
            n_trip_links = int(inputs["trip_link_offsets"][-1])
            transit = dict(
                transit=scenario.transit, ride_wait_start=ride_wait, ride_board=ride_board,
                ride_alight=ride_alight, ride_trip=ride_trip, stop_arrival=stop_arr,
                stop_departure=stop_dep,
                transit_link_enter=(enter[n_plan_links:] if inputs["network_vehicles"]
                                    else np.full(n_trip_links, np.nan)),
            )
        trajectories = Trajectories.build(
            plans, scenario.network, act_start=act_start, act_end=act_end,
            leg_departure=dep, leg_arrival=arr, link_enter=enter[:n_plan_links],
            stuck=np.asarray(stuck, dtype=bool), teleport_distance=distance,
            network_modes=scenario.modes.network_modes, **transit,
        )
        if inputs["event_mask"]:
            _dispatch(events, scenario, plans, tele_distance, ev_time, ev_code,
                      trajectories.stuck, stuck_leg, stuck_link, last_time)
        return trajectories


# ---------------------------------------------------------------- inputs


def _event_mask(events: EventsManager) -> int:
    """Bit ``1 << kind`` per event kind some handler wants."""
    mask = 0
    for event_type, kind in EVENT_KIND.items():
        if events.handlers_for(event_type):
            mask |= 1 << kind
    return mask


def round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(np.int64)


def ceil_steps(x: np.ndarray) -> np.ndarray:
    """The first whole second at or after each of ``x`` (see :func:`ceil_step`)."""
    return np.ceil(np.asarray(x, dtype=float) - EPS).astype(np.int64)


def kernel_inputs(scenario: Scenario, plans: PlanBatch, cfg: HermesSimConfig,
                  event_mask: int) -> tuple[dict, np.ndarray]:
    """The kernel's keyword arguments, plus each leg's teleport distance."""
    net = scenario.network
    free_time, headway, storage = link_parameters(
        net, cfg.flow_capacity_factor, cfg.storage_capacity_factor, cfg.effective_cell_size
    )
    free_steps = round_half_up(free_time)
    is_net = plans.network_legs(scenario.modes.network_modes)
    is_transit = (plans.mode_legs(scenario.modes.transit_modes) if scenario.transit is not None
                  else np.zeros(plans.n_legs, dtype=bool))
    tele_time = np.zeros(plans.n_legs)
    tele_distance = np.zeros(plans.n_legs)
    origin = plans.leg_origin_act()
    xy = plans.resolved_act_xy(net)
    for code, mode in enumerate(plans.modes):
        legs = np.flatnonzero(~is_net & ~is_transit & (plans.leg_mode == code))
        if not len(legs) or mode not in scenario.modes.teleported:
            continue
        a = origin[legs]
        tele_time[legs], tele_distance[legs] = teleport_many(
            xy[a], xy[a + 1], scenario.modes.teleported[mode])
        if mode in scenario.modes.person_max_speeds:  # per-person speeds
            speed = scenario.modes.teleport_speeds(mode, plans.plan_of_leg()[legs])
            tele_time[legs] = tele_distance[legs] / speed
    fixed = ~is_net & ~is_transit & ~np.isnan(plans.leg_travel_time)
    tele_time[fixed] = plans.leg_travel_time[fixed]
    n_trips = 0 if scenario.transit is None else scenario.transit.schedule.n_departures
    if max(plans.n_plans, plans.n_acts, plans.n_legs, net.n_links, n_trips) >= MAX_INDEX:
        raise ValueError(f"HermesSim supports fewer than {MAX_INDEX} persons, trips, "
                         "activities, legs and links.")
    inputs = dict(
        free_steps=free_steps,
        headway=np.asarray(headway, dtype=float),
        storage=np.asarray(storage, dtype=float),
        act_offsets=plans.act_offsets.astype(np.int64),
        act_end_time=plans.act_end_time.astype(float),
        act_duration=plans.act_duration.astype(float),
        leg_network=is_net.astype(np.uint8),
        leg_teleport_steps=round_half_up(tele_time),
        route_offsets=plans.route_offsets.astype(np.int64),
        route_links=plans.route_links.astype(np.int64),
        stuck_time=float(cfg.stuck_time),
        end_time=math.inf if cfg.end_time is None else float(cfg.end_time),
        event_mask=int(event_mask),
    )
    inputs.update(_transit_inputs(scenario, plans, xy, is_transit, free_steps))
    return inputs, tele_distance


def _transit_inputs(scenario: Scenario, plans: PlanBatch, xy: np.ndarray,
                    is_transit: np.ndarray, free_steps: np.ndarray) -> dict:
    """The kernel's transit arguments (empty without ``scenario.transit``).

    Times become whole seconds: scheduled stop times and trip starts round
    up (nothing happens before the timetable says), walk, door and
    per-passenger times round half up, as teleport times do.
    """
    i8, u1, f8 = np.int64, np.uint8, np.float64
    transit = scenario.transit
    n_legs = plans.n_legs
    if transit is None:
        z = np.zeros(0, i8)
        return dict(
            network_vehicles=0, leg_transit=np.zeros(n_legs, u1),
            ride_offsets=np.zeros(n_legs + 1, i8), ride_board=z, ride_alight=z,
            walk_before_steps=z, walk_after_steps=np.zeros(n_legs, i8),
            rs_stop=z, rs_link_pos=z, rs_await=np.zeros(0, u1), stop_blocking=np.zeros(0, u1),
            trip_k0=z, trip_k1=z, trip_start_step=z, trip_stop_offsets=np.zeros(1, i8),
            trip_stop_arr_step=z, trip_stop_dep_step=z, trip_link_offsets=np.zeros(1, i8),
            trip_links=z, trip_link_steps=z, trip_capacity=np.zeros(0, f8),
            trip_pce=np.zeros(0, f8), trip_access_steps=z, trip_egress_steps=z,
            trip_parallel=np.zeros(0, u1), door_open_steps=0, door_close_steps=0,
        )
    s, v, tc, net = transit.schedule, transit.vehicles, transit.config, scenario.network
    walks = pt_walks(plans, xy, transit)
    route = s.dep_route.astype(i8)
    rso, rlo = s.route_stop_offsets.astype(i8), s.route_link_offsets.astype(i8)
    k0, k1 = rso[route], rso[route + 1]
    n_trip_stops = k1 - k0
    trip_stop_offsets = np.concatenate([[0], np.cumsum(n_trip_stops)]).astype(i8)
    trip_rs = (np.repeat(k0 - trip_stop_offsets[:-1], n_trip_stops)
               + np.arange(trip_stop_offsets[-1]))  # global route stop of each trip stop
    dep = np.repeat(s.dep_time.astype(f8), n_trip_stops)
    n_trip_links = rlo[route + 1] - rlo[route]
    trip_link_offsets = np.concatenate([[0], np.cumsum(n_trip_links)]).astype(i8)
    pos = (np.repeat(rlo[route] - trip_link_offsets[:-1], n_trip_links)
           + np.arange(trip_link_offsets[-1]))
    trip_links = s.route_links[pos].astype(i8)
    # free-speed steps of each trip link, at the vehicle's top speed if lower;
    # the first link is not traversed (the vehicle appears at its end)
    vtype = transit.dep_type
    vmax = np.repeat(v.max_speed[vtype].astype(f8), n_trip_links)
    slower = vmax < net.freespeed[trip_links]
    with np.errstate(divide="ignore"):
        trip_link_steps = np.where(slower, round_half_up(net.length[trip_links] / vmax),
                                   free_steps[trip_links])
    trip_link_steps[trip_link_offsets[:-1][n_trip_links > 0]] = 0
    return dict(
        network_vehicles=int(tc.vehicles == "network"),
        leg_transit=is_transit.astype(u1),
        ride_offsets=plans.ride_offsets.astype(i8),
        ride_board=plans.ride_board_stop.astype(i8),
        ride_alight=plans.ride_alight_stop.astype(i8),
        walk_before_steps=round_half_up(walks.before_time),
        walk_after_steps=round_half_up(walks.after_time),
        rs_stop=s.rs_stop.astype(i8),
        rs_link_pos=s.rs_link_pos.astype(i8),
        rs_await=s.rs_await.astype(u1),
        stop_blocking=s.stop_blocking.astype(u1),
        trip_k0=k0, trip_k1=k1,
        trip_start_step=ceil_steps(s.dep_time),
        trip_stop_offsets=trip_stop_offsets,
        trip_stop_arr_step=ceil_steps(dep + s.rs_arrival[trip_rs]),
        trip_stop_dep_step=ceil_steps(dep + s.rs_departure[trip_rs]),
        trip_link_offsets=trip_link_offsets,
        trip_links=trip_links,
        trip_link_steps=trip_link_steps.astype(i8),
        trip_capacity=v.capacity[vtype].astype(f8),
        trip_pce=v.pce[vtype].astype(f8),
        trip_access_steps=round_half_up(v.access_time[vtype]),
        trip_egress_steps=round_half_up(v.egress_time[vtype]),
        trip_parallel=v.door_parallel[vtype].astype(u1),
        door_open_steps=int(round_half_up(tc.door_open_time)),
        door_close_steps=int(round_half_up(tc.door_close_time)),
    )


# ---------------------------------------------------------------- events


def _dispatch(events: EventsManager, scenario: Scenario, plans: PlanBatch,
              tele_distance: np.ndarray, ev_time, ev_code, stuck, stuck_leg, stuck_link,
              last_time) -> None:
    """Build the recorded events, in recorded order, and hand them out; then
    the stuck events (persons still unfinished when the run ended)."""
    act_types = [plans.act_types[c] for c in plans.act_type.tolist()]
    nodes = plans.act_node.tolist()
    modes = [plans.modes[c] for c in plans.leg_mode.tolist()]
    dist = tele_distance.tolist()
    fns = {kind: events.handlers_for(t) for t, kind in EVENT_KIND.items()}
    transit = None
    if scenario.transit is not None:
        s = scenario.transit.schedule
        transit = (s.rs_stop.tolist(), s.dep_time.tolist(), s.rs_arrival.tolist(),
                   s.rs_departure.tolist())
    # events share one int object per person and link and one float per
    # second, as inline events do, instead of a new object per event field
    n_trips = 0 if transit is None else len(transit[1])
    persons = list(range(max(plans.n_plans, n_trips)))
    links = list(range(scenario.network.n_links))
    times: dict = {}
    ev_time, ev_code = np.asarray(ev_time), np.asarray(ev_code)
    for start in range(0, len(ev_code), _DISPATCH_CHUNK):  # bounded temporary lists
        end = start + _DISPATCH_CHUNK
        code = ev_code[start:end]
        _dispatch_chunk(fns, act_types, nodes, modes, dist, persons, links, times, transit,
                        ev_time[start:end].tolist(), (code >> _KIND_SHIFT).tolist(),
                        ((code >> _PERSON_SHIFT) & _MASK).tolist(), (code & _MASK).tolist())
    on_stuck = fns[EV_STUCK]
    if on_stuck:
        for p, leg, link in zip(np.flatnonzero(stuck).tolist(), np.asarray(stuck_leg).tolist(),
                                np.asarray(stuck_link).tolist()):
            e = PersonStuck(last_time, p, None if leg < 0 else modes[leg],
                            None if link < 0 else link)
            for fn in on_stuck:
                fn(e)


_DISPATCH_CHUNK = 1 << 16


def _dispatch_chunk(fns, act_types, nodes, modes, dist, persons, links, times, transit,
                    ev_time, ev_kind, ev_person, ev_ref):
    for t, kind, p, a in zip(ev_time, ev_kind, ev_person, ev_ref):
        t = times.setdefault(t, t)
        p = persons[p]
        if kind == EV_LINK_ENTER:
            e = LinkEnter(t, p, links[a])
        elif kind == EV_LINK_LEAVE:
            e = LinkLeave(t, p, links[a])
        elif kind == EV_ACT_START:
            e = ActivityStart(t, p, act_types[a], nodes[a])
        elif kind == EV_ACT_END:
            e = ActivityEnd(t, p, act_types[a], nodes[a])
        elif kind == EV_DEPARTURE:
            e = PersonDeparture(t, p, modes[a], nodes[a + p])
        elif kind == EV_ARRIVAL:
            e = PersonArrival(t, p, modes[a], nodes[a + p + 1])
        elif kind == EV_TELEPORT:
            e = Teleportation(t, p, modes[a], dist[a])
        else:
            e = _transit_event(kind, t, p, a, persons, links, transit)
        for fn in fns[kind]:
            fn(e)


def _transit_event(kind, t, p, a, persons, links, transit):
    rs_stop, dep_time, rs_arr, rs_dep = transit
    if kind == EV_VEH_LINK_ENTER:
        return TransitVehicleLinkEnter(t, p, links[a])
    if kind == EV_VEH_LINK_LEAVE:
        return TransitVehicleLinkLeave(t, p, links[a])
    if kind == EV_ENTER_VEH:
        return PersonEntersVehicle(t, p, persons[a])
    if kind == EV_LEAVE_VEH:
        return PersonLeavesVehicle(t, p, persons[a])
    if kind == EV_WAITING:
        return AgentWaitingForPt(t, p, rs_stop[a])
    if kind == EV_VEH_ARRIVE:
        return VehicleArrivesAtFacility(t, p, rs_stop[a], t - (dep_time[p] + rs_arr[a]))
    if kind == EV_VEH_DEPART:
        return VehicleDepartsAtFacility(t, p, rs_stop[a], t - (dep_time[p] + rs_dep[a]))
    return TransitDriverStarts(t, p)


# ---------------------------------------------------------------- kernel


def ceil_step(t: float) -> int:
    """The first whole second at or after ``t`` (tolerating float noise)."""
    return math.ceil(t - EPS)


def run_kernel(*, free_steps, headway, storage, act_offsets, act_end_time, act_duration,
               leg_network, leg_teleport_steps, route_offsets, route_links,
               stuck_time, end_time, event_mask,
               network_vehicles, leg_transit, ride_offsets, ride_board, ride_alight,
               walk_before_steps, walk_after_steps, rs_stop, rs_link_pos, rs_await,
               stop_blocking, trip_k0, trip_k1, trip_start_step, trip_stop_offsets,
               trip_stop_arr_step, trip_stop_dep_step, trip_link_offsets, trip_links,
               trip_link_steps, trip_capacity, trip_pce, trip_access_steps,
               trip_egress_steps, trip_parallel, door_open_steps, door_close_steps):
    """The reference kernel. ``rust/mobslim-hermes/src/lib.rs`` is a line-for-line port.

    Returns ``(act_start, act_end, leg_departure, leg_arrival, link_enter,
    stuck, ev_time, ev_code, stuck_leg, stuck_link, last_time, ride_wait_start,
    ride_board, ride_alight, ride_trip, stop_arrival, stop_departure)``:

    * the trajectory arrays (``NaN`` = did not happen); ``link_enter`` covers
      the plans' route links followed, in network mode, by the trips' links;
    * the recorded events (kinds in ``event_mask``), one entry per event in
      the order they happened: ``ev_code = kind << 58 | person << 29 | ref``,
      where ``ref`` is the activity, leg, link, route stop or trip the event
      refers to (for vehicle events, ``person`` is the trip);
    * for each stuck person, in person order, the leg and link they were on
      (``-1`` = none), and the time the run ended (``last_time``), when they
      count as stuck;
    * the ride arrays and the per trip stop times (``TRANSIT_PLAN.md`` 8).

    Transit follows the QSim's rules (:mod:`mobslim.mobsim._transit`) in
    whole seconds. Vehicles ``0 .. n_plans - 1`` are the persons and
    ``n_plans + d`` is trip ``d``. Stop departures run after every other
    entry of their second (the QSim's "late" entries), so passengers
    arriving in the same second still board.
    """
    def ints(a):
        return np.asarray(a).tolist()

    free_steps, headway, storage = ints(free_steps), ints(headway), ints(storage)
    act_offsets, act_end_time = ints(act_offsets), ints(act_end_time)
    act_duration, leg_network = ints(act_duration), ints(leg_network)
    leg_teleport_steps, route_offsets = ints(leg_teleport_steps), ints(route_offsets)
    route_links = ints(route_links)
    leg_transit, ride_offsets = ints(leg_transit), ints(ride_offsets)
    ride_board, ride_alight = ints(ride_board), ints(ride_alight)
    walk_before_steps, walk_after_steps = ints(walk_before_steps), ints(walk_after_steps)
    rs_stop, rs_link_pos, rs_await = ints(rs_stop), ints(rs_link_pos), ints(rs_await)
    stop_blocking, trip_k0, trip_k1 = ints(stop_blocking), ints(trip_k0), ints(trip_k1)
    trip_start_step, trip_stop_offsets = ints(trip_start_step), ints(trip_stop_offsets)
    trip_stop_arr_step, trip_stop_dep_step = ints(trip_stop_arr_step), ints(trip_stop_dep_step)
    trip_link_offsets, trip_links = ints(trip_link_offsets), ints(trip_links)
    trip_link_steps, trip_capacity = ints(trip_link_steps), ints(trip_capacity)
    trip_pce, trip_access_steps = ints(trip_pce), ints(trip_access_steps)
    trip_egress_steps, trip_parallel = ints(trip_egress_steps), ints(trip_parallel)
    finite_stuck = math.isfinite(stuck_time)
    net_veh = bool(network_vehicles)

    n_plans, n_links = len(act_offsets) - 1, len(free_steps)
    n_trips = len(trip_k0)
    n_plan_links = len(route_links)
    n_veh = n_plans + n_trips
    tr_rstart = [0] * n_trips  # trip d's first route position (network mode)
    if net_veh:
        tr_rstart = [n_plan_links + o for o in trip_link_offsets[:-1]]
        route_links = route_links + trip_links
    nan = math.nan
    t_act_start = [nan] * len(act_end_time)
    t_act_end = [nan] * len(act_end_time)
    t_dep = [nan] * len(leg_network)
    t_arr = [nan] * len(leg_network)
    t_enter = [nan] * len(route_links)
    t_stuck = [False] * n_plans

    # persons
    a_act = act_offsets[:-1]  # current (or last) activity
    a_last = [o - 1 for o in act_offsets[1:]]
    a_leg = [-1] * n_plans
    a_done = [False] * n_plans
    a_ride = [0] * n_plans  # current pt ride
    a_ride_end = [0] * n_plans
    # vehicles (persons, then trips)
    a_rpos = [0] * n_veh  # position of the current route link
    a_rend = [0] * n_veh
    a_link = [-1] * n_veh
    a_blocked: list = [None] * n_veh  # blocked from departing since
    a_waiting = [-1] * n_veh  # registered as a waiter on this link
    v_pce = [1.0] * n_plans + trip_pce
    # links
    l_queue = [deque() for _ in range(n_links)]  # (vehicle, earliest exit)
    l_occ = [0.0] * n_links  # PCE on the link (incl. trips pulled into a stop)
    l_nft = [0.0] * n_links  # next free flow slot (unrounded)
    l_waiters = [deque() for _ in range(n_links)]  # (is_link, link or vehicle)
    l_departing = [deque() for _ in range(n_links)]  # vehicles blocked departing, FIFO
    l_blocked: list = [None] * n_links  # head blocked downstream since
    l_waiting = [-1] * n_links  # head registered as a waiter on this link
    l_pending: list = [None] * n_links  # scheduled try-leave time
    # rides
    n_rides = len(ride_board)
    t_wait = [nan] * n_rides
    t_board = [nan] * n_rides
    t_alight = [nan] * n_rides
    r_trip = [-1] * n_rides
    waiting: list = [None] * len(rs_stop)  # per route stop: deque of persons
    at_stop: list = [None] * len(rs_stop)  # per route stop: trips there
    # trips
    tr_next = list(trip_k0)  # next route stop to serve
    tr_onboard: list = [[] for _ in range(n_trips)]
    tr_load = [0] * n_trips
    tr_visit = [-1] * n_trips  # route stop being served (network mode)
    tr_state = [0] * n_trips
    tr_token = [0] * n_trips
    tr_alight_q: list = [deque() for _ in range(n_trips)]
    tr_alight_busy = [False] * n_trips
    tr_board_busy = [False] * n_trips
    tr_boarding = [-1] * n_trips
    tr_pulled = [False] * n_trips
    tr_pulled_earliest = [0] * n_trips
    t_stop_arr = [nan] * trip_stop_offsets[-1]
    t_stop_dep = [nan] * trip_stop_offsets[-1]

    rec = [bool(event_mask & (1 << k)) for k in range(N_EVENT_KINDS)]
    ev_time, ev_code = [], []

    def record(kind, p, ref):
        ev_time.append(now)
        ev_code.append(kind << _KIND_SHIFT | p << _PERSON_SHIFT | ref)

    # agenda: second -> entries in push order (and "late" entries, run after
    # the others of their second), and a heap of the seconds
    buckets: dict[int, list] = {}
    lates: dict[int, list] = {}
    seconds: list[int] = []
    now = 0

    def schedule(t, kind, obj):
        bucket = buckets.get(t)
        if bucket is None:
            bucket = buckets[t] = []
            if t not in lates:
                heapq.heappush(seconds, t)
        bucket.append((kind, obj))

    def schedule_late(t, kind, obj):
        bucket = lates.get(t)
        if bucket is None:
            bucket = lates[t] = []
            if t not in buckets:
                heapq.heappush(seconds, t)
        bucket.append((kind, obj))

    def planned_end(a, start):
        e = act_end_time[a]
        if e == e:
            return start if start > e else ceil_step(e)
        d = act_duration[a]
        if d == d:
            return max(start, ceil_step(start + d))
        return None

    # -- activities and legs ------------------------------------------------
    def end_activity(p):
        a = a_act[p]
        t_act_end[a] = now
        if rec[EV_ACT_END]:
            record(EV_ACT_END, p, a)
        leg = a - p
        a_leg[p] = leg
        t_dep[leg] = now
        if rec[EV_DEPARTURE]:
            record(EV_DEPARTURE, p, leg)
        if leg_network[leg]:
            r0, r1 = route_offsets[leg], route_offsets[leg + 1]
            if r0 == r1:
                arrive(p)
                return
            a_rpos[p], a_rend[p] = r0, r1
            try_depart(p, False)
        elif leg_transit[leg]:
            pt_depart(p)
        else:
            schedule(now + leg_teleport_steps[leg], _TELEPORT_ARRIVE, p)

    def teleport_arrive(p):
        if rec[EV_TELEPORT]:
            record(EV_TELEPORT, p, a_leg[p])
        arrive(p)

    def arrive(p):
        leg = a_leg[p]
        t_arr[leg] = now
        if rec[EV_ARRIVAL]:
            record(EV_ARRIVAL, p, leg)
        a_leg[p] = -1
        a = a_act[p] = a_act[p] + 1
        t_act_start[a] = now
        if rec[EV_ACT_START]:
            record(EV_ACT_START, p, a)
        if a == a_last[p]:
            a_done[p] = True
            return
        t = planned_end(a, now)
        if t is not None:
            schedule(t, _ACT_END, p)

    # -- departing onto the first link --------------------------------------
    def try_depart_event(v):
        if a_blocked[v] is not None:  # ignore stale wake-ups
            try_depart(v, False)

    def force_depart(v):
        since = a_blocked[v]
        if since is not None and now >= ceil_step(since + stuck_time):
            try_depart(v, True)

    def try_depart(v, force):
        link = route_links[a_rpos[v]]
        line = l_departing[link]
        if line and line[0] != v:  # others are waiting to depart here
            line.append(v)
            return
        if force or l_occ[link] < storage[link]:
            a_blocked[v] = None
            a_waiting[v] = -1
            if line:  # it was the front of the line: the next one tries now
                line.popleft()
            enter_link(v, link)
            if line:
                try_depart(line[0], False)
            return
        if not line:
            line.append(v)
        was_blocked = a_blocked[v] is not None
        if not was_blocked:
            a_blocked[v] = now
            if finite_stuck:
                schedule(ceil_step(now + stuck_time), _FORCE_DEPART, v)
        if a_waiting[v] != link:
            a_waiting[v] = link
            if was_blocked:
                l_waiters[link].appendleft((False, v))
            else:
                l_waiters[link].append((False, v))

    # -- links --------------------------------------------------------------
    def enter_link(v, link):
        t_enter[a_rpos[v]] = now
        if v < n_plans:
            if rec[EV_LINK_ENTER]:
                record(EV_LINK_ENTER, v, link)
            earliest = now + free_steps[link]
        else:
            if rec[EV_VEH_LINK_ENTER]:
                record(EV_VEH_LINK_ENTER, v - n_plans, link)
            earliest = now + trip_link_steps[a_rpos[v] - n_plan_links]
        a_link[v] = link
        l_occ[link] += v_pce[v]
        queue = l_queue[link]
        queue.append((v, earliest))
        if len(queue) == 1:
            schedule_link(link, head_time(link))

    def head_time(link):
        """When the head may act: its earliest exit, and also the next flow
        slot unless it is a trip about to serve a stop."""
        v, earliest = l_queue[link][0]
        if v >= n_plans and has_stop_here(v - n_plans):
            return max(earliest, now)  # it may have queued past its earliest exit
        return max(earliest, ceil_step(l_nft[link]))

    def schedule_link(link, t):
        """Ensure exactly one pending try-leave at or before ``t``."""
        pending = l_pending[link]
        if pending is not None and pending <= t:
            return
        l_pending[link] = t
        schedule(t, _TRY_LEAVE, link)

    def try_leave_event(link):
        if l_pending[link] != now:
            return  # superseded by an earlier wake-up
        l_pending[link] = None
        try_leave(link, False)

    def force_leave(link):
        since = l_blocked[link]
        if since is not None and now >= ceil_step(since + stuck_time):
            try_leave(link, True)

    def try_leave(link, force):
        queue = l_queue[link]
        if not queue:
            return
        v, earliest = queue[0]
        if v >= n_plans and at_link_end(v - n_plans, link, earliest):
            return
        ready = max(earliest, ceil_step(l_nft[link]))
        if now < ready:
            schedule_link(link, ready)
            return
        next_pos = a_rpos[v] + 1
        if next_pos == a_rend[v]:  # leaving the last link: arrive
            pop_head(link, ready)
            if v < n_plans:
                arrive(v)
            return
        nxt = route_links[next_pos]
        if force or l_occ[nxt] < storage[nxt]:
            pop_head(link, ready)
            a_rpos[v] = next_pos
            enter_link(v, nxt)
            return
        # blocked by downstream storage: wait for space (or stuck time)
        was_blocked = l_blocked[link] is not None
        if not was_blocked:
            l_blocked[link] = now
            if finite_stuck:
                schedule(ceil_step(now + stuck_time), _FORCE_LEAVE, link)
        if l_waiting[link] != nxt:
            l_waiting[link] = nxt
            if was_blocked:
                l_waiters[nxt].appendleft((True, link))
            else:
                l_waiters[nxt].append((True, link))

    def pop_head(link, ready):
        queue = l_queue[link]
        v, earliest = queue.popleft()
        if v < n_plans:
            if rec[EV_LINK_LEAVE]:
                record(EV_LINK_LEAVE, v, link)
        elif rec[EV_VEH_LINK_LEAVE]:
            record(EV_VEH_LINK_LEAVE, v - n_plans, link)
        a_link[v] = -1
        l_occ[link] -= v_pce[v]
        # left at its first possible second: count the next slot from the
        # unrounded time (no drift); held up: from now
        nft = l_nft[link]
        base = (earliest if earliest > nft else nft) if now == ready else now
        nft = l_nft[link] = base + headway[link] * v_pce[v]
        l_blocked[link] = None
        l_waiting[link] = -1
        if queue:
            schedule_link(link, head_time(link))
        # a slot was freed: wake the first valid waiter
        waiters = l_waiters[link]
        while waiters:
            is_link, obj = waiters.popleft()
            if is_link:
                if l_waiting[obj] != link:
                    continue  # stale token
                l_waiting[obj] = -1
                schedule_link(obj, now)
            else:
                if a_waiting[obj] != link:
                    continue
                a_waiting[obj] = -1
                schedule(now, _TRY_DEPART, obj)
            return

    # -- passengers ---------------------------------------------------------
    def pt_depart(p):
        leg = a_leg[p]
        r0, r1 = ride_offsets[leg], ride_offsets[leg + 1]
        if r0 == r1:  # walk only
            schedule(now + walk_after_steps[leg], _PAX_ARRIVE, p)
            return
        a_ride[p], a_ride_end[p] = r0, r1
        schedule(now + walk_before_steps[r0], _PAX_AT_STOP, p)

    def pax_at_stop(p):
        r = a_ride[p]
        k = ride_board[r]
        t_wait[r] = now
        if rec[EV_WAITING]:
            record(EV_WAITING, p, k)
        trips = at_stop[k]
        if not net_veh:
            if trips:
                for d in trips:
                    if tr_load[d] < trip_capacity[d]:
                        tr_load[d] += 1
                        board(d, p)
                        return
            wait(k, p)
            return
        wait(k, p)
        if trips:
            for d in trips:
                if serve_new_boarder(d):
                    return

    def wait(k, p):
        q = waiting[k]
        if q is None:
            q = waiting[k] = deque()
        q.append(p)

    def board(d, p):
        r = a_ride[p]
        t_board[r] = now
        r_trip[r] = d
        tr_onboard[d].append(p)
        if rec[EV_ENTER_VEH]:
            record(EV_ENTER_VEH, p, d)

    def alight(d, p):
        r = a_ride[p]
        t_alight[r] = now
        tr_onboard[d].remove(p)
        tr_load[d] -= 1
        if rec[EV_LEAVE_VEH]:
            record(EV_LEAVE_VEH, p, d)
        if r + 1 < a_ride_end[p]:
            a_ride[p] = r + 1
            schedule(now + walk_before_steps[r + 1], _PAX_AT_STOP, p)
        else:
            schedule(now + walk_after_steps[a_leg[p]], _PAX_ARRIVE, p)

    def alighters(d, k):
        return [p for p in tr_onboard[d] if ride_alight[a_ride[p]] == k]

    # -- trips: common ------------------------------------------------------
    def trip_start(d):
        if rec[EV_DRIVER_START]:
            record(EV_DRIVER_START, d, 0)
        if net_veh:
            v = n_plans + d
            a_rpos[v] = tr_rstart[d]
            a_rend[v] = tr_rstart[d] + trip_link_offsets[d + 1] - trip_link_offsets[d]
            try_depart(v, False)
        else:
            schedule(max(now, trip_stop_arr_step[trip_stop_offsets[d]]), _STOP_ARRIVE, d)

    def record_arrival(d, k):
        t_stop_arr[trip_stop_offsets[d] + k - trip_k0[d]] = now
        if rec[EV_VEH_ARRIVE]:
            record(EV_VEH_ARRIVE, d, k)
        trips = at_stop[k]
        if trips is None:
            at_stop[k] = [d]
        else:
            trips.append(d)

    def stop_depart(item):
        d, token = item
        if token != tr_token[d]:
            return
        if net_veh and tr_state[d] != _CLOSED:
            return
        k = tr_next[d]
        j = trip_stop_offsets[d] + k - trip_k0[d]
        t_stop_dep[j] = now
        if rec[EV_VEH_DEPART]:
            record(EV_VEH_DEPART, d, k)
        at_stop[k].remove(d)
        tr_next[d] = k + 1
        tr_visit[d] = -1
        tr_state[d] = 0
        if not net_veh:
            if k + 1 < trip_k1[d]:
                schedule(max(now, trip_stop_arr_step[j + 1]), _STOP_ARRIVE, d)
            return
        if has_stop_here(d):  # several stops on one link
            reach_stop(d)
            return
        link = route_links[a_rpos[n_plans + d]]
        if tr_pulled[d]:  # back from the stop into the lane, as its head
            tr_pulled[d] = False
            l_queue[link].appendleft((n_plans + d, tr_pulled_earliest[d]))
            l_blocked[link] = None
            l_waiting[link] = -1
        schedule_link(link, max(now, ceil_step(l_nft[link])))

    # -- trips: schedule mode -----------------------------------------------
    def sched_arrive(d):
        k = tr_next[d]
        record_arrival(d, k)
        for p in alighters(d, k):
            alight(d, p)
        q = waiting[k]
        while q and tr_load[d] < trip_capacity[d]:
            tr_load[d] += 1
            board(d, q.popleft())
        tr_token[d] += 1
        t = trip_stop_dep_step[trip_stop_offsets[d] + k - trip_k0[d]]
        schedule_late(max(now, t), _STOP_DEPART, (d, tr_token[d]))

    # -- trips: network mode ------------------------------------------------
    def has_stop_here(d):
        k = tr_next[d]
        return k < trip_k1[d] and rs_link_pos[k] == a_rpos[n_plans + d] - tr_rstart[d]

    def at_link_end(d, link, earliest):
        """``try_leave`` hook for trip ``d`` at the head of ``link``: ``True``
        if it is (now) serving a stop instead of leaving."""
        if tr_visit[d] >= 0:
            return True
        if not has_stop_here(d):
            return False
        if now < earliest:
            schedule_link(link, earliest)
            return True
        if not stop_blocking[rs_stop[tr_next[d]]]:
            queue = l_queue[link]
            queue.popleft()  # into the stop: off the lane, still in storage
            tr_pulled[d] = True
            tr_pulled_earliest[d] = earliest
            if queue:
                schedule_link(link, head_time(link))
        reach_stop(d)
        return True

    def reach_stop(d):
        k = tr_next[d]
        tr_visit[d] = k
        record_arrival(d, k)
        tr_alight_q[d] = deque(alighters(d, k))
        if tr_alight_q[d] or boarders(d):
            open_doors(d)
        else:
            ready(d)

    def boarders(d):
        q = waiting[tr_visit[d]]
        return bool(q) and tr_load[d] - len(tr_alight_q[d]) < trip_capacity[d]

    def open_doors(d):
        tr_state[d] = _OPENING
        tr_token[d] += 1
        schedule(now + door_open_steps, _DOOR_OPEN, (d, tr_token[d]))

    def door_open(item):
        d, token = item
        if token != tr_token[d] or tr_state[d] != _OPENING:
            return
        tr_state[d] = _OPEN
        next_alight(d)
        if trip_parallel[d] or not tr_alight_busy[d]:
            next_board(d)
        maybe_close(d)

    def next_alight(d):
        busy = tr_alight_busy[d] = bool(tr_alight_q[d])
        if busy:
            schedule(now + trip_egress_steps[d], _ALIGHT_DONE, d)

    def alight_done(d):
        alight(d, tr_alight_q[d].popleft())
        next_alight(d)
        if not trip_parallel[d] and not tr_alight_busy[d] and not tr_board_busy[d]:
            next_board(d)
        maybe_close(d)

    def next_board(d):
        if boarders(d):
            p = waiting[tr_visit[d]].popleft()
            tr_load[d] += 1
            tr_board_busy[d] = True
            tr_boarding[d] = p
            schedule(now + trip_access_steps[d], _BOARD_DONE, d)
        else:
            tr_board_busy[d] = False

    def board_done(d):
        p = tr_boarding[d]
        tr_boarding[d] = -1
        board(d, p)
        next_board(d)
        maybe_close(d)

    def maybe_close(d):
        if tr_state[d] == _OPEN and not tr_alight_busy[d] and not tr_board_busy[d]:
            tr_state[d] = _CLOSING
            schedule(now + door_close_steps, _DOOR_CLOSE, d)

    def door_close(d):
        if tr_state[d] == _CLOSING:
            ready(d)

    def ready(d):
        """Doors closed and nobody to serve: depart (or await the departure)."""
        if boarders(d):
            open_doors(d)
            return
        tr_state[d] = _CLOSED
        k = tr_visit[d]
        t = now
        if rs_await[k]:
            t = max(now, trip_stop_dep_step[trip_stop_offsets[d] + k - trip_k0[d]])
        tr_token[d] += 1
        schedule_late(t, _STOP_DEPART, (d, tr_token[d]))

    def serve_new_boarder(d):
        """A passenger joined the line at trip ``d``'s stop. ``True`` if it takes them."""
        if not boarders(d):
            return False
        state = tr_state[d]
        if state == _OPEN:
            if not tr_board_busy[d] and (trip_parallel[d] or not tr_alight_busy[d]):
                next_board(d)
            return True
        if state == _CLOSED:
            open_doors(d)
            return True
        return state == _OPENING or state == _CLOSING

    # -- main loop ----------------------------------------------------------
    for d in range(n_trips):  # trips first, as the QSim schedules them
        schedule(max(0, trip_start_step[d]), _TRIP_START, d)
    for p in range(n_plans):
        if a_act[p] == a_last[p]:
            a_done[p] = True
            continue
        t = planned_end(a_act[p], 0)
        if t is not None:
            schedule(t, _ACT_END, p)

    handlers = (end_activity, try_leave_event, force_leave, try_depart_event,
                force_depart, teleport_arrive, trip_start, sched_arrive, stop_depart,
                door_open, alight_done, board_done, door_close, pax_at_stop, arrive)
    while seconds:
        if seconds[0] > end_time:
            break
        now = heapq.heappop(seconds)
        bucket = buckets.setdefault(now, [])
        late = lates.setdefault(now, [])
        i = j = 0
        while True:  # entries pushed for `now` while draining are appended
            while i < len(bucket):
                kind, obj = bucket[i]
                i += 1
                handlers[kind](obj)
            if j == len(late):
                break
            kind, obj = late[j]
            j += 1
            handlers[kind](obj)
        del buckets[now], lates[now]

    if math.isfinite(end_time):
        now = max(now, end_time)
    stuck_leg, stuck_link = [], []
    for p in range(n_plans):
        if not a_done[p]:
            t_stuck[p] = True
            stuck_leg.append(a_leg[p])
            stuck_link.append(a_link[p])

    f8, i8 = np.float64, np.int64
    return (np.array(t_act_start, f8), np.array(t_act_end, f8), np.array(t_dep, f8),
            np.array(t_arr, f8), np.array(t_enter, f8), np.array(t_stuck, bool),
            np.array(ev_time, f8), np.array(ev_code, i8), np.array(stuck_leg, i8),
            np.array(stuck_link, i8), float(now), np.array(t_wait, f8),
            np.array(t_board, f8), np.array(t_alight, f8), np.array(r_trip, i8),
            np.array(t_stop_arr, f8), np.array(t_stop_dep, f8))
