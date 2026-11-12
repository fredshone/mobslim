"""An event-driven queue simulation, modelled on MATSim's QSim.

Link model (per link, as in MATSim's queue model):

* **Free-speed time**: a vehicle cannot leave a link before
  ``length / freespeed`` seconds after entering it.
* **Flow capacity**: vehicles leave a link at most every
  ``1 / (capacity * flow_capacity_factor)`` seconds (fractional headways, no
  rounding). Capacity is per link, not per lane, as in MATSim.
* **Storage capacity**: a link holds at most
  ``length * lanes / effective_cell_size * storage_capacity_factor``
  vehicles (at least enough to not restrict its own flow, and at least 1).
  A full link blocks upstream vehicles (spillback).
* **FIFO**: only the vehicle at the head of a link's queue may leave.
* **Stuck time**: a vehicle blocked for ``stuck_time`` seconds is pushed onto
  the next link regardless of storage (prevents gridlock).
* **Departures** onto a full link wait in a FIFO line per link (MATSim's
  waiting list and buffer): only the front waits for space and counts stuck
  time, from when it reached the front. So a full link takes at most one
  forced departure per ``stuck_time``, as in MATSim, not the whole line.

Unlike a fixed-time-step simulation, nothing is polled. Each link schedules
a single "try to leave" wake-up for its head vehicle. When a link frees space it
wakes vehicles waiting to enter it. That keeps the cost proportional to the
number of moves, not to ``agents x seconds``.

Legs whose mode is not a network mode are teleported. With
``scenario.transit``, pt legs ride scheduled vehicles that run on the
timetable or in the queue model (see :mod:`mobslim.mobsim._transit`), with
every timing setting (see :mod:`mobslim.mobsim._transit_tables`).
Vehicles take ``pce`` units of storage and flow (cars 1).

Data in and out are arrays. The QSim encodes the selected plans once per run
as a :class:`~mobslim.planbatch.PlanBatch` (routes are validated there, in
one vectorised pass) and returns :class:`~mobslim.trajectories.Trajectories`
aligned with it. Events are optional: an event object is only built if some
handler is registered for its type, so a run with no event handlers builds
none at all.
"""

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from mobslim.events import (
    ActivityEnd,
    ActivityStart,
    EventsManager,
    LinkEnter,
    LinkLeave,
    PersonArrival,
    PersonDeparture,
    PersonStuck,
    Teleportation,
)
from mobslim.mobsim._common import EPS as _EPS
from mobslim.mobsim._common import (
    ceil_step,
    check_timing,
    link_parameters,
    planned_end,
    round_steps,
)
from mobslim.mobsim._common import fire as _fire
from mobslim.mobsim._common import none_if_nan as _none_if_nan
from mobslim.mobsim._common import validate as _validate
from mobslim.mobsim._transit import _TransitRun
from mobslim.planbatch import PlanBatch
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories

# scheduler event kinds
_ACT_END = 0
_TRY_LEAVE = 1
_FORCE_LEAVE = 2
_TRY_DEPART = 3
_FORCE_DEPART = 4
_TELEPORT_ARRIVE = 5
_ACT_START = 6  # an activity starting after a junction delay
_LATE = 1 << 62  # sequence offset: after every ordinary event at the same time


@dataclass(frozen=True)
class QSimParams:
    flow_capacity_factor: float = 1.0
    storage_capacity_factor: float = 1.0
    stuck_time: float = 10.0
    effective_cell_size: float = 7.5  # meters per vehicle (PCU)
    end_time: Optional[float] = None  # None: run until every plan is done
    validate_routes: bool = True
    # timing (see mobslim.mobsim._common.check_timing). Whole seconds give
    # HermesSim's rules (and, with the same timing, HermesSim's results)
    continuous_time: bool = True
    junction_delays: bool = False
    rounding: str = "nearest"

    def __post_init__(self):
        check_timing("QSim", self.continuous_time, self.junction_delays, self.rounding,
                     continuous=True, whole=True)


class QSim:
    """Queue-based mobsim. Stateless between runs; safe to reuse."""

    def __init__(self, params: QSimParams = QSimParams()):
        self.params = params

    sums_traversals = True  # takes ``traversal_bins`` (see run_mobsim)

    def run(self, scenario: Scenario, events: EventsManager,
            traversal_bins: Optional[tuple[float, float]] = None,
            link_times: bool = True) -> Trajectories:
        """``traversal_bins`` and ``link_times`` are accepted for the common
        interface (see :func:`~mobslim.mobsim.base.run_mobsim`); this mobsim
        always records link times and leaves the sums to
        :meth:`Trajectories.traversal_aggregate`, with the same result."""
        plans = scenario.population.selected_batch()
        return self.run_batch(scenario, plans, events)

    def run_batch(self, scenario: Scenario, plans: PlanBatch, events: EventsManager) -> Trajectories:
        """Run already-encoded plans (``plans`` row ``i`` = person index ``i``)."""
        if plans.n_plans != len(scenario.population):
            raise ValueError("PlanBatch must have one plan per person.")
        if self.params.validate_routes:
            _validate(scenario, plans)
        return _QSimRun(scenario, plans, events, self.params).run()


class _Agent:
    __slots__ = ("index", "act", "last_act", "leg", "rpos", "rend",
                 "link", "blocked_since", "waiting_on", "distance", "done",
                 "ride", "ride_end")
    is_trip = False
    pce = 1.0

    def __init__(self, index: int, first_act: int, last_act: int):
        self.index = index
        self.act = first_act  # global index of the current (or last) activity
        self.last_act = last_act
        self.leg: Optional[int] = None  # global index of the current leg
        self.rpos = 0  # global position of the current route link
        self.rend = 0  # end of the current route (exclusive)
        self.link: Optional[int] = None  # link the agent is currently on
        self.blocked_since: Optional[float] = None  # waiting to depart
        self.waiting_on: Optional[int] = None  # registered as waiter on link
        self.distance = 0.0  # teleport distance
        self.done = False
        self.ride = 0  # current pt ride (global index)
        self.ride_end = 0


class _Link:
    __slots__ = ("index", "free_time", "headway", "storage", "queue",
                 "next_flow_time", "waiters", "blocked_since", "waiting_on",
                 "pending", "occupied", "departing")

    def __init__(self, index: int, free_time: float, headway: float, storage: float):
        self.index = index
        self.free_time = free_time
        self.headway = headway
        self.storage = storage
        self.queue: deque = deque()  # (agent, earliest_exit)
        self.next_flow_time = 0.0
        self.waiters: deque = deque()  # tokens: (is_link, link_or_agent)
        self.blocked_since: Optional[float] = None  # head blocked downstream
        self.waiting_on: Optional[int] = None  # head registered on this link
        self.pending: Optional[float] = None  # scheduled try-leave time
        self.occupied = 0.0  # PCE on the link (incl. transit vehicles at a stop)
        self.departing: deque = deque()  # agents blocked departing onto it, FIFO

    def has_space(self) -> bool:
        return self.occupied < self.storage


class _QSimRun:
    """State for a single mobsim run."""

    def __init__(self, scenario: Scenario, plans: PlanBatch, events: EventsManager,
                 cfg: QSimParams):
        self.scenario = scenario
        self.plans = plans
        self.cfg = cfg
        self.now = 0.0
        self.heap: list = []
        self.seq = itertools.count()
        self.late_seq = itertools.count(_LATE)

        # live handler lists per event type; an event is only built if its
        # list is non-empty
        h = events.handlers_for
        self.on_act_start, self.on_act_end = h(ActivityStart), h(ActivityEnd)
        self.on_departure, self.on_arrival = h(PersonDeparture), h(PersonArrival)
        self.on_enter, self.on_leave = h(LinkEnter), h(LinkLeave)
        self.on_teleport, self.on_stuck = h(Teleportation), h(PersonStuck)

        net = scenario.network
        self.modes = scenario.modes
        free_time, headway, storage = link_parameters(
            net, cfg.flow_capacity_factor, cfg.storage_capacity_factor, cfg.effective_cell_size
        )
        # whole seconds: HermesSim's rules (rounded road and teleport times,
        # activity ends and forced moves at the next whole second, exits at
        # the first whole second at or after the next flow slot)
        self.whole = not cfg.continuous_time
        if self.whole:
            free_time = round_steps(free_time, cfg.rounding).astype(float).tolist()
        self.free_time = free_time
        self.junction = 1.0 if cfg.junction_delays else 0.0
        self.links = [_Link(i, free_time[i], headway[i], storage[i]) for i in range(net.n_links)]

        # plan arrays as Python lists: element access from Python is several
        # times faster on lists than on numpy arrays
        self.act_type = [plans.act_types[c] for c in plans.act_type.tolist()]
        self.act_node = plans.act_node.tolist()
        self.act_xy = plans.resolved_act_xy(net).tolist()
        self.act_end_time = _none_if_nan(plans.act_end_time)
        self.act_duration = _none_if_nan(plans.act_duration)
        self.leg_mode = [plans.modes[c] for c in plans.leg_mode.tolist()]
        self.leg_network = plans.network_legs(self.modes.network_modes).tolist()
        self.leg_fixed_time = _none_if_nan(plans.leg_travel_time)
        self.route_offsets = plans.route_offsets.tolist()
        self.route_links = plans.route_links.tolist()
        self.leg_transit = plans.mode_legs(self.modes.transit_modes).tolist()

        # trajectories, filled as the run goes
        nan = math.nan
        self.t_act_start = [nan] * plans.n_acts
        self.t_act_end = [nan] * plans.n_acts
        self.t_departure = [nan] * plans.n_legs
        self.t_arrival = [nan] * plans.n_legs
        self.t_teleport = [0.0] * plans.n_legs
        self.t_enter = [nan] * len(self.route_links)
        self.t_stuck = [False] * plans.n_plans

        offsets = plans.act_offsets.tolist()
        self.agents = [_Agent(i, offsets[i], offsets[i + 1] - 1) for i in range(plans.n_plans)]
        self.transit = None
        if scenario.transit is not None:
            self.transit = _TransitRun(self, scenario.transit, plans,
                                       plans.resolved_act_xy(net), events)

    def schedule(self, time: float, kind: int, obj) -> None:
        heapq.heappush(self.heap, (time, next(self.seq), kind, obj))

    def schedule_late(self, time: float, kind: int, obj) -> None:
        """Like :meth:`schedule`, but after every ordinary event at ``time``."""
        heapq.heappush(self.heap, (time, next(self.late_seq), kind, obj))

    def _schedule_stuck(self, kind: int, obj) -> None:
        if math.isfinite(self.cfg.stuck_time):  # inf: never force (gridlock possible)
            t = self.now + self.cfg.stuck_time
            self.schedule(float(ceil_step(t)) if self.whole else t, kind, obj)

    def _planned_end(self, act: int, start: float) -> Optional[float]:
        """MATSim's ``tryEndTimeThenDuration`` (see ``Activity.planned_end``)."""
        t = planned_end(self.act_end_time[act], self.act_duration[act], start)
        return float(ceil_step(t)) if self.whole and t is not None else t

    def _flow_slot(self, link: _Link) -> float:
        """When the head may leave as far as flow capacity goes."""
        nft = link.next_flow_time
        return float(ceil_step(nft)) if self.whole else nft

    # -- main loop ----------------------------------------------------------
    def run(self) -> Trajectories:
        for agent in self.agents:
            if agent.act == agent.last_act:
                agent.done = True
                continue
            self.schedule(self._planned_end(agent.act, 0.0), _ACT_END, agent)

        end_time = self.cfg.end_time
        handlers = {
            _ACT_END: self._end_activity,
            _TRY_LEAVE: self._try_leave_event,
            _FORCE_LEAVE: self._force_leave,
            _TRY_DEPART: self._try_depart_event,
            _FORCE_DEPART: self._force_depart,
            _TELEPORT_ARRIVE: self._teleport_arrive,
            _ACT_START: self._start_activity,
        }
        if self.transit is not None:
            handlers.update(self.transit.handlers())
        while self.heap:
            if end_time is not None and self.heap[0][0] > end_time:
                break
            self.now, _, kind, obj = heapq.heappop(self.heap)
            handlers[kind](obj)

        if end_time is not None:
            self.now = max(self.now, float(end_time))
        self._abort_unfinished()
        return Trajectories.build(
            self.plans, self.scenario.network,
            act_start=self.t_act_start, act_end=self.t_act_end,
            leg_departure=self.t_departure, leg_arrival=self.t_arrival,
            link_enter=self.t_enter[:len(self.plans.route_links)], stuck=self.t_stuck,
            teleport_distance=self.t_teleport,
            network_modes=self.scenario.modes.network_modes,
            **({} if self.transit is None else self.transit.results()),
        )

    def _abort_unfinished(self) -> None:
        for agent in self.agents:
            if agent.done:
                continue
            self.t_stuck[agent.index] = True
            if self.on_stuck:
                mode = None if agent.leg is None else self.leg_mode[agent.leg]
                _fire(self.on_stuck, PersonStuck(self.now, agent.index, mode, agent.link))
            agent.done = True

    # -- activities ---------------------------------------------------------
    def _end_activity(self, agent: _Agent) -> None:
        a = agent.act
        self.t_act_end[a] = self.now
        if self.on_act_end:
            _fire(self.on_act_end, ActivityEnd(self.now, agent.index, self.act_type[a], self.act_node[a]))
        agent.leg = a - agent.index
        self._depart(agent)

    def _start_activity(self, agent: _Agent) -> None:
        a = agent.act
        self.t_act_start[a] = self.now
        if self.on_act_start:
            _fire(self.on_act_start,
                  ActivityStart(self.now, agent.index, self.act_type[a], self.act_node[a]))
        if a == agent.last_act:
            agent.done = True
            return
        self.schedule(self._planned_end(a, self.now), _ACT_END, agent)

    # -- legs ---------------------------------------------------------------
    def _depart(self, agent: _Agent) -> None:
        leg = agent.leg
        origin = self.act_node[agent.act]
        self.t_departure[leg] = self.now
        if self.on_departure:
            _fire(self.on_departure,
                  PersonDeparture(self.now, agent.index, self.leg_mode[leg], origin))
        if self.leg_network[leg]:
            r0, r1 = self.route_offsets[leg], self.route_offsets[leg + 1]
            if r0 == r1:
                self._arrive(agent)
                return
            agent.rpos, agent.rend = r0, r1
            self._try_depart(agent, force=False)
        elif self.leg_transit[leg]:
            self.transit.depart(agent)
        else:
            a = agent.act
            tt, agent.distance = self.modes.teleport(
                self.leg_mode[leg], self.act_xy[a], self.act_xy[a + 1], self.leg_fixed_time[leg],
                agent.index,
            )
            if self.whole:
                tt = float(round_steps(tt, self.cfg.rounding))
            self.schedule(self.now + tt, _TELEPORT_ARRIVE, agent)

    def _teleport_arrive(self, agent: _Agent) -> None:
        self.t_teleport[agent.leg] = agent.distance
        if self.on_teleport:
            _fire(self.on_teleport, Teleportation(
                self.now, agent.index, self.leg_mode[agent.leg], agent.distance))
        self._arrive(agent)

    def _arrive(self, agent: _Agent) -> None:
        if agent.is_trip:  # a transit vehicle leaves the network
            agent.link = None
            return
        leg = agent.leg
        self.t_arrival[leg] = self.now
        if self.on_arrival:
            _fire(self.on_arrival, PersonArrival(
                self.now, agent.index, self.leg_mode[leg], self.act_node[agent.act + 1]))
        agent.leg = None
        agent.act += 1
        if self.junction:  # the activity starts one second later
            self.schedule(self.now + self.junction, _ACT_START, agent)
        else:
            self._start_activity(agent)

    # departure: entering the first link of the route
    def _try_depart_event(self, agent: _Agent) -> None:
        if agent.blocked_since is not None:  # ignore stale wake-ups
            self._try_depart(agent, force=False)

    def _force_depart(self, agent: _Agent) -> None:
        if agent.blocked_since is not None and self.now >= agent.blocked_since + self.cfg.stuck_time - _EPS:
            self._try_depart(agent, force=True)

    def _try_depart(self, agent: _Agent, force: bool) -> None:
        link = self.links[self.route_links[agent.rpos]]
        line = link.departing
        if line and line[0] is not agent:  # others are waiting to depart here
            line.append(agent)
            return
        if force or link.has_space():
            agent.blocked_since = None
            agent.waiting_on = None
            if line:  # it was the front of the line: the next one tries now
                line.popleft()
            self._enter_link(agent, link)
            if line:
                self._try_depart(line[0], force=False)
            return
        if not line:
            line.append(agent)
        was_blocked = agent.blocked_since is not None
        if not was_blocked:
            agent.blocked_since = self.now
            self._schedule_stuck(_FORCE_DEPART, agent)
        if agent.waiting_on != link.index:
            agent.waiting_on = link.index
            token = (False, agent)
            link.waiters.appendleft(token) if was_blocked else link.waiters.append(token)

    # -- links --------------------------------------------------------------
    def _enter_link(self, agent: _Agent, link: _Link) -> None:
        self.t_enter[agent.rpos] = self.now
        if agent.is_trip:
            self.transit.link_enter(agent, link.index)
            earliest = self.transit.first_link_earliest(agent, link, self.now)
        else:
            if self.on_enter:
                _fire(self.on_enter, LinkEnter(self.now, agent.index, link.index))
            earliest = self.now + link.free_time
            if agent.rpos + 1 < agent.rend:  # crosses a junction after it
                earliest += self.junction
        agent.link = link.index
        link.occupied += agent.pce
        link.queue.append((agent, earliest))
        if len(link.queue) == 1:
            self._schedule_link(link, self._head_time(link))

    def _head_time(self, link: _Link) -> float:
        """When the head of ``link`` may act: its earliest exit, and also the
        next flow slot unless it is a transit vehicle about to serve a stop."""
        agent, earliest = link.queue[0]
        if agent.is_trip and self.transit.has_stop_here(agent):
            # it may have queued past the time it reaches the stop
            return max(self.transit.reach_time(agent, earliest), self.now)
        return max(earliest, self._flow_slot(link))

    def _schedule_link(self, link: _Link, time: float) -> None:
        """Ensure exactly one pending try-leave at or before ``time``."""
        if link.pending is not None and link.pending <= time:
            return
        link.pending = time
        self.schedule(time, _TRY_LEAVE, link)

    def _try_leave_event(self, link: _Link) -> None:
        if link.pending != self.now:
            return  # superseded by an earlier wake-up
        link.pending = None
        self._try_leave(link, force=False)

    def _force_leave(self, link: _Link) -> None:
        if link.blocked_since is not None and self.now >= link.blocked_since + self.cfg.stuck_time - _EPS:
            self._try_leave(link, force=True)

    def _try_leave(self, link: _Link, force: bool) -> None:
        if not link.queue:
            return
        agent, earliest = link.queue[0]
        if agent.is_trip and self.transit.at_link_end(agent, link, earliest):
            return
        ready = max(earliest, self._flow_slot(link))
        if self.now < ready - _EPS:
            self._schedule_link(link, ready)
            return

        next_pos = agent.rpos + 1
        if next_pos == agent.rend:  # leaving the last link: arrive
            self._pop_head(link)
            self._arrive(agent)
            return

        nxt = self.links[self.route_links[next_pos]]
        if force or nxt.has_space():
            self._pop_head(link)
            agent.rpos = next_pos
            self._enter_link(agent, nxt)
            return

        # blocked by downstream storage: wait for space (or stuck time)
        was_blocked = link.blocked_since is not None
        if not was_blocked:
            link.blocked_since = self.now
            self._schedule_stuck(_FORCE_LEAVE, link)
        if link.waiting_on != nxt.index:
            link.waiting_on = nxt.index
            token = (True, link)
            nxt.waiters.appendleft(token) if was_blocked else nxt.waiters.append(token)

    def _pop_head(self, link: _Link) -> None:
        agent, earliest = link.queue[0]
        base = self.now
        if self.whole:  # as HermesSim: no drift from fractional headways
            nft = link.next_flow_time
            if self.now == max(earliest, ceil_step(nft)):
                base = earliest if earliest > nft else nft
        link.queue.popleft()
        if agent.is_trip:
            self.transit.link_leave(agent, link.index)
        elif self.on_leave:
            _fire(self.on_leave, LinkLeave(self.now, agent.index, link.index))
        agent.link = None
        link.occupied -= agent.pce
        link.next_flow_time = base + link.headway * agent.pce
        link.blocked_since = None
        link.waiting_on = None
        if link.queue:
            self._schedule_link(link, self._head_time(link))
        self._wake_one(link)

    def _wake_one(self, link: _Link) -> None:
        """A slot was freed on ``link``: wake the first valid waiter."""
        while link.waiters:
            is_link, obj = link.waiters.popleft()
            if obj.waiting_on != link.index:
                continue  # stale token
            obj.waiting_on = None
            if is_link:
                self._schedule_link(obj, self.now)
            else:
                self.schedule(self.now, _TRY_DEPART, obj)
            return

