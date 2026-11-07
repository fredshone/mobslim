"""An agent-driven queue simulation: the alternative to :class:`QSim`.

Same queue model as the QSim (and MATSim), built from the same link
parameters (``mobsim._common.link_parameters``):

* **Free-speed time**: a vehicle cannot leave a link before
  ``length / freespeed`` seconds after entering it.
* **Flow capacity**: per link, ``1 / (capacity * flow_capacity_factor)``
  seconds between leaving vehicles, fractional (no rounding).
* **Storage capacity**: ``length * lanes / effective_cell_size *
  storage_capacity_factor`` vehicles (at least the link's own flow, and ≥ 1);
  full links block upstream vehicles (spillback).
* **FIFO**: only the vehicle at the head of a link's queue may leave.
* **Stuck time**: a vehicle blocked for ``stuck_time`` seconds is pushed on
  regardless of storage.

What differs is *who is scheduled*. The QSim schedules links (each link
wakes its head vehicle, a freed link wakes one waiter). Here only **agents**
are on the heap. Each agent always has exactly one pending *transition*
(end its activity and depart; leave its link and enter the next one or
arrive; finish a teleport), and executes it when it comes up:

* Waits whose end is **known** are scheduled exactly: free-speed time, flow
  headway, activity ends, teleports. When a head vehicle leaves, the next
  vehicle in that link's queue is scheduled at its earliest exit, so flow
  capacity is identical to the QSim's even for headways below the retry
  interval.
* Waits whose end is **not known** are **polled**: a vehicle blocked by a
  full downstream link (or departing onto a full link) retries every
  ``retry_interval`` seconds until there is space or ``stuck_time`` has
  passed. Blocked vehicles therefore react up to ``retry_interval`` late,
  like a time-stepped simulation (MATSim steps at 1 s), and the order in
  which blocked link heads get freed space follows their retry times rather
  than a waiting queue.
* Agents **departing** onto a full link wait in a FIFO line per link (like
  MATSim's waiting list): only the front polls, and when it enters, the next
  in line tries at once. Stuck time counts from when an agent reaches the
  front, so a full link takes at most one forced departure per
  ``stuck_time``, as in MATSim.

With ``scenario.transit``, pt legs ride scheduled trips, which are agents
too (see :mod:`mobslim.mobsim._agentsim_transit`). Storage and flow count
PCE (cars 1, trips their type's).

Without spillback (no link ever full), the two simulations produce the same
trajectories. Because each transition is an agent's own decision point,
this is also the natural place to hook in within-day (en-route) choices.

With ``stuck_time=inf`` a gridlock cannot resolve. It is detected (every
pending transition is a retry and nothing has moved for a full retry
interval) and the run ends, reporting the blocked agents as stuck, as the
QSim does.
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
from mobslim.mobsim._agentsim_transit import _TransitMixin
from mobslim.mobsim._common import (
    EPS,
    ceil_step,
    check_timing,
    fire,
    link_parameters,
    none_if_nan,
    planned_end,
    round_steps,
    validate,
)
from mobslim.planbatch import PlanBatch
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories


@dataclass(frozen=True)
class AgentSimParams:
    flow_capacity_factor: float = 1.0
    storage_capacity_factor: float = 1.0
    stuck_time: float = 10.0
    effective_cell_size: float = 7.5  # meters per vehicle (PCU)
    retry_interval: float = 1.0  # seconds between retries of a blocked move
    end_time: Optional[float] = None  # None: run until every plan is done
    validate_routes: bool = True
    # how a blocked agent picks its next retry: "poll" tries at every retry
    # tick; "chain" skips ticks at which its blocking chain cannot have
    # moved (same results, far fewer retries under heavy congestion)
    blocked: str = "poll"
    # timing (see mobslim.mobsim._common.check_timing). Whole seconds give
    # HermesSim's rules; retry_interval must then be whole seconds too
    continuous_time: bool = True
    junction_delays: bool = False
    rounding: str = "nearest"

    def __post_init__(self):
        if not self.retry_interval > 0 or not math.isfinite(self.retry_interval):
            raise ValueError("retry_interval must be a positive, finite number of seconds.")
        check_timing("AgentSim", self.continuous_time, self.junction_delays, self.rounding,
                     continuous=True, whole=True)
        if not self.continuous_time and self.retry_interval != int(self.retry_interval):
            raise ValueError("with continuous_time=False, retry_interval must be whole seconds.")
        if self.blocked not in ("poll", "chain"):
            raise ValueError(f"blocked must be 'poll' or 'chain', not {self.blocked!r}.")


class AgentSim:
    """Agent-driven queue mobsim. Stateless between runs; safe to reuse."""

    def __init__(self, params: AgentSimParams = AgentSimParams()):
        self.params = params

    def run(self, scenario: Scenario, events: EventsManager) -> Trajectories:
        plans = scenario.population.selected_batch()
        return self.run_batch(scenario, plans, events)

    def run_batch(self, scenario: Scenario, plans: PlanBatch, events: EventsManager) -> Trajectories:
        """Run already-encoded plans (``plans`` row ``i`` = person index ``i``)."""
        if plans.n_plans != len(scenario.population):
            raise ValueError("PlanBatch must have one plan per person.")
        if self.params.validate_routes:
            validate(scenario, plans)
        return _AgentSimRun(scenario, plans, events, self.params).run()


def _next_tick(since: float, interval: float, after: float) -> float:
    """The first retry tick ``since + k * interval`` (k >= 1) after ``after``."""
    k = max(1, int((after - since) // interval))
    while k > 1 and since + (k - 1) * interval > after:
        k -= 1
    while since + k * interval <= after:
        k += 1
    return since + k * interval


def _tick_at_or_after(since: float, interval: float, time: float) -> float:
    """The first retry tick ``since + k * interval`` (k >= 1) at or after ``time``."""
    k = max(1, math.ceil((time - since) / interval))
    while k > 1 and since + (k - 1) * interval >= time:
        k -= 1
    while since + k * interval < time:
        k += 1
    return since + k * interval


_MAX_CHAIN = 256  # longest blocking chain walked before settling for a looser bound


class _Agent:
    __slots__ = ("index", "act", "last_act", "leg", "rpos", "rend", "link", "earliest",
                 "transition", "blocked_since", "retry_at", "distance", "done",
                 "ride", "ride_end", "has_stale")
    is_trip = False  # transit vehicles are ``_agentsim_transit._Trip``

    def __init__(self, index: int, first_act: int, last_act: int, transition):
        self.index = index
        self.act = first_act  # global index of the current (or last) activity
        self.last_act = last_act
        self.leg: Optional[int] = None  # global index of the current leg
        self.rpos = 0  # global position of the current route link
        self.rend = 0  # end of the current route (exclusive)
        self.link: Optional[_Link] = None  # link the agent is currently on
        self.earliest = 0.0  # earliest exit from the current link (free-speed time)
        self.transition = transition  # _AgentSimRun method (unbound: no reference cycle) run next
        self.blocked_since: Optional[float] = None  # waiting for storage
        self.retry_at = math.inf  # time of the pending retry (inf: parked in a gridlock)
        self.distance = 0.0  # teleport distance
        self.done = False
        self.ride = 0  # current pt ride (global index)
        self.ride_end = 0
        self.has_stale = False  # displaced by a trip with its exit still pending


class _Link:
    __slots__ = ("index", "free_time", "headway", "storage", "queue", "next_flow_time",
                 "departing", "occupied", "n_pulled")

    def __init__(self, index: int, free_time: float, headway: float, storage: float):
        self.index = index
        self.free_time = free_time
        self.headway = headway
        self.storage = storage
        self.queue: deque = deque()  # agents, FIFO (exit time in ``agent.earliest``)
        self.next_flow_time = 0.0
        # agents waiting to depart onto this link, FIFO; only the front polls
        # (created when first needed: most links never have a line)
        self.departing: Optional[deque] = None
        # PCE on the link: cars 1, trips their type's PCE, also while at a stop
        # off the FIFO
        self.occupied = 0.0
        self.n_pulled = 0  # trips pulled into a stop on this link (off the FIFO)


class _AgentSimRun(_TransitMixin):
    """State for a single run.

    The link transition (``_leave_link``) is the hot path (one call per link
    traversal), so it is written out inline with locals rather than split
    into helpers.

    Heap entries are ``(time, is_retry, key, agent)``. At equal times,
    scheduled moves (``is_retry=0``, ``key`` = push order) come before
    retries (``is_retry=1``, ``key=(blocked_since, agent index)``), and
    retries before late transit entries (``is_retry=2``: stop departures).
    Transit trips are on the heap through tickets (``_agentsim_transit``). Retry
    times are on each agent's own grid, ``blocked_since + k * retry_interval``,
    capped at its stuck deadline. So where and in what order retries happen
    does not depend on when they were pushed, and a failed retry has no
    effect beyond scheduling the next one: skipping retries that are certain
    to fail cannot change the outcome.
    """

    def __init__(self, scenario: Scenario, plans: PlanBatch, events: EventsManager,
                 cfg: AgentSimParams):
        self.scenario = scenario
        self.plans = plans
        self.cfg = cfg
        self.now = 0.0
        self.heap: list = []
        self.seq = itertools.count()
        self.n_retries = 0  # heap entries that are retries
        self.chain = cfg.blocked == "chain"
        self.last_progress = 0.0  # time of the last successful transition

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
        # whole seconds: HermesSim's rules (see QSim)
        self.whole = not cfg.continuous_time
        if self.whole:
            free_time = round_steps(free_time, cfg.rounding).astype(float).tolist()
        self.free_time = free_time
        self.junction = 1.0 if cfg.junction_delays else 0.0
        links = self.links = [_Link(i, free_time[i], headway[i], storage[i])
                              for i in range(net.n_links)]

        # plan arrays as Python lists (fast element access in a Python loop)
        self.act_type = [plans.act_types[c] for c in plans.act_type.tolist()]
        self.act_node = plans.act_node.tolist()
        self.act_xy = plans.resolved_act_xy(net).tolist()
        self.act_end_time = none_if_nan(plans.act_end_time)
        self.act_duration = none_if_nan(plans.act_duration)
        self.leg_mode = [plans.modes[c] for c in plans.leg_mode.tolist()]
        self.leg_network = plans.network_legs(scenario.modes.network_modes).tolist()
        self.leg_fixed_time = none_if_nan(plans.leg_travel_time)
        self.route_offsets = plans.route_offsets.tolist()
        # route positions resolved to link objects up front (no index lookup per step)
        self.route = [links[i] for i in plans.route_links.tolist()]
        self.n_plan_links = len(self.route)
        self.leg_transit = plans.mode_legs(self.modes.transit_modes).tolist()
        self._leave_transition = _AgentSimRun._leave_link

        nan = math.nan
        self.t_act_start = [nan] * plans.n_acts
        self.t_act_end = [nan] * plans.n_acts
        self.t_departure = [nan] * plans.n_legs
        self.t_arrival = [nan] * plans.n_legs
        self.t_teleport = [0.0] * plans.n_legs
        self.t_enter = [nan] * len(self.route)
        self.t_stuck = [False] * plans.n_plans

        offsets = plans.act_offsets.tolist()
        end_activity = _AgentSimRun._end_activity
        self.agents = [_Agent(i, offsets[i], offsets[i + 1] - 1, end_activity)
                       for i in range(plans.n_plans)]
        self.n_persons = plans.n_plans
        self.transit = scenario.transit
        if self.transit is not None:
            self._init_transit(self.transit, plans, plans.resolved_act_xy(net), events)

    # -- scheduling ---------------------------------------------------------
    def schedule(self, time: float, agent: _Agent, transition) -> None:
        agent.transition = transition
        heapq.heappush(self.heap, (time, 0, next(self.seq), agent))

    def retry(self, agent: _Agent) -> None:
        """The agent's move is blocked by storage: try again at its next
        retry tick (or its stuck deadline, if sooner)."""
        time = self._retry_time(agent)
        if time is not None:
            self.n_retries += 1
            heapq.heappush(self.heap, (time, 1, (agent.blocked_since, agent.index), agent))

    def _retry_time(self, agent) -> Optional[float]:
        """Time of a blocked vehicle's next retry (sets ``retry_at``); ``None``
        if it is parked in a permanent gridlock."""
        now = self.now
        since = agent.blocked_since
        if since is None:
            since = agent.blocked_since = now
        interval = self.cfg.retry_interval
        deadline = self._deadline(since)
        bound = self._chain_bound(agent, deadline) if self.chain else now
        if bound > now:
            if bound == math.inf:  # permanent gridlock: nothing can ever free the way
                agent.retry_at = math.inf
                return None
            time = _tick_at_or_after(since, interval, bound)
        else:
            # usually now is the previous tick: try k + 1 first
            k = int((now - since) / interval + 0.5) + 1
            time = since + k * interval
            if time <= now or since + (k - 1) * interval > now:
                time = _next_tick(since, interval, now)
        if deadline < time:
            time = deadline
        agent.retry_at = time
        return time

    def _chain_bound(self, agent: _Agent, bound: float) -> float:
        """Earliest time the link ``agent`` is blocked by can have space.

        A full link only frees when its head leaves. An unblocked head leaves
        no earlier than its scheduled exit; a blocked head only when the link
        it wants frees (recursively) or at its stuck deadline. Following the
        chain of blocked heads to one that is not blocked gives a lower
        bound; every retry before it would fail. A chain that loops back on
        itself (a gridlock) frees only at a stuck deadline: never, with
        ``stuck_time=inf``. ``bound`` starts as the agent's own deadline.
        """
        route = self.route
        link = route[agent.rpos] if agent.link is None else route[agent.rpos + 1]
        prev = agent
        seen = set()
        for _ in range(_MAX_CHAIN):
            if link.occupied < link.storage:
                # space now: the blocked head behind it moves at its next retry
                return min(bound, prev.retry_at)
            if link in seen:  # cycle: only a stuck deadline can break it
                return bound
            seen.add(link)
            if link.n_pulled or not link.queue:
                # a trip pulled into a stop may finish its dwell, rejoin and
                # leave at any moment: no bound beyond the blocked head's
                # next retry (none at all for the agent's own link)
                return self.now if prev is agent else min(bound, prev.retry_at)
            head = link.queue[0]
            if head.blocked_since is None:  # waits on free speed / flow, or arrives
                nft = self._slot(link.next_flow_time)
                exit_time = head.earliest if head.earliest > nft else nft
                return bound if bound < exit_time else exit_time
            deadline = self._deadline(head.blocked_since)
            if deadline < bound:
                bound = deadline
            link = route[head.rpos + 1]
            prev = head
        return min(bound, prev.retry_at)  # chain too long: the last head's retry still bounds it

    def _deadline(self, since: float) -> float:
        """When a vehicle blocked since ``since`` is pushed on."""
        t = since + self.cfg.stuck_time
        return float(ceil_step(t)) if self.whole and math.isfinite(t) else t

    def _slot(self, nft: float) -> float:
        """When a head may leave as far as flow capacity goes."""
        return float(ceil_step(nft)) if self.whole else nft

    def _next_flow_time(self, link, agent, pce=1.0) -> float:
        """The next flow slot after ``agent`` (of ``pce``) leaves ``link`` now
        (whole seconds: as HermesSim, so fractional headways do not drift)."""
        now, nft = self.now, link.next_flow_time
        base = now
        if self.whole and now == max(agent.earliest, ceil_step(nft)):
            base = agent.earliest if agent.earliest > nft else nft
        return base + link.headway * pce

    def is_stuck(self, agent: _Agent) -> bool:
        return (agent.blocked_since is not None
                and self.now >= agent.blocked_since + self.cfg.stuck_time - EPS)

    # -- main loop ----------------------------------------------------------
    def run(self) -> Trajectories:
        for agent in self.agents:
            if agent.act == agent.last_act:
                agent.done = True
                continue
            self.schedule(self._planned_end(agent.act, 0.0), agent, _AgentSimRun._end_activity)

        heap = self.heap
        heappop = heapq.heappop
        end_time = self.cfg.end_time
        horizon = math.inf if end_time is None else end_time
        retry_interval = self.cfg.retry_interval
        # a finite stuck time always resolves a gridlock; only detect it otherwise
        detect_gridlock = not math.isfinite(self.cfg.stuck_time)
        while heap:
            if heap[0][0] > horizon:
                break
            time, is_retry, _, agent = heappop(heap)
            if is_retry == 1:
                self.n_retries -= 1
                # every pending transition is a retry and all of them have
                # failed since the last move: gridlock
                if (detect_gridlock and len(heap) == self.n_retries
                        and time > self.last_progress + retry_interval + EPS):
                    break
            self.now = time
            agent.transition(self, agent)

        if end_time is not None:
            self.now = max(self.now, float(end_time))
        self._abort_unfinished()
        return Trajectories.build(
            self.plans, self.scenario.network,
            act_start=self.t_act_start, act_end=self.t_act_end,
            leg_departure=self.t_departure, leg_arrival=self.t_arrival,
            link_enter=self.t_enter[:self.n_plan_links], stuck=self.t_stuck,
            teleport_distance=self.t_teleport,
            network_modes=self.scenario.modes.network_modes,
            **({} if self.transit is None else self._transit_results()),
        )

    def _abort_unfinished(self) -> None:
        for agent in self.agents:
            if agent.done:
                continue
            self.t_stuck[agent.index] = True
            if self.on_stuck:
                mode = None if agent.leg is None else self.leg_mode[agent.leg]
                link = None if agent.link is None else agent.link.index
                fire(self.on_stuck, PersonStuck(self.now, agent.index, mode, link))
            agent.done = True

    def _planned_end(self, act: int, start: float) -> Optional[float]:
        t = planned_end(self.act_end_time[act], self.act_duration[act], start)
        return float(ceil_step(t)) if self.whole and t is not None else t

    # -- transitions: activities -------------------------------------------
    def _end_activity(self, agent: _Agent) -> None:
        self.last_progress = self.now
        a = agent.act
        self.t_act_end[a] = self.now
        if self.on_act_end:
            fire(self.on_act_end, ActivityEnd(self.now, agent.index, self.act_type[a], self.act_node[a]))
        leg = agent.leg = a - agent.index
        origin = self.act_node[a]
        self.t_departure[leg] = self.now
        if self.on_departure:
            fire(self.on_departure, PersonDeparture(self.now, agent.index, self.leg_mode[leg], origin))
        if self.leg_network[leg]:
            r0, r1 = self.route_offsets[leg], self.route_offsets[leg + 1]
            if r0 == r1:
                self._arrive(agent)
                return
            agent.rpos, agent.rend = r0, r1
            agent.transition = _AgentSimRun._enter_route
            waiting = self.route[r0].departing
            if waiting:  # others are already waiting to depart here: queue behind them
                agent.blocked_since = self.now
                waiting.append(agent)
                return
            self._enter_route(agent)
        elif self.leg_transit[leg]:
            self._pt_depart(agent)
        else:
            tt, agent.distance = self.modes.teleport(
                self.leg_mode[leg], self.act_xy[a], self.act_xy[a + 1], self.leg_fixed_time[leg],
                agent.index,
            )
            if self.whole:
                tt = float(round_steps(tt, self.cfg.rounding))
            self.schedule(self.now + tt, agent, _AgentSimRun._end_teleport)

    def _start_activity(self, agent: _Agent) -> None:
        a = agent.act
        self.t_act_start[a] = self.now
        if self.on_act_start:
            fire(self.on_act_start, ActivityStart(self.now, agent.index, self.act_type[a], self.act_node[a]))
        if a == agent.last_act:
            agent.done = True
            return
        self.schedule(self._planned_end(a, self.now), agent, _AgentSimRun._end_activity)

    # -- transitions: legs --------------------------------------------------
    def _end_teleport(self, agent: _Agent) -> None:
        self.last_progress = self.now
        self.t_teleport[agent.leg] = agent.distance
        if self.on_teleport:
            fire(self.on_teleport, Teleportation(
                self.now, agent.index, self.leg_mode[agent.leg], agent.distance))
        self._arrive(agent)

    def _arrive(self, agent: _Agent) -> None:
        leg = agent.leg
        self.t_arrival[leg] = self.now
        if self.on_arrival:
            fire(self.on_arrival, PersonArrival(
                self.now, agent.index, self.leg_mode[leg], self.act_node[agent.act + 1]))
        agent.leg = None
        agent.act += 1
        if self.junction:  # the activity starts one second later
            self.schedule(self.now + self.junction, agent, _AgentSimRun._start_activity)
        else:
            self._start_activity(agent)

    def _enter_route(self, agent: _Agent) -> None:
        """Depart onto the first link of the route. Blocked departers wait
        in the link's FIFO ``departing`` line, and only its front polls."""
        link = self.route[agent.rpos]
        waiting = link.departing
        if link.occupied < link.storage or self.is_stuck(agent):
            agent.blocked_since = None
            if waiting:  # this agent is the front of the line
                waiting.popleft()
                if waiting:  # the next in line tries at once
                    self._wake_departer(waiting[0])
            self._enter_link(agent, link)
        else:
            if not waiting:  # first to be blocked here: start the line
                link.departing = deque((agent,))
            self.retry(agent)

    def _wake_departer(self, agent) -> None:
        """``agent`` reached the front of a departing line: it tries at once,
        and its stuck time counts from now (MATSim's buffer: at most one
        forced departure per ``stuck_time``)."""
        agent.blocked_since = self.now
        if agent.is_trip:
            self._push_trip(self.now, agent, _TransitMixin._trip_enter_route)
        else:
            self.schedule(self.now, agent, _AgentSimRun._enter_route)

    # -- transitions: links -------------------------------------------------
    def _enter_link(self, agent: _Agent, link: _Link) -> None:
        """Enter the first link of a route (``_leave_link`` inlines the rest)."""
        now = self.now
        self.last_progress = now
        self.t_enter[agent.rpos] = now
        if self.on_enter:
            fire(self.on_enter, LinkEnter(now, agent.index, link.index))
        agent.link = link
        agent.transition = _AgentSimRun._leave_link
        earliest = agent.earliest = now + link.free_time
        if agent.rpos + 1 < agent.rend:  # crosses a junction after it
            earliest = agent.earliest = earliest + self.junction
        link.occupied += 1.0
        queue = link.queue
        queue.append(agent)
        if len(queue) == 1:  # head of the queue: schedule its exit
            nft = self._slot(link.next_flow_time)
            heapq.heappush(self.heap, (earliest if earliest > nft else nft,
                                       0, next(self.seq), agent))
        # otherwise it is scheduled when the vehicle ahead leaves

    def _leave_link(self, agent: _Agent) -> None:
        """Leave the current link (only the queue's head is ever scheduled):
        enter the next one, or arrive."""
        link = agent.link
        next_pos = agent.rpos + 1
        if next_pos == agent.rend:  # last link of the route: arrive
            self._pop_head(link)
            self._arrive(agent)
            return
        nxt = self.route[next_pos]
        if nxt.occupied >= nxt.storage and not self.is_stuck(agent):
            self.retry(agent)  # spillback: poll for space downstream
            return
        agent.blocked_since = None
        now = self.now
        self.last_progress = now
        heap, seq = self.heap, self.seq
        # leave the current link; its next vehicle becomes the head
        queue = link.queue
        queue.popleft()
        if self.on_leave:
            fire(self.on_leave, LinkLeave(now, agent.index, link.index))
        link.occupied -= 1.0
        nft = link.next_flow_time = self._next_flow_time(link, agent)
        if queue:
            successor = queue[0]
            if successor.is_trip:
                self._trip_head(successor, link)
            else:
                earliest = successor.earliest
                nft = self._slot(nft)
                heapq.heappush(heap, (earliest if earliest > nft else nft, 0, next(seq), successor))
        # enter the next link (agent.transition stays _leave_link)
        agent.rpos = next_pos
        self.t_enter[next_pos] = now
        if self.on_enter:
            fire(self.on_enter, LinkEnter(now, agent.index, nxt.index))
        agent.link = nxt
        earliest = agent.earliest = now + nxt.free_time
        if next_pos + 1 < agent.rend:  # crosses a junction after it
            earliest = agent.earliest = earliest + self.junction
        nxt.occupied += 1.0
        nqueue = nxt.queue
        nqueue.append(agent)
        if len(nqueue) == 1:
            nft = self._slot(nxt.next_flow_time)
            heapq.heappush(heap, (earliest if earliest > nft else nft, 0, next(seq), agent))

    def _pop_head(self, link: _Link) -> None:
        queue = link.queue
        agent = queue.popleft()
        now = self.now
        self.last_progress = now
        if self.on_leave:
            fire(self.on_leave, LinkLeave(now, agent.index, link.index))
        agent.link = None
        link.occupied -= 1.0
        nft = link.next_flow_time = self._next_flow_time(link, agent)
        if queue:  # the next vehicle becomes the head
            successor = queue[0]
            if successor.is_trip:
                self._trip_head(successor, link)
            else:
                earliest = successor.earliest
                nft = self._slot(nft)
                heapq.heappush(self.heap, (earliest if earliest > nft else nft,
                                           0, next(self.seq), successor))
