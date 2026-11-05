"""Transit in AgentSim, the native way.

The semantics are the QSim's (pinned by
``tests/test_transit.py``); the control flow is AgentSim's: only agents are
scheduled, and each has at most one live pending transition.

**Passengers** are the plan agents. A pt leg walks to the first board stop
(``_pt_reach_stop``), waits in the FIFO line of that *route stop* with **no
heap entry**, is boarded and alighted by a trip, walks on to the next ride
or to the destination (``_pt_arrive``). Waiting is never polled: the trip
that arrives is the waker.

**Trips** (one per departure) are agents too. Their entries go through a
:class:`_Ticket`: a trip holds one live ticket, and scheduling a new step
replaces it, which cancels the old one. That is how a trip's next step moves
earlier (a passenger reopens the doors of a vehicle awaiting departure)
without a check in the main loop that every car would pay for.

* ``vehicles="schedule"``: ``_trip_at_stop`` at ``dep + arrival[k]`` alights
  and boards instantly; ``_trip_leave_stop`` runs *late* at
  ``dep + departure[k]`` (after every move and retry of that time).
* ``vehicles="network"``: the trip drives the queue model with its PCE
  (storage and flow), from the end of its first link. At the head of a link
  with an unserved stop it serves it (pulled out of the FIFO at a
  non-blocking stop, still taking storage). Dwell: doors open, alighting
  and boarding streams (serial or parallel), doors close, departure (late;
  awaiting the timetable if ``awaitDeparture``). Each stream's next
  completion keeps the heap key it would have had in the QSim, so even
  same-time completions happen in the QSim's order. A trip that leaves a
  non-blocking stop rejoins the lane *as head*; a car it displaces is
  handled by :meth:`_TransitMixin._displaced`.
* Blocked trips poll like cars (``retry_interval``, stuck time, the
  departing line of their first link).
"""

import heapq
import math
from collections import deque

from mobslim.events import (
    AgentWaitingForPt,
    PersonEntersVehicle,
    PersonLeavesVehicle,
    TransitDriverStarts,
    TransitVehicleLinkEnter,
    TransitVehicleLinkLeave,
    VehicleArrivesAtFacility,
    VehicleDepartsAtFacility,
)
from mobslim.mobsim._common import EPS, fire
from mobslim.mobsim._transit_tables import TransitTables

# doors (network mode)
_OPENING, _OPEN, _CLOSING, _CLOSED = 1, 2, 3, 4
_LATE = 2  # heap priority of late entries (scheduled moves 0, retries 1)


class _Ticket:
    """A trip's heap entry. Only the trip's current ticket runs its step."""

    __slots__ = ("trip",)

    def __init__(self, trip):
        self.trip = trip

    @staticmethod
    def transition(run, ticket) -> None:
        trip = ticket.trip
        if trip.ticket is ticket:
            trip.ticket = None
            trip.ticket_due = None
            trip.step(run, trip)


class _Trip:
    __slots__ = ("index", "dep", "k0", "k1", "next_rs", "rstart", "rpos", "rend", "link",
                 "earliest", "blocked_since", "retry_at", "onboard", "load", "capacity",
                 "pce", "access", "egress", "parallel", "max_speed", "visit", "state",
                 "alight_q", "alight_busy", "board_busy", "boarding", "alight_due",
                 "board_due", "pulled", "pulled_earliest", "ticket", "ticket_due", "step")
    is_trip = True

    def __init__(self, index, dep, k0, k1):
        self.index = index
        self.dep = dep
        self.k0, self.k1 = k0, k1
        self.next_rs = k0
        self.rstart = self.rpos = self.rend = 0
        self.link = None
        self.earliest = 0.0
        self.blocked_since = None
        self.retry_at = math.inf
        self.onboard: list = []
        self.load = 0
        self.visit = None  # route stop being served (network mode)
        self.state = 0
        self.alight_q: deque = deque()
        self.alight_busy = self.board_busy = False
        self.boarding = None
        self.alight_due = self.board_due = None  # (time, heap key) of the next completion
        self.pulled = False
        self.pulled_earliest = 0.0
        self.ticket = None
        self.ticket_due = None
        self.step = None


class _TransitMixin:
    """Transit transitions of ``_AgentSimRun`` (which inherits this)."""

    def _init_transit(self, transit, plans, act_xy, events) -> None:
        tb = self.tt = TransitTables(transit, plans, act_xy, self.scenario.network)
        self.pt_network = tb.network_mode
        nan = math.nan
        self.t_wait = [nan] * tb.n_rides
        self.t_board = [nan] * tb.n_rides
        self.t_alight = [nan] * tb.n_rides
        self.ride_trip = [-1] * tb.n_rides
        self.pt_waiting: dict[int, deque] = {}  # route stop -> passengers, FIFO
        self.pt_at_stop: dict[int, list] = {}  # route stop -> trips there
        self.t_stop_arr = [nan] * tb.trip_stop_offsets[-1]
        self.t_stop_dep = [nan] * tb.trip_stop_offsets[-1]

        h = events.handlers_for
        self.on_driver = h(TransitDriverStarts)
        self.on_v_arr, self.on_v_dep = h(VehicleArrivesAtFacility), h(VehicleDepartsAtFacility)
        self.on_wait = h(AgentWaitingForPt)
        self.on_pt_enter, self.on_pt_leave = h(PersonEntersVehicle), h(PersonLeavesVehicle)
        self.on_t_link_enter = h(TransitVehicleLinkEnter)
        self.on_t_link_leave = h(TransitVehicleLinkLeave)

        # trips, scheduled before any activity ends (as in the QSim)
        links = self.links
        route = self.route
        self.trips = []
        for d in range(tb.n_trips):
            trip = _Trip(d, tb.dep_time[d], tb.trip_k0[d], tb.trip_k1[d])
            trip.capacity, trip.pce = tb.trip_capacity[d], tb.trip_pce[d]
            trip.access, trip.egress = tb.trip_access[d], tb.trip_egress[d]
            trip.parallel, trip.max_speed = tb.trip_parallel[d], tb.trip_max_speed[d]
            if self.pt_network:  # the trip's links follow the plans' route links
                trip.rstart = trip.rpos = len(route)
                route.extend(links[i] for i in tb.trip_links[d])
                trip.rend = len(route)
            self.trips.append(trip)
            self._push_trip(trip.dep, trip, _TransitMixin._trip_start)
        if self.pt_network:
            self.t_enter.extend([nan] * (len(route) - self.n_plan_links))

    def _transit_results(self) -> dict:
        return self.tt.results(
            ride_wait_start=self.t_wait, ride_board=self.t_board, ride_alight=self.t_alight,
            ride_trip=self.ride_trip, stop_arrival=self.t_stop_arr,
            stop_departure=self.t_stop_dep,
            transit_link_enter=self.t_enter[self.n_plan_links:] if self.pt_network else None,
        )

    # -- scheduling trips ---------------------------------------------------
    def _push_trip(self, time, trip, step, prio=0, key=None) -> None:
        """Make ``step`` the trip's next transition, at ``time`` (cancels its
        previous one). ``key`` orders entries of equal time and priority."""
        if key is None:
            key = next(self.seq)
        trip.step = step
        ticket = trip.ticket = _Ticket(trip)
        trip.ticket_due = (time, key)
        heapq.heappush(self.heap, (time, prio, key, ticket))

    def _trip_retry(self, trip, step) -> None:
        """A trip blocked by storage: retry ``step`` on its retry grid (as cars)."""
        time = self._retry_time(trip)
        if time is None:  # parked in a permanent gridlock
            trip.ticket = None
            return
        self.n_retries += 1
        self._push_trip(time, trip, step, 1, (trip.blocked_since, self.n_persons + trip.index))

    def _trip_head(self, trip, link) -> None:
        """``trip`` became the head of ``link``: schedule its exit. A trip with a
        stop on this link serves it without waiting for a flow slot."""
        if self._has_stop_here(trip):
            time = trip.earliest if trip.earliest > self.now else self.now
        else:
            nft = link.next_flow_time
            time = trip.earliest if trip.earliest > nft else nft
        self._push_trip(time, trip, _TransitMixin._trip_leave_link)

    def _schedule_successor(self, link) -> None:
        """Schedule the new head of ``link`` (after a trip left or was pulled)."""
        queue = link.queue
        if not queue:
            return
        head = queue[0]
        if head.is_trip:
            self._trip_head(head, link)
            return
        if head.transition is _TransitMixin._displaced:
            if head.has_stale:  # its stale entry comes first and reschedules it
                return
            head.transition = self._leave_transition
        nft = link.next_flow_time
        earliest = head.earliest
        heapq.heappush(self.heap, (earliest if earliest > nft else nft, 0, next(self.seq), head))

    def _displaced(self, agent) -> None:
        """Transition of a car whose head place a trip took (it rejoined from a
        stop). Runs the car's pending entry, which is stale: if the car is head
        again, schedule its exit properly; otherwise the trip leaving will."""
        agent.has_stale = False
        link = agent.link
        if link.queue[0] is not agent:
            return  # still behind the trip
        agent.transition = leave = self._leave_transition
        nft = link.next_flow_time
        time = agent.earliest if agent.earliest > nft else nft
        if self.now >= time:
            leave(self, agent)
        else:
            heapq.heappush(self.heap, (time, 0, next(self.seq), agent))

    # -- passengers ---------------------------------------------------------
    def _pt_depart(self, agent) -> None:
        tb = self.tt
        leg = agent.leg
        r0, r1 = tb.ride_offsets[leg], tb.ride_offsets[leg + 1]
        if r0 == r1:  # walk-only pt leg
            self.schedule(self.now + tb.walk_after[leg], agent, _TransitMixin._pt_arrive)
            return
        agent.ride, agent.ride_end = r0, r1
        self.schedule(self.now + tb.walk_before[r0], agent, _TransitMixin._pt_reach_stop)

    def _pt_arrive(self, agent) -> None:
        self.last_progress = self.now
        self._arrive(agent)

    def _pt_reach_stop(self, agent) -> None:
        """Walked to the board stop: board a trip that is there, or wait
        (without a heap entry) until one takes this passenger."""
        now = self.now
        self.last_progress = now
        r = agent.ride
        k = self.tt.ride_board_rs[r]
        self.t_wait[r] = now
        if self.on_wait:
            fire(self.on_wait, AgentWaitingForPt(now, agent.index, self.tt.rs_stop[k]))
        trips = self.pt_at_stop.get(k)
        if not self.pt_network:
            if trips:
                for trip in trips:
                    if trip.load < trip.capacity:
                        trip.load += 1
                        self._pt_board(trip, agent)
                        return
            self.pt_waiting.setdefault(k, deque()).append(agent)
            return
        self.pt_waiting.setdefault(k, deque()).append(agent)
        if trips:
            for trip in trips:
                if self._serve_new_boarder(trip):
                    return

    def _pt_board(self, trip, agent) -> None:
        now = self.now
        r = agent.ride
        self.t_board[r] = now
        self.ride_trip[r] = trip.index
        trip.onboard.append(agent)
        if self.on_pt_enter:
            fire(self.on_pt_enter, PersonEntersVehicle(now, agent.index, trip.index))

    def _pt_alight(self, trip, agent) -> None:
        now = self.now
        self.last_progress = now
        tb = self.tt
        r = agent.ride
        self.t_alight[r] = now
        trip.onboard.remove(agent)
        trip.load -= 1
        if self.on_pt_leave:
            fire(self.on_pt_leave, PersonLeavesVehicle(now, agent.index, trip.index))
        if r + 1 < agent.ride_end:  # transfer: walk to the next board stop
            agent.ride = r + 1
            self.schedule(now + tb.walk_before[r + 1], agent, _TransitMixin._pt_reach_stop)
        else:
            self.schedule(now + tb.walk_after[agent.leg], agent, _TransitMixin._pt_arrive)

    def _alighters(self, trip, k) -> list:
        alight = self.tt.ride_alight_rs
        return [p for p in trip.onboard if alight[p.ride] == k]

    # -- trips: stops (both modes) ------------------------------------------
    def _trip_start(self, trip) -> None:
        if self.on_driver:
            fire(self.on_driver, TransitDriverStarts(self.now, trip.index))
        if not self.pt_network:
            self._push_trip(trip.dep + self.tt.rs_arr[trip.k0], trip, _TransitMixin._trip_at_stop)
            return
        waiting = self.route[trip.rpos].departing
        if waiting:  # others are already waiting to depart here: queue behind them
            trip.blocked_since = self.now
            waiting.append(trip)
            return
        self._trip_enter_route(trip)

    def _record_arrival(self, trip, k) -> None:
        now = self.now
        self.last_progress = now
        tb = self.tt
        self.t_stop_arr[tb.trip_stop_offsets[trip.index] + k - trip.k0] = now
        if self.on_v_arr:
            fire(self.on_v_arr, VehicleArrivesAtFacility(
                now, trip.index, tb.rs_stop[k], now - (trip.dep + tb.rs_arr[k])))
        self.pt_at_stop.setdefault(k, []).append(trip)

    def _trip_at_stop(self, trip) -> None:
        """Schedule mode: at the stop on time; alight, then board, instantly."""
        k = trip.next_rs
        self._record_arrival(trip, k)
        for p in self._alighters(trip, k):
            self._pt_alight(trip, p)
        q = self.pt_waiting.get(k)
        while q and trip.load < trip.capacity:
            trip.load += 1
            self._pt_board(trip, q.popleft())
        self._push_trip(trip.dep + self.tt.rs_dep[k], trip, _TransitMixin._trip_leave_stop, _LATE)

    def _trip_leave_stop(self, trip) -> None:
        if self.pt_network and trip.state != _CLOSED:
            return
        now = self.now
        self.last_progress = now
        tb = self.tt
        k = trip.next_rs
        self.t_stop_dep[tb.trip_stop_offsets[trip.index] + k - trip.k0] = now
        if self.on_v_dep:
            fire(self.on_v_dep, VehicleDepartsAtFacility(
                now, trip.index, tb.rs_stop[k], now - (trip.dep + tb.rs_dep[k])))
        self.pt_at_stop[k].remove(trip)
        trip.next_rs = k + 1
        trip.visit = None
        trip.state = 0
        if not self.pt_network:
            if trip.next_rs < trip.k1:
                self._push_trip(trip.dep + tb.rs_arr[trip.next_rs], trip,
                                _TransitMixin._trip_at_stop)
            return
        if self._has_stop_here(trip):  # several stops on one link
            self._trip_reach_stop(trip)
            return
        link = trip.link
        if trip.pulled:
            self._trip_rejoin(trip, link)
        nft = link.next_flow_time
        self._push_trip(now if now > nft else nft, trip, _TransitMixin._trip_leave_link)

    # -- trips: on links (network mode) -------------------------------------
    def _has_stop_here(self, trip) -> bool:
        k = trip.next_rs
        return k < trip.k1 and self.tt.rs_link_pos[k] == trip.rpos - trip.rstart

    def _trip_enter_route(self, trip) -> None:
        """Appear at the end of the first link (as ``_enter_route`` for cars)."""
        link = self.route[trip.rpos]
        waiting = link.departing
        if link.occupied < link.storage or self.is_stuck(trip):
            trip.blocked_since = None
            if waiting:  # this trip is the front of the line
                waiting.popleft()
                if waiting:
                    self._wake_departer(waiting[0])
            self._trip_enter_link(trip, link)
        else:
            if not waiting:
                link.departing = deque((trip,))
            self._trip_retry(trip, _TransitMixin._trip_enter_route)

    def _trip_enter_link(self, trip, link) -> None:
        now = self.now
        self.last_progress = now
        self.t_enter[trip.rpos] = now
        if self.on_t_link_enter:
            fire(self.on_t_link_enter, TransitVehicleLinkEnter(now, trip.index, link.index))
        trip.link = link
        i = link.index
        if trip.rpos == trip.rstart:  # the first link is not traversed
            trip.earliest = now
        elif trip.max_speed < self.tt.freespeed[i]:
            trip.earliest = now + self.tt.length[i] / trip.max_speed
        else:
            trip.earliest = now + link.free_time
        link.occupied += trip.pce
        link.queue.append(trip)
        if len(link.queue) == 1:
            self._trip_head(trip, link)

    def _trip_leave_link(self, trip) -> None:
        """At the head of its link: serve a stop here, or leave the link."""
        if trip.visit is not None:
            return
        link = trip.link
        now = self.now
        if self._has_stop_here(trip):
            if now < trip.earliest - EPS:
                self._push_trip(trip.earliest, trip, _TransitMixin._trip_leave_link)
                return
            if not self.tt.stop_blocking[self.tt.rs_stop[trip.next_rs]]:
                link.queue.popleft()  # into the stop: off the lane, still in storage
                link.n_pulled += 1
                trip.pulled = True
                trip.pulled_earliest = trip.earliest
                self._schedule_successor(link)
            self._trip_reach_stop(trip)
            return
        nft = link.next_flow_time
        ready = trip.earliest if trip.earliest > nft else nft
        if now < ready - EPS:
            self._push_trip(ready, trip, _TransitMixin._trip_leave_link)
            return
        next_pos = trip.rpos + 1
        if next_pos == trip.rend:  # leaves the network at the end of its last link
            self._trip_pop_head(trip, link)
            return
        nxt = self.route[next_pos]
        if nxt.occupied >= nxt.storage and not self.is_stuck(trip):
            self._trip_retry(trip, _TransitMixin._trip_leave_link)
            return
        trip.blocked_since = None
        self._trip_pop_head(trip, link)
        trip.rpos = next_pos
        self._trip_enter_link(trip, nxt)

    def _trip_pop_head(self, trip, link) -> None:
        now = self.now
        self.last_progress = now
        link.queue.popleft()
        if self.on_t_link_leave:
            fire(self.on_t_link_leave, TransitVehicleLinkLeave(now, trip.index, link.index))
        trip.link = None
        link.occupied -= trip.pce
        link.next_flow_time = now + link.headway * trip.pce
        self._schedule_successor(link)

    def _trip_rejoin(self, trip, link) -> None:
        """Back from a non-blocking stop into the lane, as its head."""
        trip.pulled = False
        link.n_pulled -= 1
        queue = link.queue
        if queue:
            head = queue[0]
            if head.is_trip:
                head.ticket = None  # rescheduled when this trip has left
            elif head.transition is not _TransitMixin._displaced:
                # a car has exactly one pending entry unless parked in a gridlock
                head.has_stale = not (head.blocked_since is not None and head.retry_at == math.inf)
                head.transition = _TransitMixin._displaced
            head.blocked_since = None  # as the QSim: the lane's blocked state restarts
        trip.earliest = trip.pulled_earliest
        queue.appendleft(trip)

    # -- trips: dwell (network mode) ----------------------------------------
    def _trip_reach_stop(self, trip) -> None:
        k = trip.next_rs
        trip.visit = k
        self._record_arrival(trip, k)
        trip.alight_q = deque(self._alighters(trip, k))
        if trip.alight_q or self._boarders(trip):
            self._open(trip)
        else:
            self._ready(trip)

    def _boarders(self, trip) -> bool:
        q = self.pt_waiting.get(trip.visit)
        return bool(q) and trip.load - len(trip.alight_q) < trip.capacity

    def _open(self, trip) -> None:
        trip.state = _OPENING
        self._push_trip(self.now + self.tt.door_open, trip, _TransitMixin._trip_door_open)

    def _trip_door_open(self, trip) -> None:
        trip.state = _OPEN
        self._next_alight(trip)
        if trip.parallel or not trip.alight_busy:
            self._next_board(trip)
        self._maybe_close(trip)
        self._dwell_ticket(trip)

    def _next_alight(self, trip) -> None:
        trip.alight_busy = bool(trip.alight_q)
        trip.alight_due = (self.now + trip.egress, next(self.seq)) if trip.alight_busy else None

    def _next_board(self, trip) -> None:
        if self._boarders(trip):
            trip.boarding = self.pt_waiting[trip.visit].popleft()
            trip.load += 1
            trip.board_busy = True
            trip.board_due = (self.now + trip.access, next(self.seq))
        else:
            trip.board_busy = False
            trip.board_due = None

    def _maybe_close(self, trip) -> None:
        if trip.state == _OPEN and not trip.alight_busy and not trip.board_busy:
            trip.state = _CLOSING
            self._push_trip(self.now + self.tt.door_close, trip, _TransitMixin._trip_door_close)

    def _dwell_ticket(self, trip) -> None:
        """Doors open: the trip's ticket is the earlier stream completion, with
        the heap key the QSim would have given it."""
        if trip.state != _OPEN:
            return
        a, b = trip.alight_due, trip.board_due
        due = a if b is None or (a is not None and a < b) else b
        if due is not None and due != trip.ticket_due:
            self._push_trip(due[0], trip, _TransitMixin._trip_door_step, 0, due[1])

    def _trip_door_step(self, trip) -> None:
        """The next alighting or boarding completes."""
        a, b = trip.alight_due, trip.board_due
        if a is not None and (b is None or a < b):
            trip.alight_due = None
            self._pt_alight(trip, trip.alight_q.popleft())
            self._next_alight(trip)
            if not trip.parallel and not trip.alight_busy and not trip.board_busy:
                self._next_board(trip)
        else:
            trip.board_due = None
            agent, trip.boarding = trip.boarding, None
            self._pt_board(trip, agent)
            self._next_board(trip)
        self._maybe_close(trip)
        self._dwell_ticket(trip)

    def _trip_door_close(self, trip) -> None:
        if trip.state == _CLOSING:
            self._ready(trip)

    def _ready(self, trip) -> None:
        """Doors closed and nobody to serve: depart (or await the departure)."""
        if self._boarders(trip):
            self._open(trip)
            return
        trip.state = _CLOSED
        k = trip.visit
        now = self.now
        tb = self.tt
        dep = max(now, trip.dep + tb.rs_dep[k]) if tb.rs_await[k] else now
        self._push_trip(dep, trip, _TransitMixin._trip_leave_stop, _LATE)

    def _serve_new_boarder(self, trip) -> bool:
        """A passenger joined the line at ``trip``'s stop. ``True`` if this trip takes it."""
        if not self._boarders(trip):
            return False
        if trip.state == _OPEN:
            if not trip.board_busy and (trip.parallel or not trip.alight_busy):
                self._next_board(trip)
                self._dwell_ticket(trip)
            return True
        if trip.state == _CLOSED:  # awaiting departure: reopen (cancels the departure)
            self._open(trip)
            return True
        return trip.state in (_OPENING, _CLOSING)
