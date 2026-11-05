"""Transit in the QSim: scheduled trips and their passengers (TRANSIT_PLAN.md 7.1).

A :class:`_TransitRun` shares its QSim run's heap. It is created only when
the scenario has transit.

**Trips** (departures) run in one of two ways:

* ``vehicles="schedule"``: trip ``d`` is at route stop ``k`` from
  ``dep_time + arrival[k]`` to ``dep_time + departure[k]``, off the network.
  Boarding and alighting take no time; capacity still holds; passengers
  arriving during that interval (inclusive) board.
* ``vehicles="network"``: the trip is a vehicle in the queue model. It
  appears at the end of its route's first link at its departure time, takes
  ``pce`` of storage and of flow, and serves its stops when it is at the head
  of a link whose free-speed exit time has passed. A non-blocking stop moves
  it out of the link's FIFO (it still takes storage) and back in as the head
  when it is ready; a blocking stop holds the link. Dwell: doors open
  ``door_open_time`` after arrival if anyone alights or boards; alighters and
  boarders take ``egress_time`` / ``access_time`` each, one after the other
  (``serial``: alighting first) or alongside (``parallel``); doors close
  ``door_close_time`` after the last one. A passenger arriving while the
  doors are open joins the line; one arriving while they are closed but the
  vehicle awaits its departure reopens them. With ``awaitDeparture`` the
  vehicle leaves no earlier than the scheduled departure.

**Passengers** walk to the first ride's board stop, wait in a FIFO per route
stop, board the first trip of that route with room, ride to the alight stop,
walk on to the next ride or the destination. Waiting or riding at the end of
the run means stuck.
"""

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
from mobslim.mobsim._common import EPS
from mobslim.mobsim._common import fire as _fire
from mobslim.mobsim._transit_tables import TransitTables

# heap kinds (the QSim's own kinds are 0..5)
TRIP_START = 10
STOP_ARRIVE = 11  # schedule mode
STOP_DEPART = 12  # (trip, token)
DOOR_OPEN = 13  # (trip, token)
ALIGHT_DONE = 14
BOARD_DONE = 15
DOOR_CLOSE = 16
PAX_AT_STOP = 17
PAX_ARRIVE = 18

# doors (network mode)
_OPENING, _OPEN, _CLOSING, _CLOSED = 1, 2, 3, 4


class _Trip:
    __slots__ = ("index", "dep", "k0", "k1", "next_rs", "rstart", "rpos", "rend", "link",
                 "blocked_since", "waiting_on", "onboard", "load", "capacity", "pce",
                 "access", "egress", "parallel", "max_speed", "visit", "state", "token",
                 "alight_q", "alight_busy", "board_busy", "boarding", "pulled",
                 "pulled_earliest")
    is_trip = True

    def __init__(self, index, dep, k0, k1):
        self.index = index
        self.dep = dep
        self.k0, self.k1 = k0, k1
        self.next_rs = k0
        self.rstart = self.rpos = self.rend = 0
        self.link = None
        self.blocked_since = None
        self.waiting_on = None
        self.onboard: list = []
        self.load = 0
        self.visit = None  # route stop being served (network mode)
        self.state = 0
        self.token = 0
        self.alight_q: deque = deque()
        self.alight_busy = self.board_busy = False
        self.boarding = None
        self.pulled = False
        self.pulled_earliest = 0.0


class _TransitRun:
    def __init__(self, run, transit, plans, act_xy, events):
        self.run = run
        tb = self.tables = TransitTables(transit, plans, act_xy, run.scenario.network)
        self.network_mode = tb.network_mode
        self.door_open, self.door_close = tb.door_open, tb.door_close
        self.rs_stop, self.rs_arr, self.rs_dep = tb.rs_stop, tb.rs_arr, tb.rs_dep
        self.rs_await, self.rs_link_pos = tb.rs_await, tb.rs_link_pos
        self.stop_blocking = tb.stop_blocking

        # passengers
        self.ride_offsets = tb.ride_offsets
        self.ride_board_rs, self.ride_alight_rs = tb.ride_board_rs, tb.ride_alight_rs
        self.walk_before, self.walk_after = tb.walk_before, tb.walk_after
        nan = float("nan")
        n_rides = tb.n_rides
        self.t_wait = [nan] * n_rides
        self.t_board = [nan] * n_rides
        self.t_alight = [nan] * n_rides
        self.ride_trip = [-1] * n_rides
        self.waiting: dict[int, deque] = {}
        self.at_stop: dict[int, list] = {}

        # trips
        self.trip_stop_offsets = tb.trip_stop_offsets
        self.trip_link_offsets = tb.trip_link_offsets
        self.trips = []
        for d in range(tb.n_trips):
            trip = _Trip(d, tb.dep_time[d], tb.trip_k0[d], tb.trip_k1[d])
            trip.capacity, trip.pce = tb.trip_capacity[d], tb.trip_pce[d]
            trip.access, trip.egress = tb.trip_access[d], tb.trip_egress[d]
            trip.parallel, trip.max_speed = tb.trip_parallel[d], tb.trip_max_speed[d]
            if self.network_mode:  # the trip's links follow the plans' route links
                trip.rstart = trip.rpos = len(run.route_links)
                run.route_links.extend(tb.trip_links[d])
                trip.rend = len(run.route_links)
            self.trips.append(trip)
            run.schedule(trip.dep, TRIP_START, trip)
        self.n_plan_links = len(plans.route_links)
        if self.network_mode:
            run.t_enter.extend([nan] * (len(run.route_links) - self.n_plan_links))
            self.length, self.freespeed = tb.length, tb.freespeed
        self.t_stop_arr = [nan] * self.trip_stop_offsets[-1]
        self.t_stop_dep = [nan] * self.trip_stop_offsets[-1]

        h = events.handlers_for
        self.on_driver = h(TransitDriverStarts)
        self.on_v_arr, self.on_v_dep = h(VehicleArrivesAtFacility), h(VehicleDepartsAtFacility)
        self.on_wait = h(AgentWaitingForPt)
        self.on_enter, self.on_leave = h(PersonEntersVehicle), h(PersonLeavesVehicle)
        self.on_link_enter = h(TransitVehicleLinkEnter)
        self.on_link_leave = h(TransitVehicleLinkLeave)

    def handlers(self) -> dict:
        return {
            TRIP_START: self._trip_start,
            STOP_ARRIVE: self._sched_arrive,
            STOP_DEPART: self._stop_depart,
            DOOR_OPEN: self._door_open,
            ALIGHT_DONE: self._alight_done,
            BOARD_DONE: self._board_done,
            DOOR_CLOSE: self._door_close,
            PAX_AT_STOP: self._pax_at_stop,
            PAX_ARRIVE: self.run._arrive,
        }

    def results(self) -> dict:
        run = self.run
        return self.tables.results(
            ride_wait_start=self.t_wait, ride_board=self.t_board, ride_alight=self.t_alight,
            ride_trip=self.ride_trip, stop_arrival=self.t_stop_arr, stop_departure=self.t_stop_dep,
            transit_link_enter=run.t_enter[self.n_plan_links:] if self.network_mode else None,
        )

    # ------------------------------------------------------------ passengers
    def depart(self, agent) -> None:
        run = self.run
        leg = agent.leg
        r0, r1 = self.ride_offsets[leg], self.ride_offsets[leg + 1]
        if r0 == r1:
            run.schedule(run.now + self.walk_after[leg], PAX_ARRIVE, agent)
            return
        agent.ride, agent.ride_end = r0, r1
        run.schedule(run.now + self.walk_before[r0], PAX_AT_STOP, agent)

    def _pax_at_stop(self, agent) -> None:
        now = self.run.now
        r = agent.ride
        k = self.ride_board_rs[r]
        self.t_wait[r] = now
        if self.on_wait:
            _fire(self.on_wait, AgentWaitingForPt(now, agent.index, self.rs_stop[k]))
        trips = self.at_stop.get(k)
        if not self.network_mode:
            if trips:
                for trip in trips:
                    if trip.load < trip.capacity:
                        trip.load += 1
                        self._board(trip, agent)
                        return
            self.waiting.setdefault(k, deque()).append(agent)
            return
        self.waiting.setdefault(k, deque()).append(agent)
        if trips:
            for trip in trips:
                if self._serve_new_boarder(trip):
                    return

    def _board(self, trip, agent) -> None:
        now = self.run.now
        r = agent.ride
        self.t_board[r] = now
        self.ride_trip[r] = trip.index
        trip.onboard.append(agent)
        if self.on_enter:
            _fire(self.on_enter, PersonEntersVehicle(now, agent.index, trip.index))

    def _alight(self, trip, agent) -> None:
        run = self.run
        now = run.now
        r = agent.ride
        self.t_alight[r] = now
        trip.onboard.remove(agent)
        trip.load -= 1
        if self.on_leave:
            _fire(self.on_leave, PersonLeavesVehicle(now, agent.index, trip.index))
        if r + 1 < agent.ride_end:
            agent.ride = r + 1
            run.schedule(now + self.walk_before[r + 1], PAX_AT_STOP, agent)
        else:
            run.schedule(now + self.walk_after[agent.leg], PAX_ARRIVE, agent)

    def _alighters(self, trip, k) -> list:
        alight = self.ride_alight_rs
        return [p for p in trip.onboard if alight[p.ride] == k]

    # ------------------------------------------------------------ trips: common
    def _trip_start(self, trip) -> None:
        run = self.run
        if self.on_driver:
            _fire(self.on_driver, TransitDriverStarts(run.now, trip.index))
        if self.network_mode:
            run._try_depart(trip, force=False)
        else:
            run.schedule(trip.dep + self.rs_arr[trip.k0], STOP_ARRIVE, trip)

    def _record_arrival(self, trip, k) -> None:
        now = self.run.now
        self.t_stop_arr[self.trip_stop_offsets[trip.index] + k - trip.k0] = now
        if self.on_v_arr:
            _fire(self.on_v_arr, VehicleArrivesAtFacility(
                now, trip.index, self.rs_stop[k], now - (trip.dep + self.rs_arr[k])))
        self.at_stop.setdefault(k, []).append(trip)

    def _stop_depart(self, item) -> None:
        trip, token = item
        if token != trip.token:
            return
        if self.network_mode and trip.state != _CLOSED:
            return
        run = self.run
        now = run.now
        k = trip.next_rs
        self.t_stop_dep[self.trip_stop_offsets[trip.index] + k - trip.k0] = now
        if self.on_v_dep:
            _fire(self.on_v_dep, VehicleDepartsAtFacility(
                now, trip.index, self.rs_stop[k], now - (trip.dep + self.rs_dep[k])))
        self.at_stop[k].remove(trip)
        trip.next_rs = k + 1
        trip.visit = None
        trip.state = 0
        if not self.network_mode:
            if trip.next_rs < trip.k1:
                run.schedule(trip.dep + self.rs_arr[trip.next_rs], STOP_ARRIVE, trip)
            return
        if self.has_stop_here(trip):  # several stops on one link
            self._reach_stop(trip)
            return
        link = run.links[run.route_links[trip.rpos]]
        if trip.pulled:  # back from the stop into the lane, as its head
            trip.pulled = False
            link.queue.appendleft((trip, trip.pulled_earliest))
            link.blocked_since = None
            link.waiting_on = None
        run._schedule_link(link, max(now, link.next_flow_time))

    # ------------------------------------------------------------ schedule mode
    def _sched_arrive(self, trip) -> None:
        k = trip.next_rs
        self._record_arrival(trip, k)
        for p in self._alighters(trip, k):
            self._alight(trip, p)
        q = self.waiting.get(k)
        while q and trip.load < trip.capacity:
            trip.load += 1
            self._board(trip, q.popleft())
        trip.token += 1
        self.run.schedule_late(trip.dep + self.rs_dep[k], STOP_DEPART, (trip, trip.token))

    # ------------------------------------------------------------ network mode
    def has_stop_here(self, trip) -> bool:
        k = trip.next_rs
        return k < trip.k1 and self.rs_link_pos[k] == trip.rpos - trip.rstart

    def first_link_earliest(self, trip, link, now) -> float:
        """Earliest exit of a link entered by ``trip`` (its first link: at once)."""
        if trip.rpos == trip.rstart:
            return now
        if trip.max_speed < self.freespeed[link.index]:
            return now + self.length[link.index] / trip.max_speed
        return now + link.free_time

    def at_link_end(self, trip, link, earliest) -> bool:
        """``_try_leave`` hook for a trip at the head of ``link``: ``True`` if
        it is (now) serving a stop instead of leaving."""
        if trip.visit is not None:
            return True
        if not self.has_stop_here(trip):
            return False
        run = self.run
        if run.now < earliest - EPS:
            run._schedule_link(link, earliest)
            return True
        if not self.stop_blocking[self.rs_stop[trip.next_rs]]:
            link.queue.popleft()  # into the stop: off the lane, still in storage
            trip.pulled = True
            trip.pulled_earliest = earliest
            if link.queue:
                run._schedule_link(link, run._head_time(link))
        self._reach_stop(trip)
        return True

    def _reach_stop(self, trip) -> None:
        k = trip.next_rs
        trip.visit = k
        self._record_arrival(trip, k)
        trip.alight_q = deque(self._alighters(trip, k))
        if trip.alight_q or self._boarders(trip):
            self._open(trip)
        else:
            self._ready(trip)

    def _boarders(self, trip) -> bool:
        q = self.waiting.get(trip.visit)
        return bool(q) and trip.load - len(trip.alight_q) < trip.capacity

    def _open(self, trip) -> None:
        trip.state = _OPENING
        trip.token += 1
        self.run.schedule(self.run.now + self.door_open, DOOR_OPEN, (trip, trip.token))

    def _door_open(self, item) -> None:
        trip, token = item
        if token != trip.token or trip.state != _OPENING:
            return
        trip.state = _OPEN
        self._next_alight(trip)
        if trip.parallel or not trip.alight_busy:
            self._next_board(trip)
        self._maybe_close(trip)

    def _next_alight(self, trip) -> None:
        trip.alight_busy = bool(trip.alight_q)
        if trip.alight_busy:
            self.run.schedule(self.run.now + trip.egress, ALIGHT_DONE, trip)

    def _alight_done(self, trip) -> None:
        self._alight(trip, trip.alight_q.popleft())
        self._next_alight(trip)
        if not trip.parallel and not trip.alight_busy and not trip.board_busy:
            self._next_board(trip)
        self._maybe_close(trip)

    def _next_board(self, trip) -> None:
        if self._boarders(trip):
            agent = self.waiting[trip.visit].popleft()
            trip.load += 1
            trip.board_busy = True
            trip.boarding = agent
            self.run.schedule(self.run.now + trip.access, BOARD_DONE, trip)
        else:
            trip.board_busy = False

    def _board_done(self, trip) -> None:
        agent, trip.boarding = trip.boarding, None
        self._board(trip, agent)
        self._next_board(trip)
        self._maybe_close(trip)

    def _maybe_close(self, trip) -> None:
        if trip.state == _OPEN and not trip.alight_busy and not trip.board_busy:
            trip.state = _CLOSING
            self.run.schedule(self.run.now + self.door_close, DOOR_CLOSE, trip)

    def _door_close(self, trip) -> None:
        if trip.state != _CLOSING:
            return
        self._ready(trip)

    def _ready(self, trip) -> None:
        """Doors closed and nobody to serve: depart (or await the departure)."""
        if self._boarders(trip):
            self._open(trip)
            return
        trip.state = _CLOSED
        k = trip.visit
        now = self.run.now
        dep = max(now, trip.dep + self.rs_dep[k]) if self.rs_await[k] else now
        trip.token += 1
        self.run.schedule_late(dep, STOP_DEPART, (trip, trip.token))

    def _serve_new_boarder(self, trip) -> bool:
        """A passenger joined the line at ``trip``'s stop. ``True`` if this trip takes it."""
        if not self._boarders(trip):
            return False
        if trip.state == _OPEN:
            if not trip.board_busy and (trip.parallel or not trip.alight_busy):
                self._next_board(trip)
            return True
        if trip.state == _CLOSED:
            self._open(trip)
            return True
        return trip.state in (_OPENING, _CLOSING)

    # link events for trips
    def link_enter(self, trip, link) -> None:
        if self.on_link_enter:
            _fire(self.on_link_enter, TransitVehicleLinkEnter(self.run.now, trip.index, link))

    def link_leave(self, trip, link) -> None:
        if self.on_link_leave:
            _fire(self.on_link_leave, TransitVehicleLinkLeave(self.run.now, trip.index, link))
