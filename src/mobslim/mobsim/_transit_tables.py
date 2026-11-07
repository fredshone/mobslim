"""Transit inputs and outputs shared by the mobsims.

Every mobsim runs the same transit semantics (pinned by
``tests/test_transit.py``) with its own control flow (``mobsim._transit`` for
the QSim, ``mobsim._agentsim_transit`` for AgentSim, the kernels of
HermesSim, ParallelSim and ShardSim). What they share is here:

* :func:`transit_times`: every time the transit model needs, on the run's
  clock. That is where the timing parameters apply, so all mobsims agree on
  them by construction. In whole seconds, **clock times** from the timetable
  (trip starts, scheduled stop arrivals and departures) round *up*, so
  nothing happens before the timetable says; **durations** (walks, door
  times, per-passenger times, a vehicle's time on a link) round like road
  times, by ``rounding``.
* :class:`TransitTables`: the same as Python lists for the Python sims
  (element access from Python is much faster on lists than on numpy
  arrays), and how their results are packed for
  :meth:`~mobslim.trajectories.Trajectories.build`.
"""

import numpy as np

from mobslim.mobsim._common import EPS, round_steps
from mobslim.transit.walks import pt_walks


def ceil_steps(x) -> np.ndarray:
    """The first whole second at or after each of ``x`` (tolerating float noise)."""
    return np.ceil(np.asarray(x, dtype=float) - EPS).astype(np.int64)


def transit_times(transit, plans, act_xy: np.ndarray, network, free_time,
                  whole: bool = False, rounding: str = "nearest") -> dict:
    """The transit model's arrays for one run (numpy; times in seconds).

    ``free_time`` is the run's free-speed time per link (already in whole
    seconds if ``whole``). Clock times are floats, whole if ``whole``:

    * per ride: ``walk_before`` (to its board stop); per leg: ``walk_after``
      (to the destination, pt legs);
    * per trip (departure): ``trip_k0``, ``trip_k1`` (its route stops),
      ``trip_start``, vehicle type data (``trip_capacity``, ``trip_pce``,
      ``trip_access``, ``trip_egress``, ``trip_parallel``);
    * per trip stop (CSR ``trip_stop_offsets``): ``trip_stop_arr``,
      ``trip_stop_dep``, the scheduled times;
    * per trip link (CSR ``trip_link_offsets``): ``trip_links`` and
      ``trip_link_time``, its time on the link at free speed (or the
      vehicle's top speed, if lower), 0 on the first link, which it is not
      driven along (it appears at its end);
    * ``door_open``, ``door_close``;
    * ``trip_network`` per trip and ``rs_network`` per route stop: whether
      the vehicles drive the network, else they keep the timetable
      (:attr:`~mobslim.transit.Transit.route_on_network`).
    """
    i8, f8 = np.int64, np.float64

    def dur(x):
        x = np.asarray(x, dtype=f8)
        return round_steps(x, rounding).astype(f8) if whole else x

    def clock(x):
        x = np.asarray(x, dtype=f8)
        return ceil_steps(x).astype(f8) if whole else x

    s, v, tc = transit.schedule, transit.vehicles, transit.params
    walks = pt_walks(plans, act_xy, transit)
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
    vtype = transit.dep_type
    vmax = np.repeat(v.max_speed[vtype].astype(f8), n_trip_links)
    slower = vmax < network.freespeed[trip_links]
    with np.errstate(divide="ignore"):
        link_time = np.where(slower, dur(network.length[trip_links] / vmax),
                             np.asarray(free_time, dtype=f8)[trip_links])
    link_time[trip_link_offsets[:-1][n_trip_links > 0]] = 0.0
    return dict(
        walk_before=dur(walks.before_time),
        walk_after=dur(walks.after_time),
        trip_k0=k0, trip_k1=k1,
        trip_start=clock(s.dep_time),
        trip_stop_offsets=trip_stop_offsets,
        trip_stop_arr=clock(dep + s.rs_arrival[trip_rs]),
        trip_stop_dep=clock(dep + s.rs_departure[trip_rs]),
        trip_link_offsets=trip_link_offsets,
        trip_links=trip_links,
        trip_link_time=link_time,
        trip_capacity=v.capacity[vtype].astype(f8),
        trip_pce=v.pce[vtype].astype(f8),
        trip_access=dur(v.access_time[vtype]),
        trip_egress=dur(v.egress_time[vtype]),
        trip_parallel=v.door_parallel[vtype].astype(np.uint8),
        trip_network=transit.trip_on_network.astype(np.uint8),
        rs_network=transit.rs_on_network.astype(np.uint8),
        door_open=float(dur(tc.door_open_time)),
        door_close=float(dur(tc.door_close_time)),
    )


class TransitTables:
    """Python-list views of one run's transit tables (see :func:`transit_times`)."""

    def __init__(self, transit, plans, act_xy, network, free_time=None, whole=False,
                 rounding="nearest"):
        self.transit = transit
        if free_time is None:
            free_time = network.free_speed_travel_times()
        t = transit_times(transit, plans, act_xy, network, free_time, whole, rounding)
        # per trip and route stop: on the network, else on the timetable; any
        # trip on the network: the trips' links follow the plans' route links
        self.trip_network = t["trip_network"].astype(bool).tolist()
        self.rs_network = t["rs_network"].astype(bool).tolist()
        self.network_mode = any(self.trip_network)
        self.door_open, self.door_close = t["door_open"], t["door_close"]

        # route stops
        s = transit.schedule
        self.rs_stop = s.rs_stop.tolist()
        self.rs_arr = s.rs_arrival.tolist()
        self.rs_dep = s.rs_departure.tolist()
        self.rs_await = s.rs_await.tolist()
        self.rs_link_pos = s.rs_link_pos.tolist()
        self.stop_blocking = s.stop_blocking.tolist()

        # passengers: the ride CSR and the walks around each ride
        self.ride_offsets = plans.ride_offsets.tolist()
        self.ride_board_rs = plans.ride_board_stop.tolist()
        self.ride_alight_rs = plans.ride_alight_stop.tolist()
        self.walk_before = t["walk_before"].tolist()
        self.walk_after = t["walk_after"].tolist()
        self.n_rides = plans.n_rides

        # trips (departures): route stop slice, scheduled times, vehicle type
        # parameters, links
        self.n_trips = s.n_departures
        self.dep_time = s.dep_time.tolist()  # as scheduled (event delays count from it)
        self.trip_start = t["trip_start"].tolist()
        self.trip_k0 = t["trip_k0"].tolist()
        self.trip_k1 = t["trip_k1"].tolist()
        self.trip_capacity = t["trip_capacity"].tolist()
        self.trip_pce = t["trip_pce"].tolist()
        self.trip_access = t["trip_access"].tolist()
        self.trip_egress = t["trip_egress"].tolist()
        self.trip_parallel = t["trip_parallel"].tolist()
        self.trip_stop_offsets = t["trip_stop_offsets"].tolist()
        self.trip_stop_arr = t["trip_stop_arr"].tolist()
        self.trip_stop_dep = t["trip_stop_dep"].tolist()
        self.trip_link_offsets = t["trip_link_offsets"].tolist()
        links, times = t["trip_links"].tolist(), t["trip_link_time"].tolist()
        lo = self.trip_link_offsets
        self.trip_links = [links[lo[d]:lo[d + 1]] for d in range(self.n_trips)]
        self.trip_link_time = [times[lo[d]:lo[d + 1]] for d in range(self.n_trips)]

    def stop_arrival(self, trip, k) -> float:
        """Scheduled arrival of ``trip`` at its route stop ``k``, on the run's clock."""
        return self.trip_stop_arr[self.trip_stop_offsets[trip.index] + k - trip.k0]

    def stop_departure(self, trip, k) -> float:
        """Scheduled departure of ``trip`` from its route stop ``k``, on the run's clock."""
        return self.trip_stop_dep[self.trip_stop_offsets[trip.index] + k - trip.k0]

    def results(self, *, ride_wait_start, ride_board, ride_alight, ride_trip, stop_arrival,
                stop_departure, transit_link_enter) -> dict:
        """Keyword arguments for ``Trajectories.build``. ``transit_link_enter``
        is ``None`` in schedule mode (trips never enter links)."""
        if transit_link_enter is None:
            transit_link_enter = [float("nan")] * self.trip_link_offsets[-1]
        return dict(
            transit=self.transit, ride_wait_start=ride_wait_start, ride_board=ride_board,
            ride_alight=ride_alight, ride_trip=ride_trip, stop_arrival=stop_arrival,
            stop_departure=stop_departure, transit_link_enter=transit_link_enter,
        )
