"""Transit inputs and outputs shared by the Python mobsims (``QSim``, ``AgentSim``).

Both sims run the same transit semantics (pinned by ``tests/test_transit.py``) with their
own control flow (``mobsim._transit`` for the QSim, ``mobsim._agentsim_transit``
for AgentSim). What they share is here: the schedule, vehicle and ride tables
as Python lists (element access from Python is much faster on lists than on
numpy arrays), walk times, and how the results are packed for
:meth:`~mobslim.trajectories.Trajectories.build`.
"""

from mobslim.transit.walks import pt_walks


class TransitTables:
    """Python-list views of one run's transit tables."""

    def __init__(self, transit, plans, act_xy, network):
        cfg = transit.params
        self.transit = transit
        self.network_mode = cfg.vehicles == "network"
        self.door_open, self.door_close = cfg.door_open_time, cfg.door_close_time

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
        walks = pt_walks(plans, act_xy, transit)
        self.walk_before = walks.before_time.tolist()
        self.walk_after = walks.after_time.tolist()
        self.n_rides = plans.n_rides

        # trips (departures): route stop slice, vehicle type parameters, links
        v = transit.vehicles
        tt = transit.dep_type.tolist()
        cap, pce = v.capacity.tolist(), v.pce.tolist()
        acc, egr, par = v.access_time.tolist(), v.egress_time.tolist(), v.door_parallel.tolist()
        vmax = v.max_speed.tolist()
        rso, rlo = s.route_stop_offsets.tolist(), s.route_link_offsets.tolist()
        route_links = s.route_links.tolist()
        self.n_trips = s.n_departures
        self.dep_time = s.dep_time.tolist()
        dep_route = s.dep_route.tolist()
        self.trip_k0 = [rso[r] for r in dep_route]
        self.trip_k1 = [rso[r + 1] for r in dep_route]
        self.trip_capacity = [cap[t] for t in tt]
        self.trip_pce = [pce[t] for t in tt]
        self.trip_access = [acc[t] for t in tt]
        self.trip_egress = [egr[t] for t in tt]
        self.trip_parallel = [par[t] for t in tt]
        self.trip_max_speed = [vmax[t] for t in tt]
        self.trip_links = [route_links[rlo[r]:rlo[r + 1]] for r in dep_route]
        self.trip_stop_offsets = [0]
        self.trip_link_offsets = [0]
        for d, r in enumerate(dep_route):
            self.trip_stop_offsets.append(self.trip_stop_offsets[-1] + rso[r + 1] - rso[r])
            self.trip_link_offsets.append(self.trip_link_offsets[-1] + rlo[r + 1] - rlo[r])

        if self.network_mode:
            self.length = network.length.tolist()
            self.freespeed = network.freespeed.tolist()

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
