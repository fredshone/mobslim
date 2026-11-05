"""Behavioural spec for transit simulation, on small synthetic scenarios.

The expected numbers follow from the transit rules (this module is their
specification; ``docs/design.md`` section 5 summarises them). Every scenario uses the same corridor so the arithmetic
stays easy to check by hand:

    a ──100 m── 0 ──1000 m── 1 ──1000 m── 2 ──1000 m── 3 ──1000 m── 4
              S0            S1            S2            S3            S4

* Links allow ``car`` and ``bus``, run at 10 m/s (100 s per 1000 m link) and
  carry 3600 veh/h (1 s headway).
* Stop ``Si`` sits at node ``i`` on the link that ends there.
* Route ``L/r`` runs ``a-0 … 3-4`` and serves S0..S4 at offsets 0, 120, 240,
  360, 480 s (arrival = departure, ``awaitDeparture`` on). Departures are
  07:00, 07:10 and 07:20 (``T0``, ``T0 + 600``, ``T0 + 1200``).
* Walking is 1 m/s with beeline factor 1, and activities are at stop nodes,
  so access and egress walks take 0 s unless a test says otherwise.
* Door and per-passenger times are 0 unless a test says otherwise.

With ``vehicles="schedule"`` the vehicles keep to the timetable exactly. With
``vehicles="network"`` they drive the queue network: a vehicle appears at the
end of its route's first link at its departure time, reaches the end of each
later link after 100 s, and waits there for its scheduled departure.

The module is skipped until ``mobslim.transit`` provides the planned API.
"""

import math

import numpy as np
import pytest

from mobslim import (
    ActivityParams,
    ModeParams,
    Plan,
    Population,
    Scenario,
    ScoringParams,
)
from mobslim.events import EventLog, EventsManager, PersonStuck
from mobslim.network import NetworkBuilder
from mobslim.planbatch import PlanBatch
from mobslim.routing import TeleportedModeParams
from mobslim.scoring import CharyparNagelScorer
from tests import conftest
from tests.conftest import commuters, make_mobsim, routed

try:
    from mobslim.events import (
        AgentWaitingForPt,
        PersonEntersVehicle,
        PersonLeavesVehicle,
        VehicleArrivesAtFacility,
        VehicleDepartsAtFacility,
    )
    from mobslim.transit import (
        RaptorParams,
        RaptorRouter,
        TransitBuilder,
        TransitParams,
        TransitPassengerRoute,
    )
except ImportError:  # pragma: no cover - until the transit API exists
    pytest.skip("transit simulation is not implemented yet",
                allow_module_level=True)

H = 3600.0
T0 = 7 * H  # 25200
WALK = TeleportedModeParams(speed=1.0, beeline_distance_factor=1.0)

# mobsims that implement transit (all of them)
TRANSIT_MOBSIMS = (["qsim", "agentsim", "agentsim_chain", "hermes_py"]
                   + (["hermes"] if conftest.rust_available() else []))


@pytest.fixture(params=TRANSIT_MOBSIMS, autouse=True)
def mobsim_kind(request, monkeypatch):
    monkeypatch.setattr(conftest, "DEFAULT_MOBSIM", request.param)
    return request.param


def tol():
    """Whole-second mobsims may differ by one step on fractional times."""
    return conftest.time_step() + 1e-9


# ---------------------------------------------------------------- scenarios


def corridor_network():
    b = NetworkBuilder()
    b.add_node("a", -100.0, 0.0)
    for i in range(5):
        b.add_node(i, 1000.0 * i, 0.0)
    b.add_link("a-0", "a", 0, 100.0, 10.0, 3600.0, modes=("car", "bus"))
    for i in range(4):
        b.add_link(f"{i}-{i + 1}", i, i + 1, 1000.0, 10.0, 3600.0, modes=("car", "bus"))
    return b.build()


STOP_LINKS = {"S0": "a-0", "S1": "0-1", "S2": "1-2", "S3": "2-3", "S4": "3-4"}


def corridor_transit(net, *, vehicles="schedule", capacity=10, access=0.0, egress=0.0,
                     doors="serial", door_open=0.0, door_close=0.0, await_departure=True,
                     departures=(T0, T0 + 600, T0 + 1200), extra_routes=None, blocking=False,
                     stops=None):
    b = TransitBuilder(net)
    b.vehicle_type("bus", seats=capacity, standing=0, access_time=access, egress_time=egress,
                   doors=doors, pce=1.0)
    for i in range(5):
        b.stop(f"S{i}", 1000.0 * i, 0.0, link=STOP_LINKS[f"S{i}"], blocking=blocking)
    route_stops = stops or [(f"S{i}", 120.0 * i, 120.0 * i, await_departure) for i in range(5)]
    b.route("L", "r", mode="bus", links=["a-0", "0-1", "1-2", "2-3", "3-4"], stops=route_stops)
    b.departures("L", "r", times=list(departures), vehicle_type="bus")
    if extra_routes:
        extra_routes(b)
    return b.build(TransitParams(vehicles=vehicles, walk=WALK, door_open_time=door_open,
                                 door_close_time=door_close))


def node(net, i):
    return net.node_index(i)


def pt_scenario(population, **transit):
    net = corridor_network()
    pop = population(net) if callable(population) else population
    return Scenario(net, pop, teleported_modes={"walk": WALK},
                    transit=corridor_transit(net, **transit))


def passengers(n, origin, destination, dep=25000.0):
    return lambda net: commuters(n, node(net, origin), node(net, destination), dep=dep, mode="pt")


def run(scenario, handlers=(), **config):
    events = EventsManager(list(handlers))
    events.reset(0)
    tr = make_mobsim(**config).run(scenario, events)
    events.finish()
    return tr


def rides_of(tr, person):
    """Ride indices of ``person``'s (single) pt leg."""
    b = tr.plans
    leg = int(b.leg_offsets[person])
    return range(int(b.ride_offsets[leg]), int(b.ride_offsets[leg + 1]))


# ---------------------------------------------------------------- data model


def test_default_vehicle_mode_is_schedule():
    assert TransitParams().vehicles == "schedule"


def test_access_walk_starts_at_activity_coordinates():
    """The activity is at node 0 for the network, but at (0, 50): 50 m to S0."""
    def population(net):
        plan = (Plan.builder().act("home", node(net, 0), end_time=25000.0, xy=(0.0, 50.0))
                .leg("pt").act("work", node(net, 3)).build())
        return Population.from_plans({0: plan})
    scenario = routed(pt_scenario(population))
    batch = PlanBatch.from_plans(scenario.population.selected_plans())
    assert batch.act_xy.tolist() == [[0.0, 50.0], [3000.0, 0.0]]  # node xy if none given
    tr = run(scenario)
    assert tr.ride_wait_start[0] == pytest.approx(25050.0)
    assert tr.leg_arrival[0] == pytest.approx(T0 + 360)  # no egress walk: work has no xy


def test_teleport_from_activity_coordinates():
    """Teleported legs (here walk) also run between activity coordinates:
    (0, 50) -> node 1 at (1000, 0), 1 m/s."""
    def population(net):
        plan = (Plan.builder().act("home", node(net, 0), end_time=25000.0, xy=(0.0, 50.0))
                .leg("walk").act("work", node(net, 1)).build())
        return Population.from_plans({0: plan})
    tr = run(routed(pt_scenario(population)))
    assert tr.leg_distance[0] == pytest.approx(math.hypot(1000.0, 50.0))
    assert tr.leg_arrival[0] == pytest.approx(25000.0 + math.hypot(1000.0, 50.0), abs=tol())


def test_modes_pt_is_a_transit_mode_even_if_links_allow_it():
    net = corridor_network()
    scenario = Scenario(net, commuters(1, 1, 2, mode="pt"), transit=corridor_transit(net))
    assert scenario.modes.is_transit_mode("pt")
    assert not scenario.modes.is_network_mode("pt")
    assert scenario.modes.is_network_mode("car")


def test_schedule_tables():
    net = corridor_network()
    s = corridor_transit(net).schedule
    assert (s.n_stops, s.n_routes, s.n_route_stops, s.n_departures) == (5, 1, 5, 3)
    assert s.rs_stop.tolist() == [s.stop_ids.index(f"S{i}") for i in range(5)]
    assert s.rs_arrival.tolist() == [0, 120, 240, 360, 480]
    assert s.rs_link_pos.tolist() == [0, 1, 2, 3, 4]  # stop i is on the route's i-th link
    assert s.dep_time.tolist() == [T0, T0 + 600, T0 + 1200]
    assert s.route_stop("L", "r", "S3") == 3
    # stop times are derived, not stored: departure + offset
    assert s.stop_times(0).tolist() == [[T0 + 120 * k for k in range(5)],
                                        [T0 + 600 + 120 * k for k in range(5)],
                                        [T0 + 1200 + 120 * k for k in range(5)]]
    for name in ("rs_stop", "rs_arrival", "dep_time", "route_links"):
        assert not getattr(s, name).flags.writeable


def test_builder_rejects_stop_not_on_route():
    net = corridor_network()
    b = TransitBuilder(net)
    b.vehicle_type("bus", seats=10)
    b.stop("S4", 4000.0, 0.0, link="3-4")
    b.route("L", "r", mode="bus", links=["a-0", "0-1"], stops=[("S4", 0.0, 0.0)])
    with pytest.raises(ValueError, match="S4"):
        b.build()


def test_plan_batch_encodes_rides_as_csr():
    scenario = routed(pt_scenario(passengers(2, 0, 3)))
    batch = PlanBatch.from_plans(scenario.population.selected_plans())
    assert batch.ride_offsets.tolist() == [0, 1, 2]
    s = scenario.transit.schedule
    assert batch.ride_board_stop.tolist() == [s.route_stop("L", "r", "S0")] * 2
    assert batch.ride_alight_stop.tolist() == [s.route_stop("L", "r", "S3")] * 2


def test_pt_route_is_packed_and_hashable():
    a = TransitPassengerRoute([(0, 3), (5, 7)], travel_time=100.0, distance=10.0)
    assert a.rides == ((0, 3), (5, 7))
    assert a.n_rides == 2
    assert len(a.data) == 16  # two int32 pairs
    assert a == TransitPassengerRoute([(0, 3), (5, 7)], 100.0, 10.0) and hash(a) == hash(TransitPassengerRoute(a.rides, 100.0, 10.0))


# ---------------------------------------------------------------- routing


def transfer_routes(b):
    """A: S0→S2 at 07:00; B: S2→S4 at 07:05 and 07:15; C: slow direct S0→S4 at 07:01."""
    b.route("A", "a", mode="bus", links=["a-0", "0-1", "1-2"],
            stops=[("S0", 0, 0), ("S1", 120, 120), ("S2", 240, 240)])
    b.departures("A", "a", times=[T0], vehicle_type="bus")
    b.route("B", "b", mode="bus", links=["1-2", "2-3", "3-4"],
            stops=[("S2", 0, 0), ("S3", 120, 120), ("S4", 240, 240)])
    b.departures("B", "b", times=[T0 + 300, T0 + 900], vehicle_type="bus")
    b.route("C", "c", mode="bus", links=["a-0", "0-1", "1-2", "2-3", "3-4"],
            stops=[("S0", 0, 0), ("S4", 1200, 1200)])
    b.departures("C", "c", times=[T0 + 60], vehicle_type="bus")


def transfer_scenario(**transit):
    return pt_scenario(passengers(1, 0, 4), departures=(), extra_routes=transfer_routes, **transit)


def test_raptor_takes_the_transfer_when_it_is_faster():
    scenario = transfer_scenario()
    router = RaptorRouter(scenario.network, scenario.transit)
    route = router.route(node(scenario.network, 0), node(scenario.network, 4), 25000.0)
    assert scenario.transit.schedule.describe_rides(route.rides) == [
        ("A", "a", "S0", "S2"), ("B", "b", "S2", "S4")]
    assert route.travel_time == pytest.approx(T0 + 540 - 25000.0)  # arrives 07:09


def test_raptor_line_switch_penalty_prefers_the_direct_line():
    scenario = transfer_scenario()
    router = RaptorRouter(scenario.network, scenario.transit, RaptorParams(line_switch=-1000.0))
    route = router.route(node(scenario.network, 0), node(scenario.network, 4), 25000.0)
    assert scenario.transit.schedule.describe_rides(route.rides) == [("C", "c", "S0", "S4")]


def test_raptor_min_transfer_time_can_force_the_later_connection():
    scenario = transfer_scenario()
    # 07:04 at S2 + 120 s misses B at 07:05; B at 07:15 still beats C (07:21)
    # (with the default line-switch penalty the later connection would lose to C)
    router = RaptorRouter(scenario.network, scenario.transit,
                          RaptorParams(min_transfer_time=120.0, line_switch=0.0))
    route = router.route(node(scenario.network, 0), node(scenario.network, 4), 25000.0)
    assert [r[0] for r in scenario.transit.schedule.describe_rides(route.rides)] == ["A", "B"]
    assert route.travel_time == pytest.approx(T0 + 1140 - 25000.0)


def test_raptor_walks_when_transit_cannot_help():
    scenario = pt_scenario(passengers(1, 0, "a"))
    router = RaptorRouter(scenario.network, scenario.transit)
    route = router.route(node(scenario.network, 0), node(scenario.network, "a"), 25000.0)
    assert route.n_rides == 0
    assert route.travel_time == pytest.approx(100.0)  # 100 m at 1 m/s


def test_raptor_walks_after_the_last_departure():
    scenario = pt_scenario(passengers(1, 0, 3, dep=T0 + 1300))
    route = RaptorRouter(scenario.network, scenario.transit).route(
        node(scenario.network, 0), node(scenario.network, 3), T0 + 1300)
    assert route.n_rides == 0
    assert route.travel_time == pytest.approx(3000.0)


def test_route_many_equals_route():
    scenario = transfer_scenario()
    router = RaptorRouter(scenario.network, scenario.transit)
    net = scenario.network
    queries = [(node(net, o), node(net, d), t) for o, d, t in
               [(0, 4, 25000.0), (0, 3, 25000.0), (1, 4, 25100.0), (0, 4, 25900.0), (2, 4, 25000.0)]]
    many = router.route_many(*map(list, zip(*queries)))
    assert many == [router.route(*q) for q in queries]


def test_prepare_routes_pt_legs():
    scenario = routed(pt_scenario(passengers(1, 0, 3)))
    leg = scenario.population[0].selected_plan.legs[0]
    assert isinstance(leg.route, TransitPassengerRoute)
    assert scenario.transit.schedule.describe_rides(leg.route.rides) == [("L", "r", "S0", "S3")]


# ---------------------------------------------------------------- mobsim


@pytest.mark.parametrize("vehicles, alight", [("schedule", T0 + 360), ("network", T0 + 340)])
def test_single_ride(vehicles, alight):
    tr = run(routed(pt_scenario(passengers(1, 0, 3), vehicles=vehicles)))
    (r,) = rides_of(tr, 0)
    assert tr.leg_departure[0] == 25000.0
    assert tr.ride_wait_start[r] == 25000.0  # walk to S0 takes 0 s
    assert tr.ride_board[r] == pytest.approx(T0, abs=tol())
    assert tr.ride_alight[r] == pytest.approx(alight, abs=tol())
    assert tr.ride_trip[r] == 0  # the 07:00 departure
    assert tr.leg_arrival[0] == pytest.approx(alight, abs=tol())
    assert tr.leg_distance[0] == pytest.approx(3000.0)  # in-vehicle distance along the route
    assert not tr.stuck[0]


def test_network_vehicle_follows_queue_model_and_awaits_departure():
    log = EventLog()
    run(routed(pt_scenario(passengers(1, 0, 3), vehicles="network")), [log])
    arr = [(e.stop, e.time) for e in log.of_type(VehicleArrivesAtFacility) if e.trip == 0]
    dep = [(e.stop, e.time) for e in log.of_type(VehicleDepartsAtFacility) if e.trip == 0]
    stops = list(range(5))  # events carry stop indices; S0..S4 were added in order
    # appears at the end of a-0 at 07:00, 100 s per link, waits for each offset
    assert [s for s, _ in arr] == stops
    assert [t for _, t in arr] == pytest.approx([T0, T0 + 100, T0 + 220, T0 + 340, T0 + 460], abs=tol())
    assert [t for _, t in dep] == pytest.approx([T0, T0 + 120, T0 + 240, T0 + 360, T0 + 480], abs=tol())
    early = [e.delay for e in log.of_type(VehicleArrivesAtFacility) if e.trip == 0][1]
    assert early == pytest.approx(-20.0, abs=tol())


def test_passenger_event_sequence():
    log = EventLog()
    run(routed(pt_scenario(passengers(1, 0, 3))), [log])
    mine = [type(e).__name__ for e in log.events if getattr(e, "person", None) == 0]
    assert mine == ["ActivityEnd", "PersonDeparture", "AgentWaitingForPt", "PersonEntersVehicle",
                    "PersonLeavesVehicle", "PersonArrival", "ActivityStart"]
    (enter,) = log.of_type(PersonEntersVehicle)
    (wait,) = log.of_type(AgentWaitingForPt)
    assert enter.trip == 0 and wait.stop == 0  # stop index of S0
    assert log.of_type(PersonLeavesVehicle)[0].time == pytest.approx(T0 + 360)


@pytest.mark.parametrize("vehicles, ride_time", [("schedule", 360.0), ("network", 340.0)])
def test_capacity_leaves_passengers_behind(vehicles, ride_time):
    tr = run(routed(pt_scenario(passengers(5, 0, 3), vehicles=vehicles, capacity=2)))
    boards = [tr.ride_board[r] for p in range(5) for r in rides_of(tr, p)]
    # first come, first served; ties by person index
    expected = [T0, T0, T0 + 600, T0 + 600, T0 + 1200]
    assert boards == pytest.approx(expected, abs=tol())
    assert tr.ride_trip.tolist() == [0, 0, 1, 1, 2]
    assert tr.leg_arrival.tolist() == pytest.approx([b + ride_time for b in expected], abs=tol())
    assert tr.denied_boardings().tolist() == [0, 0, 1, 1, 2]  # derived, not recorded
    assert tr.pt_load().max() == 2


def test_dwell_time_serial_doors():
    tr = run(routed(pt_scenario(passengers(3, 0, 3), vehicles="network", access=2.0, egress=3.0,
                                door_open=1.0, door_close=1.0)), [log := EventLog()])
    # S0: doors open 07:00:01, boardings every 2 s, doors closed at 07:00:08 -> 8 s late
    assert tr.ride_board.tolist() == pytest.approx([T0 + 3, T0 + 5, T0 + 7], abs=tol())
    dep_s0 = next(e for e in log.of_type(VehicleDepartsAtFacility) if e.stop == 0)
    assert dep_s0.time == pytest.approx(T0 + 8, abs=tol()) and dep_s0.delay == pytest.approx(8.0, abs=tol())
    # the delay is absorbed by awaiting S1's departure; S3 reached at T0 + 340
    assert tr.ride_alight.tolist() == pytest.approx([T0 + 344, T0 + 347, T0 + 350], abs=tol())


@pytest.mark.parametrize("doors, board_time", [("serial", T0 + 226), ("parallel", T0 + 223)])
def test_dwell_time_door_operation(doors, board_time):
    # p0 rides S0 -> S2; p1 boards at S2. The vehicle reaches S2 at T0 + 220.
    def population(net):
        return Population.from_plans({
            0: Plan.builder().act("home", node(net, 0), end_time=25000.0).leg("pt")
                .act("work", node(net, 2)).build(),
            1: Plan.builder().act("home", node(net, 2), end_time=25000.0).leg("pt")
                .act("work", node(net, 3)).build(),
        })
    tr = run(routed(pt_scenario(population, vehicles="network", access=2.0, egress=3.0,
                                door_open=1.0, door_close=1.0, doors=doors)))
    (r0,), (r1,) = rides_of(tr, 0), rides_of(tr, 1)
    assert tr.ride_alight[r0] == pytest.approx(T0 + 224, abs=tol())  # open + 3 s
    assert tr.ride_board[r1] == pytest.approx(board_time, abs=tol())  # after alighting, or alongside


@pytest.mark.parametrize("vehicles, board, alight", [
    ("schedule", T0 + 120, T0 + 360),  # the timetable says 07:02 at S1: caught
    ("network", T0 + 700, T0 + 900),   # the bus ran early (07:01:40) and left: next one
])
def test_without_await_departure_early_vehicles_leave(vehicles, board, alight):
    tr = run(routed(pt_scenario(passengers(1, 1, 3, dep=T0 + 110), vehicles=vehicles,
                                await_departure=False)))
    (r,) = rides_of(tr, 0)
    assert tr.ride_board[r] == pytest.approx(board, abs=tol())
    assert tr.ride_alight[r] == pytest.approx(alight, abs=tol())


@pytest.mark.parametrize("vehicles", ["schedule", "network"])
def test_missed_last_departure_is_stuck(vehicles):
    net = corridor_network()
    transit = corridor_transit(net, vehicles=vehicles)
    s = transit.schedule
    route = TransitPassengerRoute([(s.route_stop("L", "r", "S0"), s.route_stop("L", "r", "S3"))], 360.0, 3000.0)
    plan = (Plan.builder().act("home", node(net, 0), end_time=T0 + 1300).leg("pt", route=route)
            .act("work", node(net, 3)).build())
    log = EventLog()
    tr = run(Scenario(net, Population.from_plans({0: plan}), teleported_modes={"walk": WALK},
                      transit=transit), [log])
    assert tr.stuck[0]
    assert math.isnan(tr.leg_arrival[0]) and math.isnan(tr.ride_board[0])
    assert tr.ride_wait_start[0] == T0 + 1300
    assert [e.mode for e in log.of_type(PersonStuck)] == ["pt"]


@pytest.mark.parametrize("vehicles", ["schedule", "network"])
def test_walk_only_pt_leg(vehicles):
    tr = run(routed(pt_scenario(passengers(1, 0, "a"), vehicles=vehicles)))
    assert len(rides_of(tr, 0)) == 0
    assert tr.leg_arrival[0] == pytest.approx(25100.0)
    assert tr.leg_distance[0] == pytest.approx(100.0)


def test_transfer_is_simulated():
    tr = run(routed(transfer_scenario(vehicles="schedule")))
    r1, r2 = rides_of(tr, 0)
    assert tr.ride_alight[r1] == pytest.approx(T0 + 240)
    assert tr.ride_wait_start[r2] == pytest.approx(T0 + 240)  # same stop, 0 s walk
    assert tr.ride_board[r2] == pytest.approx(T0 + 300)
    assert tr.leg_arrival[0] == pytest.approx(T0 + 540)


def car_behind_bus(blocking):
    """A bus waits 200 s at S1 (offset 300, reached after 100 s). A car enters
    0-1 50 s behind it."""
    net = corridor_network()
    plan = (Plan.builder().act("home", node(net, 0), end_time=T0 + 50).leg("car")
            .act("work", node(net, 2)).build())
    transit = corridor_transit(net, vehicles="network", departures=(T0,), blocking=blocking,
                               stops=[("S0", 0, 0, True), ("S1", 300, 300, True)])
    return routed(Scenario(net, Population.from_plans({0: plan}), teleported_modes={"walk": WALK},
                           transit=transit))


@pytest.mark.parametrize("blocking, car_arrival", [
    (False, T0 + 250),        # the bus pulls into the stop; the car passes at free speed
    (True, T0 + 300 + 1 + 100),  # the car waits for the bus, then one headway
])
def test_stop_blocking_in_mixed_traffic(blocking, car_arrival):
    tr = run(car_behind_bus(blocking))
    assert tr.leg_arrival[0] == pytest.approx(car_arrival, abs=tol())
    v = tr.transit
    # vehicle link entries per (trip, route link); the first link is not traversed
    trip0 = v.link_enter[v.trip_link_offsets[0]:v.trip_link_offsets[1]]
    assert trip0.tolist() == pytest.approx([T0, T0, T0 + 300, T0 + 400, T0 + 500], abs=tol())


def test_vehicle_queued_past_its_earliest_exit_serves_the_stop_now():
    """Cars queue on 0-1 (low flow) ahead of the bus, so the bus reaches the
    head long after its free-speed exit: it serves S1 then, and time never
    runs backwards."""
    net = corridor_network()
    plans = {i: Plan.builder().act("home", node(net, 0), end_time=T0 - 30 + i).leg("car")
             .act("work", node(net, 2)).build() for i in range(30)}
    plans[30] = (Plan.builder().act("home", node(net, 1), end_time=T0).leg("pt")
                 .act("work", node(net, 3)).build())
    transit = corridor_transit(net, vehicles="network", departures=(T0,),
                               stops=[("S0", 0, 0, False), ("S1", 120, 120, False),
                                      ("S3", 360, 360, False)])
    scenario = routed(Scenario(net, Population.from_plans(plans), teleported_modes={"walk": WALK},
                               transit=transit))
    log = EventLog()
    tr = run(scenario, [log], flow_capacity_factor=0.05)  # one car every 20 s
    times = [e.time for e in log.events]
    assert times == sorted(times)
    (r,) = rides_of(tr, 30)
    arrive_s1 = next(e.time for e in log.of_type(VehicleArrivesAtFacility) if e.stop == 1)
    assert arrive_s1 > T0 + 100 + 100  # far behind its free-speed arrival
    assert tr.ride_board[r] == pytest.approx(arrive_s1, abs=tol())


def test_trajectories_match_events():
    """Ride arrays rebuilt from events equal the mobsim's own."""
    from mobslim.trajectories import TrajectoryCollector

    scenario = routed(pt_scenario(passengers(5, 0, 3), vehicles="network", capacity=2))
    batch = PlanBatch.from_plans(scenario.population.selected_plans())
    collector = TrajectoryCollector(scenario.network, batch, scenario.transit)
    tr = run(scenario, [collector])
    rebuilt = collector.trajectories()
    for name in ("ride_wait_start", "ride_board", "ride_alight", "ride_trip", "leg_arrival"):
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(tr, name))


def test_trajectory_collector_without_transit_ignores_vehicle_events():
    from mobslim.trajectories import TrajectoryCollector

    scenario = routed(pt_scenario(passengers(3, 0, 3), vehicles="network"))
    batch = PlanBatch.from_plans(scenario.population.selected_plans())
    collector = TrajectoryCollector(scenario.network, batch)  # no schedule
    tr = run(scenario, [collector])
    np.testing.assert_array_equal(collector.trajectories().leg_arrival, tr.leg_arrival)


def test_deterministic():
    scenario = routed(pt_scenario(passengers(5, 0, 3), vehicles="network", capacity=2))
    a, b = run(scenario), run(scenario)
    np.testing.assert_array_equal(a.ride_board, b.ride_board)


# ---------------------------------------------------------------- scoring


def scorer(**kw):
    params = dict(
        activities={"home": ActivityParams(12 * H), "work": ActivityParams(8 * H)},
        modes={"pt": ModeParams(marginal_utility_of_traveling=-6.0),
               "walk": ModeParams(marginal_utility_of_traveling=-12.0)},
        marginal_utility_of_waiting_pt=-6.0,
        utility_of_line_switch=-1.0,
    )
    params.update(kw)
    return CharyparNagelScorer(ScoringParams(**params))


def scores(tr, s):
    vectorised = s.score_trajectories(tr)
    per_plan = np.array([s.score(p) for p in tr.experienced_plans()])
    np.testing.assert_allclose(vectorised, per_plan)
    return vectorised


def test_pt_leg_scores_waiting_and_in_vehicle_time_separately():
    tr = run(routed(pt_scenario(passengers(1, 0, 3))))  # waits 200 s, rides 360 s
    base = scores(tr, scorer())
    no_wait = scores(tr, scorer(marginal_utility_of_waiting_pt=0.0))
    assert (base - no_wait)[0] == pytest.approx(-6.0 * 200 / H)
    no_ride = scores(tr, scorer(modes={"pt": ModeParams(marginal_utility_of_traveling=0.0),
                                       "walk": ModeParams(marginal_utility_of_traveling=-12.0)}))
    assert (base - no_ride)[0] == pytest.approx(-6.0 * 360 / H)


def test_line_switch_is_penalised():
    tr = run(routed(transfer_scenario()))
    diff = scores(tr, scorer()) - scores(tr, scorer(utility_of_line_switch=0.0))
    assert diff[0] == pytest.approx(-1.0)


def test_access_walk_scored_as_walk():
    tr = run(routed(pt_scenario(passengers(1, 0, "a"))))  # 100 s walk-only pt leg
    no_walk = scorer(modes={"pt": ModeParams(marginal_utility_of_traveling=-6.0),
                            "walk": ModeParams(marginal_utility_of_traveling=0.0)})
    diff = scores(tr, scorer()) - scores(tr, no_walk)
    assert diff[0] == pytest.approx(-12.0 * 100 / H)


def test_experienced_rides():
    scenario = routed(transfer_scenario())
    tr = run(scenario)
    (leg,) = tr.experienced_plans()[0].legs
    rides = [(r.board_route_stop, r.alight_route_stop) for r in leg.rides]
    assert scenario.transit.schedule.describe_rides(rides) == [
        ("A", "a", "S0", "S2"), ("B", "b", "S2", "S4")]
    assert leg.rides[1].wait_time == pytest.approx(60.0)


def test_learn_from_transit_adds_vehicle_link_times_without_dwell():
    from dataclasses import replace

    scenario = car_behind_bus(blocking=True)
    net = scenario.network
    link, t_bin = net.link_index("0-1"), int(T0 // 900)
    off = run(scenario).travel_time_table(net).times[link, t_bin]
    assert off == pytest.approx(251.0, abs=tol())  # the car, held by the bus at S1
    learn = replace(scenario.transit.params, learn_from_transit=True)
    on = run(replace(scenario, transit=scenario.transit.with_params(learn)))
    # the bus: entered 0-1 at T0, left at T0 + 300, of which 200 s dwell
    assert on.travel_time_table(net).times[link, t_bin] == pytest.approx((off + 100.0) / 2)
