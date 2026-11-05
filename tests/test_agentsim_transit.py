"""Transit in AgentSim (``AGENTSIM_TRANSIT_PLAN.md``). The shared transit spec
runs against AgentSim in ``test_transit.py``; this module checks where its
native implementation must exactly agree with the QSim, and the mechanisms
only AgentSim has (tickets, displaced heads, waiting without polling)."""

import heapq
import random

import numpy as np
import pytest

from mobslim import (
    AgentSim,
    AgentSimConfig,
    NetworkBuilder,
    Plan,
    Population,
    QSim,
    QSimConfig,
    Scenario,
)
from mobslim.events import EventLog, EventsManager, VehicleDepartsAtFacility
from mobslim.mobsim import agentsim
from mobslim.transit import PtRoute, TransitBuilder, TransitConfig
from tests.conftest import routed
from tests.test_transit import (
    STOP_LINKS,
    T0,
    WALK,
    corridor_network,
    corridor_transit,
    node,
    passengers,
    pt_scenario,
    rides_of,
)
from tests.test_transit_scenarios import load

INF = float("inf")
_ARRAYS = ("act_start", "act_end", "leg_departure", "leg_arrival", "link_enter", "stuck",
           "ride_wait_start", "ride_board", "ride_alight", "ride_trip")


def assert_same(a, b):
    for name in _ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    for name in ("stop_arrival", "stop_departure", "link_enter"):
        np.testing.assert_array_equal(getattr(a.transit, name), getattr(b.transit, name),
                                      err_msg=f"transit.{name}")


def _sorted_events(events):
    return sorted(events, key=lambda e: (e.time, type(e).__name__, repr(e)))


def run_both(scenario, **config):
    """QSim and AgentSim trajectories and events on ``scenario``."""
    out = []
    for sim in (QSim(QSimConfig(**config)), AgentSim(AgentSimConfig(**config))):
        log = EventLog()
        out.append((sim.run(scenario, EventsManager([log])), log.events))
    return out


@pytest.fixture
def checked_heads(monkeypatch):
    """Assert that only a link's head ever leaves it (displaced heads included)."""
    leave = agentsim._AgentSimRun._leave_link

    def checked(self, agent):
        assert agent.link.queue[0] is agent
        leave(self, agent)

    monkeypatch.setattr(agentsim._AgentSimRun, "_leave_link", checked)


# ---------------------------------------------------------------- parity with the QSim


@pytest.mark.parametrize("vehicles", ["schedule", "network"])
def test_pt_tutorial_matches_qsim(vehicles):
    """No spillback: identical trajectories; events equal up to the order of
    same-time events."""
    (q, qe), (a, ae) = run_both(routed(load("pt_tutorial", vehicles)), storage_capacity_factor=1e6)
    assert_same(q, a)
    assert np.isfinite(a.ride_board).sum() > 100
    if vehicles == "schedule":
        assert _sorted_events(qe) == _sorted_events(ae)


def mixed_corridor(seed=3, n_cars=120, n_pax=80, headway=90.0, doors="parallel",
                   blocking=False, pce=1.0):
    """Buses every ``headway`` s in mixed traffic with cars and pt passengers
    on the corridor (non-blocking stops by default: buses pull in and
    rejoin as head, displacing queued cars)."""
    rng = random.Random(seed)
    net = corridor_network()
    b = TransitBuilder(net)
    b.vehicle_type("bus", seats=12, standing=0, access_time=2.0, egress_time=2.0,
                   doors=doors, pce=pce)
    for i in range(5):
        b.stop(f"S{i}", 1000.0 * i, 0.0, link=STOP_LINKS[f"S{i}"], blocking=blocking)
    b.route("L", "r", mode="bus", links=["a-0", "0-1", "1-2", "2-3", "3-4"],
            stops=[(f"S{i}", 120.0 * i, 120.0 * i, i % 2 == 0) for i in range(5)])
    b.departures("L", "r", times=[T0 + headway * k for k in range(20)], vehicle_type="bus")
    transit = b.build(TransitConfig(vehicles="network", walk=WALK, door_open_time=1.0,
                                    door_close_time=1.0))
    plans = {}
    for i in range(n_cars):
        o = rng.randrange(0, 3)
        plans[i] = (Plan.builder().act("home", node(net, o), end_time=T0 + rng.uniform(0, 1800))
                    .leg("car").act("work", node(net, rng.randrange(o + 1, 5))).build())
    for i in range(n_cars, n_cars + n_pax):
        o = rng.randrange(0, 3)
        plans[i] = (Plan.builder()
                    .act("home", node(net, o), xy=(1000.0 * o, 0.0), end_time=T0 + rng.uniform(-300, 1500))
                    .leg("pt").act("work", node(net, rng.randrange(o + 1, 5))).build())
    return routed(Scenario(net, Population.from_plans(plans), teleported_modes={"walk": WALK},
                           transit=transit))


@pytest.mark.parametrize("doors", ["serial", "parallel"])
@pytest.mark.parametrize("blocking", [False, True])
def test_mixed_traffic_matches_qsim_without_spillback(doors, blocking, checked_heads):
    scenario = mixed_corridor(doors=doors, blocking=blocking)
    (q, _), (a, _) = run_both(scenario, storage_capacity_factor=1e6, flow_capacity_factor=0.05)
    assert_same(q, a)
    assert not a.stuck.any()
    assert (a.ride_board > a.ride_wait_start).any()  # some passengers waited


def test_displaced_car_leaves_after_the_bus(checked_heads):
    """A bus finishes its dwell at a non-blocking stop while a car is still on
    the link: it rejoins ahead of the car, which leaves after it (one flow
    headway later at the earliest), exactly as in the QSim."""
    net = corridor_network()
    # the bus reaches S1 (on 0-1) at T0 + 100 and awaits T0 + 300; the car
    # enters 0-1 at T0 + 250 and could leave at T0 + 350
    plan = (Plan.builder().act("home", node(net, 0), end_time=T0 + 250).leg("car")
            .act("work", node(net, 2)).build())
    transit = corridor_transit(net, vehicles="network", departures=(T0,),
                               stops=[("S0", 0, 0, True), ("S1", 300, 300, True)])
    scenario = routed(Scenario(net, Population.from_plans({0: plan}),
                               teleported_modes={"walk": WALK}, transit=transit))
    for flow, car_leaves in [(1.0, T0 + 350), (0.01, T0 + 300 + 100)]:  # headway 1 s / 100 s
        (q, _), (a, _) = run_both(scenario, flow_capacity_factor=flow)
        assert_same(q, a)
        assert a.link_enter[1] == pytest.approx(car_leaves)  # enters 1-2
        bus_links = a.transit.link_enter[a.transit.trip_link_offsets[0]:a.transit.trip_link_offsets[1]]
        assert bus_links[2] == pytest.approx(T0 + 300)  # the bus left 0-1 first


def test_passenger_reopens_doors_of_bus_awaiting_departure():
    """The bus reaches S1 at T0 + 100 and awaits T0 + 120 with its doors
    closed; a passenger arriving at T0 + 110 reopens them (cancelling the
    pending departure): doors 1 s, boarding 2 s, doors 1 s, still on time."""
    def pop(net):
        return Population.from_plans({0: Plan.builder()
                                      .act("home", node(net, 1), xy=(1000.0 - 110.0, 0.0), end_time=T0)
                                      .leg("pt").act("work", node(net, 3)).build()})

    scenario = routed(pt_scenario(pop, vehicles="network", departures=(T0,), access=2.0,
                                  door_open=1.0, door_close=1.0))
    (q, qe), (a, ae) = run_both(scenario)
    assert_same(q, a)
    (r,) = rides_of(a, 0)
    assert a.ride_wait_start[r] == pytest.approx(T0 + 110)
    assert a.ride_board[r] == pytest.approx(T0 + 113)
    dep_s1 = [e.time for e in ae if isinstance(e, VehicleDepartsAtFacility) and e.stop == 1]
    assert dep_s1 == pytest.approx([T0 + 120])


def test_pce_takes_storage():
    """A 2.5-PCE bus holds a 3-PCU link at a blocking stop: one car fits
    behind it, the next waits until the bus leaves."""
    b = NetworkBuilder()
    for i, x in enumerate([-100.0, 0.0, 22.5, 1022.5]):
        b.add_node(i, x, 0.0)
    b.add_link("in", 0, 1, 100.0, 10.0, 3600.0, modes=("car", "bus"))
    b.add_link("short", 1, 2, 22.5, 10.0, 3600.0, modes=("car", "bus"))  # stores 3 vehicles
    b.add_link("out", 2, 3, 1000.0, 10.0, 3600.0, modes=("car", "bus"))
    net = b.build()
    tb = TransitBuilder(net)
    tb.vehicle_type("bus", seats=10, pce=2.5)
    tb.stop("A", -100.0, 0.0, link="in")
    tb.stop("B", 22.5, 0.0, link="short", blocking=True)
    tb.route("L", "r", mode="bus", links=["in", "short", "out"], stops=[("A", 0, 0), ("B", 300, 300)])
    tb.departures("L", "r", times=[T0], vehicle_type="bus")
    transit = tb.build(TransitConfig(vehicles="network", walk=WALK))
    plans = {i: Plan.builder().act("home", 1, end_time=T0 + 10 + i).leg("car").act("work", 3).build()
             for i in range(2)}
    scenario = routed(Scenario(net, Population.from_plans(plans), teleported_modes={"walk": WALK},
                               transit=transit))
    for tr in (q for q, _ in run_both(scenario, stuck_time=1e5)):
        # each car's route starts on "short"
        first, second = (tr.link_enter[tr.plans.route_offsets[tr.plans.leg_offsets[i]]] for i in (0, 1))
        assert first == pytest.approx(T0 + 10)
        # entered once the bus left at T0 + 300: at once in the QSim, at the
        # next retry tick (the bus leaves late in its second) in AgentSim
        assert T0 + 300 <= second <= T0 + 301


# ---------------------------------------------------------------- spillback


_SPILLBACK = [
    dict(storage_capacity_factor=0.05, flow_capacity_factor=0.05),
    dict(storage_capacity_factor=0.05, flow_capacity_factor=0.05, stuck_time=INF),
    dict(storage_capacity_factor=0.05, flow_capacity_factor=0.05, stuck_time=60.0,
         retry_interval=2.5),
]


@pytest.mark.parametrize("config", _SPILLBACK, ids=["stuck10", "no_stuck", "stuck60"])
@pytest.mark.parametrize("pce", [1.0, 2.5])
def test_mixed_traffic_spillback_chain_matches_poll(config, pce, checked_heads):
    """Buses blocked downstream poll like cars; chain mode only skips retries
    that must fail, also with buses in stops filling links."""
    scenario = mixed_corridor(pce=pce)
    poll = AgentSim(AgentSimConfig(**config)).run(scenario, EventsManager())
    chain = AgentSim(AgentSimConfig(blocked="chain", **config)).run(scenario, EventsManager())
    assert_same(poll, chain)
    # every arrived passenger rode; times along each trip never decrease
    v = poll.transit
    for d in range(len(v.trip_stop_offsets) - 1):
        arr = v.stop_arrival[v.trip_stop_offsets[d]:v.trip_stop_offsets[d + 1]]
        dep = v.stop_departure[v.trip_stop_offsets[d]:v.trip_stop_offsets[d + 1]]
        times = np.column_stack([arr, dep]).ravel()
        times = times[~np.isnan(times)]
        assert (np.diff(times) >= 0).all()


def test_storage_is_released(monkeypatch):
    """After a run, every link's PCE occupancy is back to zero (no leak through
    stops, rejoining or displaced heads)."""
    runs = []
    original = agentsim._AgentSimRun._abort_unfinished
    monkeypatch.setattr(agentsim._AgentSimRun, "_abort_unfinished",
                        lambda self: (runs.append(self), original(self)))
    AgentSim(AgentSimConfig(storage_capacity_factor=0.05, flow_capacity_factor=0.05)).run(
        mixed_corridor(pce=2.5), EventsManager())
    (run,) = runs
    assert not any(t.link is not None for t in run.trips)
    assert all(abs(link.occupied) < 1e-9 for link in run.links)


# ---------------------------------------------------------------- waiting


def _heap_pops(scenario, monkeypatch):
    count = [0]
    pop = heapq.heappop

    def counting(heap):
        count[0] += 1
        return pop(heap)

    monkeypatch.setattr(agentsim.heapq, "heappop", counting)
    tr = AgentSim().run(scenario, EventsManager())
    monkeypatch.setattr(agentsim.heapq, "heappop", pop)
    return tr, count[0]


def test_waiting_passengers_are_not_polled(monkeypatch):
    """Waiting costs no heap entries: passengers who wait 50 min pop as many
    entries as passengers who wait 5 min."""
    # 25 passengers, 3 buses of 10 seats: the last 5 wait 20 min longer
    short, n_short = _heap_pops(routed(pt_scenario(passengers(25, 0, 3, dep=T0 - 300))), monkeypatch)
    long, n_long = _heap_pops(routed(pt_scenario(passengers(25, 0, 3, dep=T0 - 3000))), monkeypatch)
    assert n_short == n_long
    assert (long.ride_board - long.ride_wait_start > 2900).all()


@pytest.mark.parametrize("vehicles", ["schedule", "network"])
def test_missed_last_departure_is_stuck_without_stuck_time(vehicles):
    """With ``stuck_time=inf`` the run still ends after the last trip; the
    passenger left waiting (no heap entry) is stuck."""
    net = corridor_network()
    transit = corridor_transit(net, vehicles=vehicles)
    s = transit.schedule
    route = PtRoute([(s.route_stop("L", "r", "S0"), s.route_stop("L", "r", "S3"))], 360.0, 3000.0)
    plan = (Plan.builder().act("home", node(net, 0), end_time=T0 + 1300).leg("pt", route=route)
            .act("work", node(net, 3)).build())
    scenario = Scenario(net, Population.from_plans({0: plan}), teleported_modes={"walk": WALK},
                        transit=transit)
    tr = AgentSim(AgentSimConfig(stuck_time=INF)).run(scenario, EventsManager())
    assert tr.stuck[0]
    assert np.isnan(tr.ride_board[0]) and tr.ride_wait_start[0] == T0 + 1300
