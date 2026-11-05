"""Transit integration tests on MATSim's example scenarios.

| Fixture | What it checks |
|---|---|
| ``pt_simple`` | XML loading; one line, one passenger, ``awaitDeparture`` |
| ``pt_lineswitch`` | RAPTOR finds a two-line trip with a walking transfer; the line-switch penalty |
| ``pt_tutorial`` | 900 persons, car and a train line: the full loop with mode choice, capacity, reproducibility |
| Sioux Falls 2014 | Loading a real-sized schedule (only if ``$MOBSLIM_SIOUXFALLS`` is set) |

Exact timings are pinned down in ``test_transit.py``. Here, against MATSim's
own inputs, the tests check structure (which lines, stops and vehicles),
exact walk times (walks start at the activities' coordinates), and vehicle
times within windows that allow for MATSim's 1 s steps.

The module is skipped until ``mobslim.transit`` provides the planned API.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from mobslim import (
    ActivityParams,
    CharyparNagelScorer,
    Controller,
    ControllerParams,
    ModeParams,
    PlanStrategy,
    Scenario,
    ScoringParams,
    StrategyManager,
    read_network_xml,
    read_population_xml,
    route_scenario,
)
from mobslim.events import EventLog, EventsManager
from mobslim.replanning import (
    ChangeExpBetaSelector,
    ChangeTripMode,
    ProcessPoolRunner,
    RandomSelector,
    Reroute,
    SerialRunner,
    TimeAllocationMutator,
)
from mobslim.timeutils import parse_time
from tests.conftest import FIXTURES, make_mobsim, routed

try:
    from mobslim.events import PersonEntersVehicle, VehicleDepartsAtFacility
    from mobslim.transit import (
        TransitParams,
        read_transit,
        read_transit_schedule_xml,
        read_transit_vehicles_xml,
        rust_raptor_available,
    )
except ImportError:  # pragma: no cover - until the transit API exists
    pytest.skip("transit simulation is not implemented yet",
                allow_module_level=True)

H = 3600.0
VEHICLE_MODES = ["schedule", "network"]
# the Python mobsims that run the MATSim fixtures here (HermesSim: test_hermes.py)
FIXTURE_MOBSIMS = ["qsim", "agentsim"]


def load(name, vehicles="schedule"):  # the default vehicle mode
    d = FIXTURES / name
    network = read_network_xml(d / "network.xml")
    plans = d / "population.xml"
    population = read_population_xml(plans if plans.exists() else d / "population.xml.gz", network)
    transit = read_transit(d / "transitSchedule.xml", d / "transitVehicles.xml", network,
                           TransitParams(vehicles=vehicles))
    return Scenario(network, population, transit=transit)


def run(scenario, handlers=(), mobsim=None):
    events = EventsManager(list(handlers))
    events.reset(0)
    tr = make_mobsim(mobsim).run(scenario, events)
    events.finish()
    return tr


def t(hms):
    return parse_time(hms)


# ---------------------------------------------------------------- loading


def test_load_pt_simple():
    network = read_network_xml(FIXTURES / "pt_simple" / "network.xml")
    s = read_transit_schedule_xml(FIXTURES / "pt_simple" / "transitSchedule.xml", network)
    assert (s.n_stops, s.n_lines, s.n_routes, s.n_departures) == (4, 1, 1, 1)
    assert s.dep_time.tolist() == [t("05:05:00")]
    assert s.rs_arrival.tolist() == [0, 30, 1800, 1830]
    assert s.rs_await.all()
    assert s.route_links.tolist() == [network.link_index(k) for k in ("1-2", "2-3", "3-4", "4-5", "5-6")]
    assert s.rs_link_pos.tolist() == [0, 1, 3, 4]  # stop 495 is on 4-5, the route's 4th link
    assert s.stop_link[s.stop_ids.index("495")] == network.link_index("4-5")
    assert s.route_mode_name(0) == "pt"

    v = read_transit_vehicles_xml(FIXTURES / "pt_simple" / "transitVehicles.xml")
    vt = v.vehicle_type[v.vehicle_ids.index("1000")]
    assert v.capacity[vt] == 150  # 50 seats + 100 standing
    assert v.length[vt] == 12.0
    # not in the file: MATSim's defaults
    assert (v.access_time[vt], v.egress_time[vt], v.pce[vt]) == (1.0, 1.0, 1.0)
    assert not v.door_parallel[vt]


def test_load_pt_lineswitch_skips_commented_stops():
    network = read_network_xml(FIXTURES / "pt_lineswitch" / "network.xml")
    s = read_transit_schedule_xml(FIXTURES / "pt_lineswitch" / "transitSchedule.xml", network)
    assert sorted(s.stop_ids) == sorted(["293", "495", "54", "32", "47", "910"])
    assert [s.line_ids.id(i) for i in s.route_line.tolist()] == ["gelb", "rot", "blau"]
    assert s.dep_time.tolist() == [t("05:05:00"), t("05:20:00"), t("05:30:00")]
    gelb = s.route_index("gelb", "gelb_1")
    assert s.route_stops_of(gelb)["link_pos"].tolist() == [2, 4]  # on 2-3 and 4-5 of 7 links


def test_load_pt_tutorial_normalises_offsets():
    network = read_network_xml(FIXTURES / "pt_tutorial" / "network.xml")
    s = read_transit_schedule_xml(FIXTURES / "pt_tutorial" / "transitSchedule.xml", network)
    assert (s.n_stops, s.n_lines, s.n_routes, s.n_departures) == (4, 1, 2, 100)
    r = s.route_index("Blue Line", "1to3")
    stops = s.route_stops_of(r)
    # a missing arrival offset is the departure offset and vice versa
    assert stops["arrival"].tolist() == [0, 200, 540]
    assert stops["departure"].tolist() == [0, 240, 540]
    assert not stops["await"].any()  # awaitDeparture defaults to false
    times = s.dep_time[s.route_dep_offsets[r]:s.route_dep_offsets[r + 1]]
    assert len(times) == 50 and np.all(np.diff(times) > 0)  # sorted per route
    assert (times[0], times[-1]) == (t("06:00:00"), t("23:00:00"))
    # transportMode differs between the routes and is kept as is
    assert s.route_mode_name(r) == "train"
    assert s.route_mode_name(s.route_index("Blue Line", "3to1")) == "pt"

    v = read_transit_vehicles_xml(FIXTURES / "pt_tutorial" / "transitVehicles.xml")
    assert sorted(v.vehicle_ids) == ["tr_1", "tr_2"]
    assert v.capacity.tolist() == [80]


def test_schedule_and_vehicles_pickle():
    import pickle

    transit = load("pt_tutorial").transit
    again = pickle.loads(pickle.dumps(transit))
    np.testing.assert_array_equal(again.schedule.dep_time, transit.schedule.dep_time)
    np.testing.assert_array_equal(again.vehicles.capacity, transit.vehicles.capacity)


SIOUXFALLS = os.environ.get("MOBSLIM_SIOUXFALLS")

# file names and expected sizes of the two published Sioux Falls scenarios:
# matsim-eth/sioux-falls (src/main/resources/ch/ethz/matsim/sioux_falls) and
# matsim-libs examples/scenarios/siouxfalls-2014
SIOUXFALLS_LAYOUTS = {
    "matsim-eth": dict(network="network.xml", schedule="schedule.xml", vehicles="vehicles.xml",
                       population="population.xml.gz", sizes=(150, 10, 10, 2050)),
    "siouxfalls-2014": dict(network="Siouxfalls_network_PT.xml",
                            schedule="Siouxfalls_transitSchedule.xml",
                            vehicles="Siouxfalls_vehicles.xml",
                            population="Siouxfalls_population.xml.gz", sizes=(133, 10, 10, 2040)),
}


def siouxfalls_layout(d: Path) -> dict:
    for layout in SIOUXFALLS_LAYOUTS.values():
        if (d / layout["schedule"]).exists():
            return layout
    pytest.fail(f"{d} holds neither Sioux Falls layout ({', '.join(SIOUXFALLS_LAYOUTS)})")


@pytest.mark.skipif(not SIOUXFALLS, reason="set MOBSLIM_SIOUXFALLS to a Sioux Falls scenario dir")
def test_load_siouxfalls():
    d = Path(SIOUXFALLS)
    f = siouxfalls_layout(d)
    network = read_network_xml(d / f["network"])
    s = read_transit_schedule_xml(d / f["schedule"], network)
    assert (s.n_stops, s.n_lines, s.n_routes, s.n_departures) == f["sizes"]
    v = read_transit_vehicles_xml(d / f["vehicles"])
    assert len(v.vehicle_ids) == s.n_departures
    assert v.capacity.tolist() == [90]  # 38 seats + 52 standing
    assert v.access_time.tolist() == [1.0] and v.egress_time.tolist() == [1.0]
    assert s.route_links.size and s.dep_vehicle.min() >= 0  # every departure has a vehicle


@pytest.mark.skipif(not SIOUXFALLS, reason="set MOBSLIM_SIOUXFALLS to a Sioux Falls scenario dir")
def test_siouxfalls_runs():
    """The whole population routes and rides (legacy ``transit_walk`` legs
    read as pt); nobody is stuck in either vehicle mode."""
    from dataclasses import replace

    from mobslim import HermesSim, HermesSimParams

    d = Path(SIOUXFALLS)
    f = siouxfalls_layout(d)
    network = read_network_xml(d / f["network"])
    transit = read_transit(d / f["schedule"], d / f["vehicles"], network)
    population = read_population_xml(d / f["population"], network)
    assert {leg.mode for plan in population.selected_plans() for leg in plan.legs} <= {
        "car", "pt", "walk"}
    scenario = route_scenario(Scenario(network, population, transit=transit))
    for vehicles in VEHICLE_MODES:
        s = replace(scenario, transit=transit.with_params(
            replace(transit.params, vehicles=vehicles)))
        tr = HermesSim(HermesSimParams(flow_capacity_factor=0.7)).run(s, EventsManager())
        assert not tr.stuck.any(), vehicles
        assert np.isfinite(tr.ride_alight).all() and len(tr.ride_alight) > 10_000
        assert tr.pt_load().max() <= 90


# ---------------------------------------------------------------- pt-simple


@pytest.mark.parametrize("mobsim", FIXTURE_MOBSIMS)
@pytest.mark.parametrize("vehicles", VEHICLE_MODES)
def test_pt_simple(vehicles, mobsim):
    scenario = routed(load("pt_simple", vehicles))
    s = scenario.transit.schedule
    assert scenario.modes.is_transit_mode("pt")  # although the links allow "pt"
    leg = scenario.population[0].selected_plan.legs[0]
    # least cost: alighting at 495 and walking 14 m beats riding 30 s on to 596
    # and walking 10 m (0.074 vs 0.10 util with the default RaptorParams)
    assert s.describe_rides(leg.route.rides) == [("gelb", "gelb_1", "192", "495")]

    log = EventLog()
    tr = run(scenario, [log], mobsim)
    assert not tr.stuck[0]
    assert tr.ride_board[0] >= t("05:05:00")
    # walks run between activity and stop coordinates: (0, 10) -> stop 192 at
    # (5, 0), and stop 495 at (2005, 0) -> (2015, 10); 3 km/h, beeline x 1.3
    walk = 1.3 / (3 / 3.6)
    assert tr.ride_wait_start[0] - tr.leg_departure[0] == pytest.approx(np.hypot(5, 10) * walk)
    assert tr.leg_arrival[0] - tr.ride_alight[0] == pytest.approx(np.hypot(10, 10) * walk)
    if vehicles == "schedule":  # scheduled arrival at 495 is 05:35:00
        assert t("05:35:00") <= tr.leg_arrival[0] <= t("05:36:00")
    else:
        # the vehicle reaches 495 early (about 05:07) and awaits its 05:35:00
        # departure there; passengers alight on arrival
        assert t("05:07:00") <= tr.leg_arrival[0] <= t("05:08:00")
        dep_495 = [e.time for e in log.of_type(VehicleDepartsAtFacility) if e.stop == s.stop_ids.index("495")]
        assert dep_495 == pytest.approx([t("05:35:00")], abs=1.0)


# ---------------------------------------------------------------- pt-simple-lineswitch


@pytest.mark.parametrize("mobsim", FIXTURE_MOBSIMS)
@pytest.mark.parametrize("vehicles", VEHICLE_MODES)
def test_pt_lineswitch_routes_and_runs_the_transfer(vehicles, mobsim):
    scenario = routed(load("pt_lineswitch", vehicles))
    s = scenario.transit.schedule
    first, second = scenario.population[0].selected_plan.legs
    assert s.describe_rides(first.route.rides) == [("gelb", "gelb_1", "293", "495")]
    # back: rot to 47, walk 20 m to 910, blau to 32
    assert s.describe_rides(second.route.rides) == [("rot", "rot_1", "54", "47"),
                                                    ("blau", "blau_1", "910", "32")]

    log = EventLog()
    tr = run(scenario, [log], mobsim)
    assert not tr.stuck[0]
    assert [e.trip for e in log.of_type(PersonEntersVehicle)] == [0, 1, 2]
    assert t("05:06:30") <= tr.leg_arrival[0] <= t("05:08:00")  # gelb reaches 495 at 05:07
    if vehicles == "schedule":
        assert t("05:31:30") <= tr.leg_arrival[1] <= t("05:33:30")  # blau reaches 32 at 05:32
        # waited at 910 from about 05:22:30 for blau at 05:30
        assert 6 * 60 <= tr.ride_board[2] - tr.ride_wait_start[2] <= 9 * 60
    else:
        # on the network rot and blau run ahead of the timetable (they await
        # departures, not arrivals): rot reaches 47 at about 05:20:40 and blau
        # reaches 32 at about 05:30:40
        assert t("05:30:30") <= tr.leg_arrival[1] <= t("05:33:30")
        assert 8 * 60 <= tr.ride_board[2] - tr.ride_wait_start[2] <= 10 * 60


def test_pt_lineswitch_line_switch_penalty():
    tr = run(routed(load("pt_lineswitch", "schedule")))

    def score(line_switch):
        params = ScoringParams(activities={"home": ActivityParams(18 * H)},
                               modes={"walk": ModeParams(marginal_utility_of_traveling=-12.0)},
                               utility_of_line_switch=line_switch)
        return CharyparNagelScorer(params).score_trajectories(tr)[0]

    assert score(-1.0) - score(0.0) == pytest.approx(-1.0)  # one switch, on the second trip


# ---------------------------------------------------------------- pt-tutorial


def tutorial_controller(iterations, runner=None, vehicles="schedule"):
    scenario = load("pt_tutorial", vehicles)
    scoring = ScoringParams(
        activities={"h": ActivityParams(12 * H), "w": ActivityParams(8 * H), "s": ActivityParams(40 * 60)},
        modes={"car": ModeParams(marginal_utility_of_traveling=-6.0),
               "pt": ModeParams(marginal_utility_of_traveling=-6.0),
               "walk": ModeParams(marginal_utility_of_traveling=-12.0)},
        utility_of_line_switch=-1.0,
    )
    strategies = StrategyManager([
        (0.7, PlanStrategy(ChangeExpBetaSelector())),
        (0.1, PlanStrategy(RandomSelector(), [Reroute()])),
        (0.1, PlanStrategy(RandomSelector(), [TimeAllocationMutator(), Reroute()])),
        (0.1, PlanStrategy(RandomSelector(), [ChangeTripMode(["car", "pt"]), Reroute()])),
    ])
    return Controller(scenario, scorer=CharyparNagelScorer(scoring), strategies=strategies,
                      runner=runner, params=ControllerParams(iterations=iterations, seed=4711))


@pytest.mark.parametrize("vehicles", VEHICLE_MODES)
def test_pt_tutorial_learns(vehicles):
    result = tutorial_controller(8, vehicles=vehicles).run()
    m = result.metrics
    assert all(row["stuck"] == 0 for row in m)
    assert all(row["pt_boardings"] > 0 for row in m)
    assert all(row["pt_max_load"] <= 80 for row in m)  # train capacity is never exceeded
    assert m[-1]["score_best"] > m[0]["score_executed"]
    # mode choice is active: plans of both modes exist at the end
    modes = {leg.mode for p in result.scenario.population for plan in p.plans for leg in plan.legs}
    assert modes == {"car", "pt"}


def test_pt_tutorial_serial_equals_parallel():
    serial = tutorial_controller(3, runner=SerialRunner()).run()
    with ProcessPoolRunner(workers=2) as runner:
        parallel = tutorial_controller(3, runner=runner).run()
    assert serial.metrics == parallel.metrics


def test_pt_tutorial_router():
    scenario = load("pt_tutorial")
    from mobslim.transit import RaptorRouter

    router = RaptorRouter(scenario.network, scenario.transit)
    s, net = scenario.transit.schedule, scenario.network
    # node 11 is 71 m from station 1, node 41 71 m from station 3: take the train
    # (walk 110 s to the station misses 07:00, so the 07:10 departure)
    route = router.route(net.node_index("11"), net.node_index("41"), t("07:00:00"))
    assert s.describe_rides(route.rides) == [("Blue Line", "1to3", "1", "3")]
    assert t("07:19:00") - t("07:00:00") < route.travel_time < t("07:22:00") - t("07:00:00")
    # node 14 is 1.5 km from the nearest station, beyond the 1 km search radius:
    # the search is extended, but walking straight to node 44 is better
    route = router.route(net.node_index("14"), net.node_index("44"), t("07:00:00"))
    assert route.n_rides == 0
    assert route.travel_time == pytest.approx(3000 * 1.3 / (3 / 3.6))


# ---------------------------------------------------------------- compiled router


def random_queries(scenario, n, seed):
    """Random points around the stops (some shared origins), random times in
    the service window."""
    rng = np.random.default_rng(seed)
    s = scenario.transit.schedule
    xy = s.stop_xy
    lo, hi = xy.min(axis=0) - 500.0, xy.max(axis=0) + 500.0
    o = rng.uniform(lo, hi, (n, 2))
    o[n // 2:] = o[:n - n // 2][rng.integers(0, n // 4 + 1, n - n // 2)]  # repeated origins
    d = rng.uniform(lo, hi, (n, 2))
    d[: n // 5] = xy[rng.integers(0, len(xy), n // 5)]  # exactly at stops
    t = np.round(rng.uniform(s.dep_time.min() - 900, s.dep_time.max(), n) / 60.0) * 60.0
    return o, d, t


@pytest.mark.skipif(not rust_raptor_available(), reason="mobslim_router (Rust) not built")
@pytest.mark.parametrize("name", ["pt_simple", "pt_lineswitch", "pt_tutorial"])
@pytest.mark.parametrize("config", [
    {}, {"min_transfer_time": 120.0, "max_transfer_distance": 400.0},
    {"line_switch": -10.0, "waiting": -1.0, "search_radius": 300.0, "extension_radius": 50.0},
    {"max_transfers": 1, "max_transfers_after_first_arrival": 0},
])
def test_rust_raptor_matches_python(name, config):
    from mobslim.transit import RaptorParams, RaptorRouter, RustRaptorRouter

    scenario = load(name)
    cfg = RaptorParams(**config)
    py = RaptorRouter(scenario.network, scenario.transit, cfg)
    rs = RustRaptorRouter(scenario.network, scenario.transit, cfg, threads=3)
    o, d, t = random_queries(scenario, 300, seed=len(name))
    expected = py.route_many_xy([tuple(p) for p in o.tolist()], [tuple(p) for p in d.tolist()],
                                t.tolist())
    assert rs.route_many_xy(o, d, t) == expected
    for q in range(0, 300, 37):  # single queries, as `route_xy` / `route`
        assert rs.route_xy(o[q], d[q], t[q]) == expected[q]
    assert sum(r.n_rides > 0 for r in expected) >= 10  # the queries used transit
    nodes = np.arange(min(20, scenario.network.n_nodes))
    assert rs.route_many(nodes, nodes[::-1], [7 * H] * len(nodes)) == \
        py.route_many(nodes, nodes[::-1], [7 * H] * len(nodes))
