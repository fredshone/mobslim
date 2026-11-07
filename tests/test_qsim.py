import math

import pytest

from mobslim import NetworkBuilder, Plan, Population, Route, Scenario
from mobslim.events import (
    ActivityEnd,
    ActivityStart,
    LinkEnter,
    LinkLeave,
    PersonArrival,
    PersonStuck,
    Teleportation,
)
from mobslim.mobsim import SimulationError
from tests import conftest
from tests.conftest import chain_network, commuters, routed, simulate

# Documented deviations from this spec, per mobsim: test -> reason. They are
# strict xfails, so a deviation that goes away is noticed.
_SHARD = "ShardSim follows matsim-rust's queue dynamics: "
DEVIATIONS = {
    "shard": {
        "test_free_flow_travel_time": _SHARD + "a vehicle reaches the buffer in one second "
                                      "and crosses the node in the next (+1 s per link)",
        "test_flow_capacity_headway": _SHARD + "vehicles ending their leg leave without flow capacity",
        "test_fractional_flow_capacity_is_not_rounded": _SHARD + "arrivals use no flow capacity",
        "test_flow_capacity_factor_scales_flow": _SHARD + "arrivals use no flow capacity",
        "test_storage_capacity_spillback": _SHARD + "arrivals use no flow capacity",
        "test_departures_onto_a_full_link_are_forced_one_per_stuck_time":
            _SHARD + "stuck means waited *more* than stuck_time (whole seconds)",
        "test_activity_end_time_vs_duration": _SHARD + "an activity starts one second after arrival",
    },
}


@pytest.fixture(params=sorted(conftest.MOBSIMS), autouse=True)
def mobsim_kind(request, monkeypatch):
    """Run every test in this module (the mobsim behavioural spec) against
    every mobsim."""
    monkeypatch.setattr(conftest, "DEFAULT_MOBSIM", request.param)
    reason = DEVIATIONS.get(request.param, {}).get(request.node.originalname)
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    return request.param


def arrivals(events):
    return [e.time for e in events if isinstance(e, PersonArrival)]


def test_free_flow_travel_time():
    net = chain_network([100, 100, 100], freespeed=10)
    events = simulate(routed(Scenario(net, commuters(1, 0, 3, dep=100))))
    assert [e.time for e in events if isinstance(e, LinkEnter)] == [100, 110, 120]
    assert arrivals(events) == [130]


def test_flow_capacity_headway():
    net = chain_network([100], freespeed=10, capacity=360)  # 1 veh / 10 s
    events = simulate(routed(Scenario(net, commuters(10, 0, 1))))
    assert arrivals(events) == [10.0 * (k + 1) for k in range(10)]


def test_fractional_flow_capacity_is_not_rounded():
    net = chain_network([100], freespeed=10, capacity=2000)  # 1.8 s headway
    n = 51
    times = arrivals(simulate(routed(Scenario(net, commuters(n, 0, 1)))))
    step = conftest.time_step()
    if not step:
        assert [round(b - a, 6) for a, b in zip(times, times[1:])] == [1.8] * (n - 1)
    else:  # whole-second clock: each exit within one step of the exact one,
        # and no drift over time
        exact = [10.0 + 1.8 * k for k in range(n)]
        assert all(0 <= t - e < step for t, e in zip(times, exact))


def test_capacity_is_per_link_not_per_lane():
    one = chain_network([100], capacity=360, lanes=1)
    two = chain_network([100], capacity=360, lanes=2)
    a = arrivals(simulate(routed(Scenario(one, commuters(5, 0, 1)))))
    b = arrivals(simulate(routed(Scenario(two, commuters(5, 0, 1)))))
    assert a == b


def test_flow_capacity_factor_scales_flow():
    net = chain_network([100], freespeed=10, capacity=360)
    times = arrivals(simulate(routed(Scenario(net, commuters(3, 0, 1))), flow_capacity_factor=0.5))
    assert times == [10, 30, 50]


def test_fifo_by_identity():
    net = chain_network([100, 100], capacity=720)
    events = simulate(routed(Scenario(net, commuters(20, 0, 2))))
    for link in (0, 1):
        enters = [e.person for e in events if isinstance(e, LinkEnter) and e.link == link]
        leaves = [e.person for e in events if isinstance(e, LinkLeave) and e.link == link]
        assert enters == leaves


def test_storage_capacity_spillback():
    # downstream link: 15 m, 1 lane -> storage 2 vehicles, slow outflow
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 1000, 0)
    b.add_node(2, 1015, 0)
    b.add_link("up", 0, 1, 1000, 10, 3600)
    b.add_link("down", 1, 2, 15, 10, 60)  # 1 veh/min
    net = b.build()
    events = simulate(routed(Scenario(net, commuters(10, 0, 2))), stuck_time=1e9)
    occupancy, peak = 0, 0
    for e in events:
        if isinstance(e, LinkEnter) and e.link == 1:
            occupancy += 1
        elif isinstance(e, LinkLeave) and e.link == 1:
            occupancy -= 1
        peak = max(peak, occupancy)
    assert peak == 2
    assert len(arrivals(events)) == 10
    times = arrivals(events)
    assert all(round(b - a, 6) >= 60 for a, b in zip(times, times[1:]))


def _gridlock_scenario():
    """Two 1-vehicle links forming a loop, with one vehicle on each wanting the other."""
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 7.5, 0)
    b.add_link("a", 0, 1, 7.5, 7.5, 360)
    b.add_link("b", 1, 0, 7.5, 7.5, 360)
    net = b.build()
    loop0 = Plan.builder().act("h", 0, end_time=0).leg("car", Route((0, 1), 2, 15)).act("h", 0).build()
    loop1 = Plan.builder().act("h", 1, end_time=0).leg("car", Route((1, 0), 2, 15)).act("h", 1).build()
    return Scenario(net, Population.from_plans({"x": loop0, "y": loop1}))


def test_stuck_time_resolves_gridlock():
    events = simulate(_gridlock_scenario(), stuck_time=10)
    assert len(arrivals(events)) == 2
    assert not [e for e in events if isinstance(e, PersonStuck)]
    assert max(arrivals(events)) >= 10


def test_departures_onto_a_full_link_are_forced_one_per_stuck_time():
    """A link holding one vehicle (10 s free-speed time, 360 veh/h); five cars depart
    onto it within a second. Blocked departures wait in a FIFO line, and only
    its front counts stuck time, from when it reached the front (MATSim's
    departure buffer): one forced entry per stuck time, in departure order,
    not the whole line at once."""
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 7.5, 0)
    b.add_link("a", 0, 1, 7.5, 0.75, 360)
    plans = {i: Plan.builder().act("h", 0, end_time=0.2 * i).leg("car").act("w", 1).build()
             for i in range(5)}
    events = simulate(routed(Scenario(b.build(), Population.from_plans(plans))), stuck_time=3.0)
    entries = [(e.person, e.time) for e in events if isinstance(e, LinkEnter)]
    assert [p for p, _ in entries] == [0, 1, 2, 3, 4]
    step = conftest.time_step()
    expected = [0.0, 3.2, 6.2, 9.2, 12.2]  # still over-full when 0 leaves at 10 s
    assert [t for _, t in entries] == pytest.approx(expected, abs=step + 1e-9)


def test_unresolved_gridlock_reports_stuck():
    events = simulate(_gridlock_scenario(), stuck_time=math.inf)
    stuck = [e for e in events if isinstance(e, PersonStuck)]
    assert len(stuck) == 2 and {e.mode for e in stuck} == {"car"}


# One car leaves at 100 s over three equal roads, then starts an activity:
# (road length, continuous_time, junction_delays, rounding) ->
# (road 2 and 3 entered, arrival, activity start). Junction delays add a
# second at each junction crossed and before the activity; whole seconds
# round each road's time (10.5 s: nearest 11, down 10).
TIMING_SPEC = {
    (100, True, False, "nearest"): ([110, 120], 130, 130),
    (100, True, True, "nearest"): ([111, 122], 132, 133),
    (100, False, False, "nearest"): ([110, 120], 130, 130),
    (100, False, True, "nearest"): ([111, 122], 132, 133),
    (100, False, False, "down"): ([110, 120], 130, 130),
    (100, False, True, "down"): ([111, 122], 132, 133),
    (105, True, False, "nearest"): ([110.5, 121], 131.5, 131.5),
    (105, True, True, "nearest"): ([111.5, 123], 133.5, 134.5),
    (105, False, False, "nearest"): ([111, 122], 133, 133),
    (105, False, True, "nearest"): ([112, 124], 135, 136),
    (105, False, False, "down"): ([110, 120], 130, 130),
    (105, False, True, "down"): ([111, 122], 132, 133),
}
CONTINUOUS = {"qsim", "agentsim", "agentsim_chain"}  # the others: whole seconds only


@pytest.mark.parametrize("key", sorted(TIMING_SPEC), ids=lambda k: "-".join(map(str, k)))
def test_timing_options(key, mobsim_kind):
    length, continuous, junction, rounding = key
    timing = dict(continuous_time=continuous, junction_delays=junction, rounding=rounding)
    if continuous and mobsim_kind not in CONTINUOUS:
        with pytest.raises(ValueError, match="whole seconds only"):
            conftest.make_mobsim(mobsim_kind, **timing)
        return
    net = chain_network([length] * 3, freespeed=10)
    plan = (Plan.builder().act("h", 0, end_time=100).leg("car").act("w", 3, duration=60)
            .leg("walk").act("x", 3).build())
    events = simulate(routed(Scenario(net, Population.from_plans({"a": plan}))), **timing)
    enters = [e.time for e in events if isinstance(e, LinkEnter)]
    start = [e.time for e in events if isinstance(e, ActivityStart) and e.act_type == "w"]
    expected_enters, arrival, act_start = TIMING_SPEC[key]
    assert enters == [100] + expected_enters
    assert arrivals(events)[0] == arrival
    assert start == [act_start]


def test_rounding_needs_whole_seconds(mobsim_kind):
    with pytest.raises(ValueError, match="whole seconds only"):
        conftest.make_mobsim(mobsim_kind, continuous_time=True, rounding="down")
    with pytest.raises(ValueError, match="rounding must be"):
        conftest.make_mobsim(mobsim_kind, continuous_time=False, rounding="up")


def test_end_time_aborts_unfinished():
    net = chain_network([1000], freespeed=1)
    events = simulate(routed(Scenario(net, commuters(1, 0, 1))), end_time=100)
    stuck = [e for e in events if isinstance(e, PersonStuck)]
    assert len(stuck) == 1 and stuck[0].link == 0 and stuck[0].time == 100


def test_activity_end_time_vs_duration():
    net = chain_network([100], freespeed=10)

    def work_end(work):
        plan = Plan.builder().act("h", 0, end_time=0).leg("car").act("w", 1, **work).leg("car").act("x", 1).build()
        events = simulate(routed(Scenario(net, Population.from_plans({"a": plan}))))
        return next(e.time for e in events if isinstance(e, ActivityEnd) and e.act_type == "w")

    assert work_end({"end_time": 5}) == 10  # arrived at 10 > end_time 5: leaves immediately
    assert work_end({"end_time": 50}) == 50
    assert work_end({"duration": 5}) == 15  # arrived at 10 + duration 5


def test_teleported_leg():
    net = chain_network([1000])
    scenario = Scenario(net, commuters(1, 0, 1, mode="walk"))
    events = simulate(scenario)
    tele = [e for e in events if isinstance(e, Teleportation)]
    assert tele[0].distance == pytest.approx(1300)  # beeline 1000 * 1.3
    assert arrivals(events) == [pytest.approx(1300 / (3 / 3.6))]


def test_unrouted_network_leg_raises():
    net = chain_network([100])
    with pytest.raises(SimulationError, match="no route"):
        simulate(Scenario(net, commuters(1, 0, 1)))


def test_invalid_route_raises():
    net = chain_network([100, 100])
    plan = Plan.builder().act("h", 0, end_time=0).leg("car", Route((1,), 1, 1)).act("w", 2).build()
    with pytest.raises(SimulationError, match="invalid"):
        simulate(Scenario(net, Population.from_plans({"a": plan})))


def test_mobsim_does_not_mutate_scenario():
    net = chain_network([100, 100], capacity=360)
    scenario = routed(Scenario(net, commuters(5, 0, 2)))
    before = list(scenario.population)
    simulate(scenario)
    assert list(scenario.population) == before


def test_mobsim_is_deterministic(equil_scenario):
    scenario = routed(equil_scenario)
    assert simulate(scenario) == simulate(scenario)


def test_route_ending_at_wrong_node_raises():
    net = chain_network([100, 100])
    plan = Plan.builder().act("h", 0, end_time=0).leg("car", Route((0,), 1, 1)).act("w", 2).build()
    with pytest.raises(SimulationError, match="invalid"):
        simulate(Scenario(net, Population.from_plans({"a": plan})))


def test_route_over_link_without_the_mode_raises():
    b = NetworkBuilder()
    for i in range(3):
        b.add_node(i, 100 * i, 0)
    b.add_link("a", 0, 1, 100, 10, 3600, modes={"car", "bike"})
    b.add_link("b", 1, 2, 100, 10, 3600, modes={"car"})
    plan = Plan.builder().act("h", 0, end_time=0).leg("bike", Route((0, 1), 1, 1)).act("w", 2).build()
    with pytest.raises(SimulationError, match="invalid 'bike' route"):
        simulate(Scenario(b.build(), Population.from_plans({"a": plan})))
