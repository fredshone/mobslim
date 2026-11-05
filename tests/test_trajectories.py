"""Plan encoding, trajectories, and the array-based outputs derived from them.

The central checks compare independent paths that must agree:

* the QSim's directly recorded trajectories vs. trajectories rebuilt from its
  events (:class:`TrajectoryCollector`);
* experienced plans built from the arrays vs. the event-based
  :class:`ExperiencedPlanCollector`;
* vectorised scoring vs. per-plan scoring;
* the travel-time table from the arrays vs. :class:`TravelTimeCollector`.
"""

import math

import numpy as np
import pytest

from mobslim import CharyparNagelScorer, Plan, Population, Route, Scenario
from mobslim.events import EventsManager
from mobslim.experience import ExperiencedPlanCollector
from mobslim.planbatch import PlanBatch
from mobslim.routing import TravelTimeCollector, TravelTimeTable
from mobslim.scoring import ActivityParams, ModeParams, ScoringParams
from mobslim.trajectories import Trajectories, TrajectoryCollector
from tests import conftest
from tests.conftest import chain_network, commuters, make_mobsim, routed
from tests.test_qsim import _gridlock_scenario

ARRAYS = ("act_start", "act_end", "leg_departure", "leg_arrival", "leg_distance",
          "link_enter", "stuck")


@pytest.fixture(params=sorted(conftest.MOBSIMS), autouse=True)
def mobsim_kind(request, monkeypatch):
    """Every mobsim's direct trajectories must agree with its events."""
    monkeypatch.setattr(conftest, "DEFAULT_MOBSIM", request.param)
    return request.param


def run_all_ways(scenario: Scenario, **config):
    """Run once with event handlers attached; return
    ``(direct trajectories, trajectories from events, experienced plans from
    events, travel-time table from events)``."""
    plans = PlanBatch.from_plans(scenario.population.selected_plans())
    events = EventsManager()
    from_events = events.add_handler(TrajectoryCollector(scenario.network, plans))
    experienced = events.add_handler(ExperiencedPlanCollector(scenario.network, scenario.population))
    tt = events.add_handler(TravelTimeCollector(scenario.network))
    events.reset(0)
    direct = make_mobsim(**config).run_batch(scenario, plans, events)
    events.finish()
    return direct, from_events.trajectories(), experienced.plans(), tt.table()


def mixed_scenario() -> Scenario:
    """Car, teleported walk, a zero-length car leg and a single-activity plan."""
    net = chain_network([500, 500, 500], capacity=360)
    plans = {
        "car-walk": Plan.builder().act("home", 0, end_time=100).leg("car")
        .act("work", 3, duration=600).leg("walk").act("home", 2).build(),
        "same-node": Plan.builder().act("home", 1, end_time=50).leg("car")
        .act("shop", 1, duration=60).leg("car").act("home", 3).build(),
        "stay": Plan.builder().act("home", 2).build(),
    }
    plans |= {f"c{i}": Plan.builder().act("home", 0, end_time=100 + i).leg("car")
              .act("work", 3).build() for i in range(5)}
    return routed(Scenario(net, Population.from_plans(plans)))


def stuck_scenarios():
    yield "gridlock", _gridlock_scenario(), {"stuck_time": math.inf}
    # end_time cuts persons mid-activity (400 s: at work) and mid-teleport
    # (900 s: walking home, which takes 780 s from 850 s)
    yield "end_time", mixed_scenario(), {"end_time": 400}
    yield "end_mid_teleport", mixed_scenario(), {"end_time": 900}
    net = chain_network([1000], freespeed=1)
    yield "end_mid_link", routed(Scenario(net, commuters(3, 0, 1))), {"end_time": 100}


def all_scenarios(equil):
    yield "equil", routed(equil), {}
    yield "mixed", mixed_scenario(), {}
    yield from stuck_scenarios()


@pytest.fixture
def scenarios(equil_scenario):
    return list(all_scenarios(equil_scenario))


def test_direct_trajectories_match_event_based(scenarios):
    for name, scenario, config in scenarios:
        direct, from_events, _, _ = run_all_ways(scenario, **config)
        for field in ARRAYS:
            np.testing.assert_array_equal(
                getattr(direct, field), getattr(from_events, field), err_msg=f"{name}.{field}"
            )


def test_experienced_plans_match_event_collector(scenarios):
    for name, scenario, config in scenarios:
        direct, _, experienced, _ = run_all_ways(scenario, **config)
        assert direct.experienced_plans() == experienced, name


def test_stuck_cases_are_recorded(scenarios):
    by_name = {name: run_all_ways(s, **c)[0] for name, s, c in scenarios}
    assert by_name["gridlock"].stuck.all()
    assert by_name["end_mid_link"].stuck.all()
    assert np.isnan(by_name["end_mid_link"].leg_arrival).all()
    mixed = by_name["end_time"]
    assert mixed.stuck.sum() == 1 and np.isnan(mixed.leg_departure).sum() == 1
    walking = by_name["end_mid_teleport"]
    walk = walking.plans.modes.index("walk")
    walk_leg = int(np.flatnonzero(walking.plans.leg_mode == walk)[0])
    assert not np.isnan(walking.leg_departure[walk_leg])
    assert np.isnan(walking.leg_arrival[walk_leg])
    assert walking.leg_distance[walk_leg] == 0.0  # not arrived: no distance yet


@pytest.mark.parametrize("windows", [False, True])
def test_vectorised_scoring_matches_per_plan(scenarios, windows):
    """Also with every MATSim activity term: opening times, lateness, early
    departure, minimal durations, waiting, relative t_0 and daily constants."""
    H = 3600.0
    w = dict(opening_time=7 * H, closing_time=18 * H, latest_start_time=9 * H,
             earliest_end_time=16 * H, minimal_duration=6 * H) if windows else {}
    rel = dict(zero_utility="relative") if windows else {}
    acts = {"home": ActivityParams(12 * H, **rel), "work": ActivityParams(8 * H, **w, **rel),
            "shop": ActivityParams(3600, priority=2.0, **rel), "h": ActivityParams(12 * H),
            "w": ActivityParams(8 * H, **w)}
    extra = dict(early_departure=-3.0, waiting=-1.0,
                 modes={"car": ModeParams(daily_constant=-2.0, monetary_distance_rate=-1e-4)}
                 ) if windows else {}
    scorer = CharyparNagelScorer(ScoringParams(activities=acts, **extra))
    for name, scenario, config in scenarios:
        direct, *_ = run_all_ways(scenario, **config)
        vectorised = scorer.score_trajectories(direct)
        per_plan = [scorer.score(p) for p in direct.experienced_plans()]
        np.testing.assert_allclose(vectorised, per_plan, rtol=1e-12, atol=1e-9, err_msg=name)


def test_travel_time_table_matches_event_collector(scenarios):
    for name, scenario, config in scenarios:
        direct, _, _, from_events = run_all_ways(scenario, **config)
        table = direct.travel_time_table(scenario.network)
        np.testing.assert_allclose(table.times, from_events.times, rtol=1e-12, err_msg=name)


def test_events_are_optional(equil_scenario):
    scenario = routed(equil_scenario)
    quiet = make_mobsim().run(scenario, EventsManager())
    loud, *_ = run_all_ways(scenario)
    for field in ARRAYS:
        np.testing.assert_array_equal(getattr(quiet, field), getattr(loud, field))


def test_plan_batch_layout():
    net = chain_network([100, 100])
    a = Plan.builder().act("h", 0, end_time=0).leg("car", Route((0, 1), 20, 200)).act("w", 2).build()
    b = Plan.builder().act("w", 2, end_time=5).leg("walk").act("h", 1, duration=7) \
        .leg("car", Route((), 0, 0)).act("h", 1).build()
    batch = PlanBatch.from_plans([a, b])
    assert batch.act_offsets.tolist() == [0, 2, 5]
    assert batch.leg_offsets.tolist() == [0, 1, 3]
    assert [batch.act_types[c] for c in batch.act_type] == ["h", "w", "w", "h", "h"]
    assert [batch.modes[c] for c in batch.leg_mode] == ["car", "walk", "car"]
    assert batch.route_offsets.tolist() == [0, 2, 2, 2]
    assert batch.leg_routed.tolist() == [True, False, True]
    assert batch.leg_origin_act().tolist() == [0, 2, 3]
    assert np.isnan(batch.act_end_time[1]) and batch.act_duration[3] == 7
    assert batch.invalid_legs(net).tolist() == []
    with pytest.raises(ValueError):
        batch.act_node[0] = 5  # read-only


def test_trajectories_check_shapes():
    batch = PlanBatch.from_plans([Plan.builder().act("h", 0).build()])
    with pytest.raises(ValueError, match="act_start"):
        Trajectories(batch, np.zeros(2), np.zeros(1), np.zeros(0), np.zeros(0), np.zeros(0),
                     np.zeros(0), np.zeros(1, bool))


def test_travel_time_table_shares_identical_bins():
    net = chain_network([100, 200])
    table = TravelTimeTable.free_speed(net)
    first = table.link_travel_times(0.0)
    assert table.link_travel_times(20 * 3600.0) is first  # all bins equal: one tuple
    assert first == (10.0, 20.0)
    times = np.array(table.times)
    times[0, 3] = 99.0
    other = TravelTimeTable(times, table.bin_size)
    assert other.link_travel_times(3 * 900.0)[0] == 99.0
    assert other.link_travel_times(0.0) is other.link_travel_times(4 * 900.0)
    with pytest.raises(ValueError):
        other.times[0, 0] = 1.0  # read-only view
    times[0, 0] = 1.0  # ...but the caller's array is untouched
