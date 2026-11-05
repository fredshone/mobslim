import math

import pytest

from mobslim import ActivityParams, ModeParams, Scenario, ScoringParams
from mobslim.events import EventLog, EventsManager
from mobslim.experience import (
    ExperiencedActivity,
    ExperiencedLeg,
    ExperiencedPlan,
    ExperiencedPlanCollector,
)
from mobslim.mobsim import QSim, QSimParams
from mobslim.scoring import CharyparNagelScorer, Scorer
from tests.conftest import chain_network, commuters, routed

H = 3600.0


def scorer():
    return CharyparNagelScorer(
        ScoringParams(
            activities={"home": ActivityParams(12 * H), "work": ActivityParams(8 * H)},
            modes={"car": ModeParams(constant=-1.0)},
        )
    )


def test_is_a_scorer():
    assert isinstance(scorer(), Scorer)


def test_activity_at_typical_duration_scores_ten_hours_of_performing():
    s = scorer()
    assert s.activity_utility("work", 8 * H) == pytest.approx(60.0)
    assert s.activity_utility("work", 9 * H) > s.activity_utility("work", 8 * H)
    assert s.activity_utility("work", 0) < 0  # linear below t_0


def test_leg_utility():
    leg = ExperiencedLeg("car", 0, H, 1000, ())
    assert scorer().leg_utility(leg) == pytest.approx(-1.0 - 6.0)


def test_overnight_wrap_combines_first_and_last():
    s = scorer()
    plan = ExperiencedPlan((
        ExperiencedActivity("home", 0, None, 8 * H),
        ExperiencedLeg("car", 8 * H, 9 * H, 0, ()),
        ExperiencedActivity("work", 1, 9 * H, 17 * H),
        ExperiencedLeg("car", 17 * H, 18 * H, 0, ()),
        ExperiencedActivity("home", 0, 18 * H, None),
    ))
    expected = (
        s.activity_utility("home", 14 * H)
        + s.activity_utility("work", 8 * H)
        + 2 * (-1 - 6)
    )
    assert s.score(plan) == pytest.approx(expected)


def test_stuck_penalty():
    plan = ExperiencedPlan((ExperiencedActivity("home", 0, None, None),), stuck=True)
    s = scorer()
    assert s.score(plan) == pytest.approx(s.activity_utility("home", 24 * H) - 100)


def test_unknown_activity_type_without_default_raises():
    s = CharyparNagelScorer(ScoringParams(default_activity=None))
    with pytest.raises(KeyError):
        s.activity_utility("shop", H)


def test_experienced_plans_from_mobsim():
    net = chain_network([100, 100], freespeed=10)
    scenario = routed(Scenario(net, commuters(2, 0, 2, dep=50)))
    collector = ExperiencedPlanCollector(net, scenario.population)
    em = EventsManager([collector, EventLog()])
    QSim(QSimParams()).run(scenario, em)
    em.finish()
    plan = collector.plans()[0]
    home, leg, work = plan.elements
    assert home == ExperiencedActivity("home", 0, None, 50)
    assert leg.departure_time == 50 and leg.travel_time == 20
    assert leg.distance == 200 and leg.links == (0, 1)
    assert work.start == 70 and work.end is None
    assert not plan.stuck


def test_apply_scores_checks_length(equil_scenario):
    from mobslim.scoring import apply_scores

    with pytest.raises(ValueError):
        apply_scores(equil_scenario.population, [1.0])


# ------------------------------------------- MATSim's calcActScore terms
# Expected values are MATSim 2025.0's CharyparNagelActivityScoring formulas,
# written out by hand. verification/ checks the whole scorer against MATSim.


def _perf(beta, t_typ, t_0, dur):
    """MATSim's performing utility (hours), with the linear part below t_0."""
    if dur >= t_0:
        return beta * t_typ * math.log(dur / t_0)
    return beta * t_typ / t_0 * (dur - t_0)


def test_zero_utility_duration_uniform_and_relative():
    uniform = ActivityParams(8 * H, priority=2.0)
    relative = ActivityParams(8 * H, priority=2.0, zero_utility="relative")
    assert uniform.zero_utility_duration == pytest.approx(8 * H * math.exp(-10 / 8 / 2))
    assert relative.zero_utility_duration == pytest.approx(8 * H * math.exp(-1 / 2))
    s = CharyparNagelScorer(ScoringParams(activities={"w": relative}))
    t_0 = 8 * math.exp(-0.5)
    assert s.activity_utility("w", 8 * H) == pytest.approx(_perf(6, 8, t_0, 8))
    assert s.activity_utility("w", 1 * H) == pytest.approx(_perf(6, 8, t_0, 1))  # linear part
    with pytest.raises(ValueError):
        ActivityParams(H, zero_utility="other")


def test_opening_times_waiting_and_late_arrival():
    """Arrive 06:30, leave 19:00 at an activity open 07:00-18:00 that should
    start by 06:45: performs 07:00-18:00 (11 h), waits 1.5 h, is late 15 min."""
    w = ActivityParams(8 * H, opening_time=7 * H, closing_time=18 * H, latest_start_time=6.75 * H)
    s = CharyparNagelScorer(ScoringParams(activities={"w": w}, waiting=-2.0, late_arrival=-18.0))
    t_0 = w.zero_utility_duration / H
    expected = _perf(6, 8, t_0, 11) - 2.0 * 1.5 - 18.0 * 0.25
    assert s.activity_score("w", 6.5 * H, 19 * H) == pytest.approx(expected)


def test_closed_all_the_time_is_not_performed():
    w = ActivityParams(8 * H, opening_time=7 * H, closing_time=18 * H)
    s = CharyparNagelScorer(ScoringParams(activities={"w": w}, waiting=-1.0))
    t_0 = w.zero_utility_duration / H
    # 19:00-20:00, after closing: zero duration (linear part), and waits the hour
    assert s.activity_score("w", 19 * H, 20 * H) == pytest.approx(_perf(6, 8, t_0, 0) - 1.0)


def test_early_departure_and_minimal_duration():
    """09:00-13:00 at an activity that should last 6 h and not end before
    16:00: 3 h early, 2 h short, both priced at early_departure."""
    w = ActivityParams(8 * H, earliest_end_time=16 * H, minimal_duration=6 * H)
    s = CharyparNagelScorer(ScoringParams(activities={"w": w}, early_departure=-4.0))
    t_0 = w.zero_utility_duration / H
    assert s.activity_score("w", 9 * H, 13 * H) == pytest.approx(_perf(6, 8, t_0, 4) - 4.0 * 5)


def test_overnight_wrap_after_midnight_is_not_clamped():
    """As MATSim: the wrapped home activity runs from last.start to
    first.end + 24 h, even when the agent gets home after midnight."""
    s = scorer()
    plan = ExperiencedPlan((
        ExperiencedActivity("home", 0, None, 8 * H),
        ExperiencedLeg("car", 8 * H, 9 * H, 0, ()),
        ExperiencedActivity("work", 1, 9 * H, 25 * H),
        ExperiencedLeg("car", 25 * H, 26 * H, 0, ()),
        ExperiencedActivity("home", 0, 26 * H, None),
    ))
    expected = s.activity_utility("home", 6 * H) + s.activity_utility("work", 16 * H) + 2 * (-7)
    assert s.score(plan) == pytest.approx(expected)


def test_daily_constants_and_money_per_meter():
    params = ScoringParams(
        activities={"home": ActivityParams(12 * H)}, marginal_utility_of_money=2.0,
        modes={"car": ModeParams(marginal_utility_of_traveling=0.0, daily_constant=-3.0,
                                 daily_monetary_constant=-1.0, monetary_distance_rate=-0.001)})
    s = CharyparNagelScorer(params)
    legs = (ExperiencedLeg("car", 8 * H, 8 * H, 1000.0, ()), ExperiencedLeg("car", 9 * H, 9 * H, 500.0, ()))
    plan = ExperiencedPlan((
        ExperiencedActivity("home", 0, None, 8 * H), legs[0],
        ExperiencedActivity("home", 0, 8 * H, 9 * H), legs[1],
        ExperiencedActivity("home", 0, 9 * H, None),
    ))
    base = CharyparNagelScorer(ScoringParams(activities={"home": ActivityParams(12 * H)}))
    acts_only = base.score(ExperiencedPlan(plan.activities))  # no legs: same activity part
    # money per meter: -0.001 * 2 * 1500 m; daily: -3 + -1 * 2, once (one mode)
    assert s.score(plan) - acts_only == pytest.approx(-3.0 - 5.0)


def test_unpriced_nan_distance_does_not_spoil_the_score():
    """MATSim's pt walks have distance NaN; it only counts if priced."""
    leg = ExperiencedLeg("walk", 0, H, math.nan, ())
    assert scorer().leg_utility(leg) == pytest.approx(-6.0)
