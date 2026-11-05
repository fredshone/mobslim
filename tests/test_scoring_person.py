"""Person-specific scoring: parameter sets, overrides and rules."""

from dataclasses import replace

import numpy as np
import pytest

from mobslim import Plan, Population, Scenario
from mobslim.attributes import PersonAttributes
from mobslim.events import EventsManager
from mobslim.scoring import (
    ActivityParams,
    CharyparNagelScorer,
    FromAttributes,
    IncomeDependentMoney,
    ModeConstants,
    ModeParams,
    PersonParams,
    ScoringParams,
    Subpopulation,
    numeric_column,
    parse_field,
    score_all,
)
from tests.conftest import chain_network, commuters, make_mobsim, routed
from tests.test_trajectories import all_scenarios

H = 3600.0


def _params(**kw):
    acts = {"home": ActivityParams(12 * H), "work": ActivityParams(8 * H, opening_time=7 * H,
                                                                   latest_start_time=9 * H),
            "shop": ActivityParams(H, priority=2.0), "h": ActivityParams(12 * H),
            "w": ActivityParams(8 * H, zero_utility="relative")}
    modes = {"car": ModeParams(monetary_distance_rate=-1e-4, daily_constant=-1.0),
             "walk": ModeParams(marginal_utility_of_traveling=-12.0),
             "pt": ModeParams(constant=-0.5, marginal_utility_of_distance=-1e-5)}
    return ScoringParams(**{"activities": acts, "modes": modes, **kw})


def _rules():
    def age(a):
        return numeric_column(a, "age")

    return [
        IncomeDependentMoney(),
        ModeConstants(),
        FromAttributes("modes.walk.marginal_utility_of_traveling",
                       lambda a, cur: np.where(age(a) > 70, 2 * cur, np.nan)),
        FromAttributes("activities.work.typical_duration",
                       lambda a, cur: np.where(age(a) > 50, 6 * H, np.nan)),
        FromAttributes("performing", lambda a, cur: np.where(age(a) < 30, 8.0, np.nan)),
        FromAttributes("utility_of_line_switch", lambda a, cur: np.where(age(a) > 60, -5.0, np.nan)),
    ]


def _scorer():
    return CharyparNagelScorer(
        _params(),
        subpopulations={"a": _params(performing=4.0, late_arrival=-30.0),
                        "b": _params(performing=9.0, walk_mode="walk",
                                     marginal_utility_of_money=0.5)},
        person_rules=_rules(),
    )


def _attributes(n, seed=0):
    rng = np.random.default_rng(seed)
    sub = [None if r < 0.3 else ("a" if r < 0.6 else "b") for r in rng.random(n)]
    income = [None if r < 0.2 else float(v) for r, v in zip(rng.random(n), rng.uniform(500, 5000, n))]
    asc_car = [None if r < 0.5 else float(v) for r, v in zip(rng.random(n), rng.normal(0, 1, n))]
    return PersonAttributes(n, {"subpopulation": sub, "income": income, "asc_car": asc_car,
                                "asc_walk": rng.normal(0, 1, n).tolist(),
                                "age": rng.integers(18, 90, n).tolist()})


def _assert_matches_reference(scorer, tr, attributes, name=""):
    vectorised = scorer.score_trajectories(tr, attributes=attributes)
    pp = scorer.person_params(attributes)
    per_plan = [scorer.score(p, params=pp.params_of(i)) for i, p in enumerate(tr.experienced_plans())]
    np.testing.assert_allclose(vectorised, per_plan, rtol=1e-12, atol=1e-9, err_msg=name)
    return vectorised


def test_person_specific_scores_match_per_plan_reference(equil_scenario):
    for name, scenario, config in all_scenarios(equil_scenario):
        tr = make_mobsim(**config).run(scenario, EventsManager())
        attrs = _attributes(tr.plans.n_plans)
        scores = _assert_matches_reference(_scorer(), tr, attrs, name)
        if name == "equil":  # the rules change something
            plain = CharyparNagelScorer(_params()).score_trajectories(tr)
            assert not np.allclose(scores, plain)


def test_person_specific_pt_scores_match_per_plan_reference():
    from tests.test_transit_scenarios import load, run

    scenario = routed(load("pt_lineswitch"))
    tr = run(scenario)
    attrs = _attributes(tr.plans.n_plans, seed=3)
    rules = _rules() + [FromAttributes("modes.pt.marginal_utility_of_traveling",
                                       lambda a, cur: np.where(numeric_column(a, "age") > 40, -20.0, np.nan)),
                        FromAttributes("marginal_utility_of_waiting_pt",
                                       lambda a, cur: np.where(numeric_column(a, "age") < 40, -1.0, np.nan))]
    scorer = CharyparNagelScorer(_params(), subpopulations={"a": _params(walk_mode="car"),
                                                            "b": _params()}, person_rules=rules)
    _assert_matches_reference(scorer, tr, attrs)


def test_scores_without_rules_are_unchanged_by_attributes(equil_scenario):
    tr = make_mobsim().run(routed(equil_scenario), EventsManager())
    attrs = _attributes(tr.plans.n_plans)
    scorer = CharyparNagelScorer(_params())
    np.testing.assert_array_equal(scorer.score_trajectories(tr, attrs), scorer.score_trajectories(tr))


def _commute(n=2, **kw):
    net = chain_network([1000, 1000])
    scenario = routed(Scenario(net, commuters(n, 0, 2, dep=8 * H)))
    return make_mobsim().run(scenario, EventsManager())


def test_income_scales_money_terms():
    tr = _commute()
    priced = _params(modes={"car": ModeParams(monetary_distance_rate=-1e-3)})
    free = _params(modes={"car": ModeParams()})
    attrs = PersonAttributes(2, {"income": [1000.0, 2000.0]})
    rule = [IncomeDependentMoney(reference=2000.0)]
    money = (CharyparNagelScorer(priced, person_rules=rule).score_trajectories(tr, attrs)
             - CharyparNagelScorer(free).score_trajectories(tr))
    assert money[1] == pytest.approx(-1e-3 * 2000)  # 2 km at reference income
    assert money[0] == pytest.approx(2 * money[1])  # half the income: money counts twice
    # elasticity and the population mean as reference (1500)
    rule = [IncomeDependentMoney(elasticity=0.5)]
    money = (CharyparNagelScorer(priced, person_rules=rule).score_trajectories(tr, attrs)
             - CharyparNagelScorer(free).score_trajectories(tr))
    np.testing.assert_allclose(money, -2.0 * np.sqrt(1500 / np.array([1000, 2000])))


def test_missing_income_keeps_the_sets_money():
    priced = _params(modes={"car": ModeParams(monetary_distance_rate=-1e-3)})
    attrs = PersonAttributes(3, {"income": [None, 1000.0, -5.0]})
    pp = CharyparNagelScorer(priced, person_rules=[IncomeDependentMoney()]).person_params(attrs)
    np.testing.assert_allclose(pp.value("marginal_utility_of_money"), [1.0, 1.0, 1.0])
    attrs = PersonAttributes(3, {"income": [None, 1000.0, 4000.0]})
    pp = CharyparNagelScorer(priced, person_rules=[IncomeDependentMoney()]).person_params(attrs)
    np.testing.assert_allclose(pp.value("marginal_utility_of_money"), [1.0, 2.5, 0.625])


def test_mode_constants_add_per_leg():
    tr = _commute(3)
    attrs = PersonAttributes(3, {"asc_car": [1.5, None, -2.0], "asc_bike": [9.0, 9.0, 9.0]})
    base = CharyparNagelScorer(_params()).score_trajectories(tr)
    scores = CharyparNagelScorer(_params(), person_rules=[ModeConstants()]).score_trajectories(tr, attrs)
    np.testing.assert_allclose(scores - base, [1.5, 0.0, -2.0])  # one car leg each, no bike


def test_rules_compose_in_order():
    attrs = PersonAttributes(3, {"subpopulation": [None, "slow", "slow"], "age": [20, 20, 80]})
    slow = _params(modes={"walk": ModeParams(marginal_utility_of_traveling=-10.0)})
    field = "modes.walk.marginal_utility_of_traveling"
    old = FromAttributes(field, lambda a, cur: np.where(numeric_column(a, "age") > 70, cur * 3, np.nan))
    scorer = CharyparNagelScorer(_params(), subpopulations={"slow": slow}, person_rules=[old])
    pp = scorer.person_params(attrs)
    np.testing.assert_allclose(pp.value(field), [-12.0, -10.0, -30.0])
    assert pp.params_of(2).mode("walk").marginal_utility_of_traveling == -30.0
    assert pp.params_of(0) is scorer.params  # nothing overridden: the set itself
    # an explicit Subpopulation rule replaces the automatic one, and runs where placed
    late = CharyparNagelScorer(_params(), subpopulations={"slow": slow},
                               person_rules=[old, Subpopulation()])
    np.testing.assert_allclose(late.person_params(attrs).value(field), [-12.0, -10.0, -36.0])


def test_activity_overrides_do_not_create_params():
    tr = _commute(1)
    params = ScoringParams(activities={"work": ActivityParams(8 * H)}, default_activity=None)
    rule = FromAttributes("activities.home.typical_duration", lambda a, cur: np.full(a.n_persons, 10 * H))
    scorer = CharyparNagelScorer(params, person_rules=[rule])
    with pytest.raises(KeyError, match="home"):
        scorer.score_trajectories(tr, PersonAttributes(1))
    assert np.isnan(scorer.person_params(PersonAttributes(1)).value("activities.home.typical_duration"))


def test_activity_override_recomputes_zero_utility_duration():
    tr = _commute(2)
    rule = FromAttributes("activities.work.priority", lambda a, cur: np.array([np.nan, 4.0]))
    scorer = CharyparNagelScorer(_params(), person_rules=[rule])
    scores = scorer.score_trajectories(tr, PersonAttributes(2))
    other = CharyparNagelScorer(replace(_params(), activities={
        **_params().activities, "work": replace(_params().activities["work"], priority=4.0)}))
    assert scores[1] == pytest.approx(other.score_trajectories(tr)[1])
    assert scores[0] == pytest.approx(CharyparNagelScorer(_params()).score_trajectories(tr)[0])


def test_mode_overrides_follow_mode_names_not_batch_codes():
    """Plan batches code modes in order of first appearance."""
    net = chain_network([1000, 1000])
    plans = {0: Plan.builder().act("home", 0, end_time=8 * H).leg("walk").act("work", 2).build(),
             1: Plan.builder().act("home", 0, end_time=8 * H).leg("car").act("work", 2).build()}
    tr = make_mobsim().run(routed(Scenario(net, Population.from_plans(plans))), EventsManager())
    assert tr.plans.modes == ("walk", "car")
    attrs = PersonAttributes(2, {"asc_car": [5.0, 5.0], "asc_walk": [1.0, 1.0]})
    base = CharyparNagelScorer(_params()).score_trajectories(tr)
    scores = CharyparNagelScorer(_params(), person_rules=[ModeConstants()]).score_trajectories(tr, attrs)
    np.testing.assert_allclose(scores - base, [1.0, 5.0])


def test_subpopulation_cost_does_not_depend_on_attribute_order(equil_scenario):
    tr = make_mobsim().run(routed(equil_scenario), EventsManager())
    n = tr.plans.n_plans
    subs = {f"s{i}": _params(performing=4.0 + i) for i in range(8)}
    scorer = CharyparNagelScorer(_params(), subpopulations=subs)
    attrs = PersonAttributes(n, {"subpopulation": [f"s{i % 8}" for i in range(n)]})
    scores = scorer.score_trajectories(tr, attrs)
    for i in range(8):
        alone = CharyparNagelScorer(subs[f"s{i}"]).score_trajectories(tr)
        np.testing.assert_allclose(scores[i::8], alone[i::8])


def test_person_params_are_cached_per_attribute_table(equil_scenario):
    scorer = CharyparNagelScorer(_params(), person_rules=[ModeConstants()])
    attrs = _attributes(5)
    assert scorer.person_params(attrs) is scorer.person_params(attrs)
    assert scorer.person_params(_attributes(5)) is not scorer.person_params(attrs)
    with pytest.raises(ValueError):
        scorer.person_params(attrs, n_persons=6)


def test_per_plan_score_with_attribute_row():
    tr = _commute(2)
    attrs = PersonAttributes(2, {"asc_car": [2.0, None]})
    scorer = CharyparNagelScorer(_params(), person_rules=[ModeConstants()])
    plans = tr.experienced_plans()
    per_plan = score_all(type("PerPlan", (), {"score": scorer.score})(), tr, attrs)
    np.testing.assert_allclose(per_plan, scorer.score_trajectories(tr, attrs))
    assert scorer.score(plans[0], attributes={"asc_car": 2.0}) == pytest.approx(per_plan[0])


def test_transit_weights_from_person_params():
    from mobslim.transit.raptor import RaptorWeights

    attrs = PersonAttributes(4, {"age": [20, 80, 85, 30]})
    rule = FromAttributes("modes.walk.marginal_utility_of_traveling",
                          lambda a, cur: np.where(numeric_column(a, "age") > 70, -30.0, np.nan))
    scorer = CharyparNagelScorer(_params(), person_rules=[rule])
    w = scorer.transit_weights(attrs, "pt")
    assert w.codes.tolist() == [-1, 0, 0, -1]
    assert w.of(1) == RaptorWeights.from_scoring(replace(_params(), modes={
        **_params().modes, "walk": ModeParams(marginal_utility_of_traveling=-30.0)}), "pt")
    # rules that don't touch routing: no weights
    assert CharyparNagelScorer(_params(), person_rules=[ModeConstants()]).transit_weights(
        PersonAttributes(4, {"asc_car": [1.0] * 4}), "pt") is None


def test_field_paths():
    assert str(parse_field("performing")) == "performing"
    assert parse_field("modes.car.constant").name == "car"
    assert parse_field("activities.work.v2.typical_duration").name == "work.v2"
    for bad in ("perfoming", "modes.car.speed", "activities.work", "modes..constant"):
        with pytest.raises(KeyError):
            parse_field(bad)


def test_person_params_validation():
    p = ScoringParams()
    with pytest.raises(ValueError):
        PersonParams((p,), (None,), [0, 1])
    with pytest.raises(ValueError):
        PersonParams((p,), (None,), [0, 0], {"performing": [1.0]})
    pp = PersonParams((p,), (None,), [0, 0], {"performing": [np.nan, np.nan]})
    assert pp.is_uniform and not pp.overrides
    with pytest.raises(ValueError, match="numeric"):
        numeric_column(PersonAttributes(1, {"income": ["lots"]}), "income")
