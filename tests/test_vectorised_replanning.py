"""The vectorised replanning path applies the same rules as the per-person one.

Selectors and modules are checked decision by decision: both forms get the
same uniform draws (stub generators), so they must make the same choices.
Routing is deterministic, so batched and per-plan routes must be equal.
"""

import math
from bisect import bisect_right
from dataclasses import replace

import numpy as np
import pytest

from mobslim import PlanBatch, Population, Scenario, route_scenario
from mobslim.population import Activity, Leg, Person, Plan, PtRoute, Route
from mobslim.replanning import (
    BestScoreSelector,
    ChangeExpBetaSelector,
    ChangeTripMode,
    ExpBetaSelector,
    KeepSelected,
    PersonStrategyManager,
    PlanStrategy,
    RandomPlanSelector,
    ReplanningContext,
    ReRoute,
    SerialRunner,
    StrategyManager,
    TimeAllocationMutator,
    VectorRunner,
    WorstScoreSelector,
)
from mobslim.replanning.modules import (
    _clear_routes,
    clear_routes_batch,
    route_batch,
    route_plans,
)
from mobslim.routing import TravelTimeTable
from tests.test_planstore import batches_equal

# ------------------------------------------------------------ stub generators


class ScalarDraws:
    """A ``random.Random`` stand-in that maps each call to the next uniform
    the same way the vectorised code does."""

    def __init__(self, us):
        self.us = list(us)

    def random(self):
        return self.us.pop(0)

    def randrange(self, n):
        return min(int(self.random() * n), n - 1)

    def uniform(self, a, b):
        return a + (b - a) * self.random()

    def choice(self, seq):
        return seq[self.randrange(len(seq))]

    def choices(self, population, weights=None, cum_weights=None):
        if cum_weights is None:
            cum_weights = list(np.cumsum(weights))
        x = self.random() * cum_weights[-1]
        return [population[min(bisect_right(cum_weights, x), len(population) - 1)]]


class ArrayDraws:
    """A ``numpy.random.Generator`` stand-in returning preset uniforms."""

    def __init__(self, *arrays):
        self.arrays = [np.asarray(a, dtype=float) for a in arrays]

    def random(self, n):
        a = self.arrays.pop(0)
        assert len(a) == n
        return a

    def uniform(self, low, high, n):
        return low + (high - low) * self.random(n)


# ------------------------------------------------------------------ selectors

def random_tables(rng, m=200, k=5):
    n = rng.integers(1, k + 1, m)
    scores = rng.normal(0, 3, (m, k))
    scores[rng.random((m, k)) < 0.15] = np.nan  # some unscored
    scores[np.arange(k)[None, :] >= n[:, None]] = np.nan  # empty slots
    selected = (rng.random(m) * n).astype(int)
    return scores, n, selected


def person_of(scores_row, n, selected):
    plans = tuple(Plan((Activity("h", i),)) for i in range(n))
    return Person("p", plans, int(selected),
                  tuple(None if math.isnan(s) else float(s) for s in scores_row[:n]))


@pytest.mark.parametrize("selector, draws", [
    (KeepSelected(), 0), (BestScoreSelector(), 0), (RandomPlanSelector(), 1),
    (ExpBetaSelector(beta=0.7), 1), (ChangeExpBetaSelector(beta=2.0), 2),
    (ChangeExpBetaSelector(beta=200.0), 2), (WorstScoreSelector(), 1),
])
def test_select_many_matches_select(selector, draws):
    rng = np.random.default_rng(3)
    scores, n, selected = random_tables(rng)
    us = [rng.random(len(n)) for _ in range(draws)]
    many = selector.select_many(scores, n, selected, ArrayDraws(*us))
    for i in range(len(n)):
        person = person_of(scores[i], n[i], selected[i])
        own = [u[i] for u in us]
        expected = _scalar_select(selector, person, own)
        assert many[i] == expected, (type(selector).__name__, i)


def _scalar_select(selector, person, us):
    """``selector.select`` fed the person's draws (only those it uses, in order)."""
    unscored_first = not isinstance(selector, (KeepSelected, WorstScoreSelector))
    if unscored_first and any(s is None for s in person.scores):
        return selector.select(person, ScalarDraws([]))
    if isinstance(selector, ChangeExpBetaSelector):
        return selector.select(person, ScalarDraws(us))
    if isinstance(selector, WorstScoreSelector) and any(s is not None for s in person.scores):
        return selector.select(person, ScalarDraws([]))
    return selector.select(person, ScalarDraws(us))


# -------------------------------------------------------------------- modules

@pytest.fixture
def equil(equil_scenario):
    scenario = route_scenario(equil_scenario)
    table = TravelTimeTable.free_speed(scenario.network)
    return scenario, ReplanningContext.from_scenario(scenario, table, 1)


def test_time_allocation_mutator_batch(equil):
    scenario, ctx = equil
    plans = scenario.population.selected_plans()
    plans[3] = plans[3].with_elements(  # a duration-only first activity
        [replace(plans[3].elements[0], end_time=None, duration=100.0), *plans[3].elements[1:]])
    batch = PlanBatch.from_plans(plans)
    u = np.random.default_rng(1).random(batch.n_acts)
    tam = TimeAllocationMutator(mutation_range=900)
    out = tam.mutate_batch(batch, np.arange(len(plans)), ctx, ArrayDraws(u))
    offsets = batch.act_offsets.tolist()
    expected = [tam.mutate(p, ctx, ScalarDraws(u[offsets[k]:offsets[k + 1] - 1]))
                for k, p in enumerate(plans)]
    assert batches_equal(out, PlanBatch.from_plans(expected))


def test_change_trip_mode_batch(equil):
    scenario, _ = equil
    pop = scenario.population.with_attributes(
        {pid: {"car_ok": i % 3 != 0} for i, pid in enumerate(scenario.population.ids)})
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, pop),
                                          TravelTimeTable.free_speed(scenario.network), 1)
    plans = pop.selected_plans()
    plans[0] = Plan(plans[0].elements[:1])  # no legs: unchanged
    plans[1] = plans[1].with_elements([Leg("walk") if isinstance(e, Leg) else e
                                       for e in plans[1].elements])
    persons = np.arange(len(plans))
    u = np.random.default_rng(2).random(len(plans))
    for module in (ChangeTripMode(["car", "walk", "bike"]),
                   ChangeTripMode(["car", "walk"], ignore_car_availability=False,
                                  availability_attribute="car_ok")):
        out = module.mutate_batch(PlanBatch.from_plans(plans), persons, ctx, ArrayDraws(u))
        expected = [p if not p.legs else module.mutate(p, ctx.for_person(k), ScalarDraws([u[k]]))
                    for k, p in enumerate(plans)]
        assert _plans_of(out) == expected


def _plans_of(batch):
    from mobslim.planstore import PlanStore

    return PlanStore.from_batch(batch).to_plans()


def test_clear_routes_batch(equil):
    scenario, ctx = equil
    plans = scenario.population.selected_plans()
    # a walk leg with a stale car route, and a car leg with a pt route
    p = plans[0].elements
    plans[0] = plans[0].with_elements([*p[:1], Leg("walk", p[1].route), *p[2:]])
    plans[1] = plans[1].with_elements([*plans[1].elements[:1], Leg("car", PtRoute([], 1.0, 1.0)),
                                       *plans[1].elements[2:]])
    batch = PlanBatch.from_plans(plans)
    for keep in (True, False):
        expected = [_clear_routes(pl, ctx, keep) for pl in plans]
        assert _plans_of(clear_routes_batch(batch, ctx, keep)) == expected


def _mixed(plans):
    """Every third plan walks its second leg; every fifth drops its routes."""
    out = []
    for k, p in enumerate(plans):
        if k % 3 == 0 and len(p.elements) > 3:
            p = p.with_elements([Leg("walk") if i == 3 else e for i, e in enumerate(p.elements)])
        if k % 5 == 0:
            p = p.with_elements([Leg(e.mode) if isinstance(e, Leg) else e for e in p.elements])
        out.append(p)
    return out


def test_route_batch_equals_route_plans(equil):
    scenario, _ = equil
    free = TravelTimeTable.free_speed(scenario.network)
    slow = TravelTimeTable(free.times * np.linspace(1, 3, free.times.shape[1])[None, :],
                           free.bin_size)  # time-dependent: departure times matter
    plans = _mixed(scenario.population.selected_plans())
    for table in (free, slow):
        ctx = ReplanningContext.from_scenario(scenario, table, 1)
        assert batches_equal(route_batch(PlanBatch.from_plans(plans), ctx),
                             PlanBatch.from_plans(route_plans(plans, ctx)))


def test_route_batch_with_links_and_person_weights():
    from tests.test_transit_router_scoring import _pt_tutorial_split

    scenario, weights = _pt_tutorial_split()
    ctx = ReplanningContext.from_scenario(scenario, TravelTimeTable.free_speed(scenario.network), 1,
                                          transit_weights=weights)
    plans = scenario.population.selected_plans()
    persons = list(range(len(plans)))
    expected = route_plans(plans, ctx, [weights.of(i) for i in persons], persons)
    got = route_batch(PlanBatch.from_plans(plans), ctx, np.arange(len(plans)))
    assert batches_equal(got, PlanBatch.from_plans(expected))
    assert got.leg_pt_route.any()


# --------------------------------------------------------------------- runner

def scored(scenario, seed=0):
    rng = np.random.default_rng(seed)
    return scenario.population.with_selected_scores(rng.normal(size=len(scenario.population)))


@pytest.mark.parametrize("strategy", [
    PlanStrategy(KeepSelected(), [ReRoute()]),
    PlanStrategy(BestScoreSelector()),
    PlanStrategy(KeepSelected(), [ChangeTripMode(["car", "walk"]), ReRoute()]),
])
def test_runner_matches_serial_for_deterministic_choices(equil, strategy):
    """With one strategy and no random selection, both paths agree exactly
    (ChangeTripMode between two modes has one choice)."""
    scenario, _ = equil
    pop = scored(scenario)
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, pop),
                                          TravelTimeTable.free_speed(scenario.network), 2)
    for max_plans in (1, 2):
        manager = StrategyManager([(1.0, strategy)], max_plans=max_plans)
        a = SerialRunner().run(pop, manager, ctx, seed=5)
        b = VectorRunner().run(pop, manager, ctx, seed=5)
        assert list(a) == list(b)


class SwapToWalk:
    """A custom module (no ``mutate_batch``): runs per person."""

    def mutate(self, plan, ctx, rng):
        return plan.with_elements([Leg("walk") if isinstance(e, Leg) else e for e in plan.elements])


def test_runner_falls_back_per_person_for_custom_modules(equil):
    scenario, _ = equil
    pop = scored(scenario)
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, pop),
                                          TravelTimeTable.free_speed(scenario.network), 2)
    manager = StrategyManager([(1.0, PlanStrategy(RandomPlanSelector(), [SwapToWalk(), ReRoute()]))],
                              max_plans=2)
    assert list(VectorRunner().run(pop, manager, ctx, 9)) == list(SerialRunner().run(pop, manager, ctx, 9))


def test_runner_mixes_vectorised_and_per_person_strategies(equil):
    scenario, _ = equil
    pop = scored(scenario)
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, pop),
                                          TravelTimeTable.free_speed(scenario.network), 2)
    manager = StrategyManager([(0.5, PlanStrategy(KeepSelected(), [SwapToWalk()])),
                               (0.5, PlanStrategy(KeepSelected(), [TimeAllocationMutator(), ReRoute()]))])
    out = VectorRunner().run(pop, manager, ctx, 1)
    assert (out.n_plans == 2).all()
    walked = sum(all(leg.mode == "walk" for leg in p.selected_plan.legs) for p in out)
    assert 30 < walked < 70
    # every new plan is selected, unscored and routed
    assert np.isnan(out.selected_scores()).all()
    assert all(leg.route is not None for p in out.selected_plans() for leg in p.legs if leg.mode == "car")


def test_person_strategy_manager(equil):
    scenario, _ = equil
    pop = scored(scenario)
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, pop),
                                          TravelTimeTable.free_speed(scenario.network), 2)
    keep = StrategyManager([(1.0, PlanStrategy(KeepSelected()))])
    innovate = StrategyManager([(1.0, PlanStrategy(KeepSelected(), [ReRoute()]))])
    codes = np.arange(len(pop)) % 2
    out = VectorRunner().run(pop, PersonStrategyManager([keep, innovate], codes), ctx, 3)
    assert out.n_plans.tolist() == (1 + codes).tolist()


def test_runner_respects_plan_memory_over_many_iterations(equil):
    scenario, _ = equil
    net = scenario.network
    pop = scored(scenario)
    manager = StrategyManager.default(max_plans=3)
    rng = np.random.default_rng(0)
    n_innovated = 0
    for it in range(1, 30):
        ctx = ReplanningContext.from_scenario(Scenario(net, pop), TravelTimeTable.free_speed(net), it)
        before = len(pop.plans)
        pop = VectorRunner().run(pop, manager, ctx, seed=11)
        n_innovated += max(0, len(pop.plans) - before)
        assert (pop.n_plans <= 3).all() and (pop.n_plans >= 1).all()
        assert (pop.selected < pop.n_plans).all()
        assert len(pop.plans) <= 2 * int(pop.n_plans.sum()) + len(pop)  # compacted
        pop = pop.with_selected_scores(rng.normal(size=len(pop)))
    assert pop.plans.n_plans < 29 * len(pop)  # old plans were dropped


def test_strategy_shares_follow_weights(equil):
    scenario, _ = equil
    big = Population([Person(f"p{i}-{k}", (p,)) for i in range(20)
                      for k, p in enumerate(scenario.population.selected_plans())])
    big = big.with_selected_scores(np.zeros(len(big)))
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, big),
                                          TravelTimeTable.free_speed(scenario.network), 1)
    manager = StrategyManager([(0.8, PlanStrategy(KeepSelected())),
                               (0.2, PlanStrategy(KeepSelected(), [ReRoute()]))])
    out = VectorRunner().run(big, manager, ctx, 0)
    share = (out.n_plans == 2).mean()
    assert 0.17 < share < 0.23


def test_route_batch_handles_plans_without_legs(equil):
    _, ctx = equil
    plans = [Plan((Activity("h", 0),)),
             Plan((Activity("h", 0, end_time=1.0), Leg("car", Route([], 0.0, 0.0)), Activity("w", 0)))]
    batch = PlanBatch.from_plans(plans)
    assert route_batch(batch, ctx) is batch
