from random import Random

import pytest

from mobslim import Person, Plan, Scenario
from mobslim.replanning import (
    BestScoreSelector,
    ChangeExpBetaSelector,
    ChangeTripMode,
    ExpBetaSelector,
    KeepSelectedSelector,
    PlanStrategy,
    ProcessPoolRunner,
    RandomSelector,
    ReplanningContext,
    Reroute,
    SerialRunner,
    StrategyManager,
    TimeAllocationMutator,
    WorstScoreSelector,
    person_rng,
)
from mobslim.routing import TravelTimeTable
from tests.conftest import routed


def scored_person(*scores, selected=0):
    plans = tuple(
        Plan.builder().act("h", i, end_time=0).leg("car").act("w", 0).build()
        for i in range(len(scores))
    )
    return Person("p", plans, selected, scores)


@pytest.fixture
def ctx(equil_scenario):
    return ReplanningContext.from_scenario(
        equil_scenario, TravelTimeTable.free_speed(equil_scenario.network), iteration=1
    )


def test_selectors():
    rng = Random(0)
    p = scored_person(1.0, 5.0, 3.0)
    assert BestScoreSelector().select(p, rng) == 1
    assert WorstScoreSelector().select(p, rng) == 0
    assert KeepSelectedSelector().select(p, rng) == 0
    assert RandomSelector().select(scored_person(1.0, None), rng) == 1  # unscored first
    picks = [ExpBetaSelector(beta=100).select(p, rng) for _ in range(20)]
    assert set(picks) == {1}


def test_change_exp_beta():
    rng = Random(0)
    # much better alternative: always switch; much worse: never switch
    assert all(ChangeExpBetaSelector(beta=10).select(scored_person(0.0, 100.0), rng) == 1 for _ in range(20))
    assert all(ChangeExpBetaSelector(beta=10).select(scored_person(100.0, 0.0), rng) == 0 for _ in range(20))
    assert ChangeExpBetaSelector().select(scored_person(1.0), rng) == 0


def test_reroute(equil_scenario, ctx):
    plan = equil_scenario.population[0].selected_plan
    new = Reroute().mutate(plan, ctx, Random(0))
    assert plan.legs[0].route is None  # input untouched
    assert all(leg.route is not None for leg in new.legs)
    assert new.legs[1].route.links == ()  # work -> work at the same node


def test_time_allocation_mutator(equil_scenario, ctx):
    plan = equil_scenario.population[0].selected_plan
    new = TimeAllocationMutator(mutation_range=600).mutate(plan, ctx, Random(1))
    old_end, new_end = plan.activities[0].end_time, new.activities[0].end_time
    assert new_end != old_end and abs(new_end - old_end) <= 600
    assert new.activities[-1] == plan.activities[-1]


def test_change_trip_mode(equil_scenario, ctx):
    plan = Reroute().mutate(equil_scenario.population[0].selected_plan, ctx, Random(0))
    new = ChangeTripMode(["car", "walk"]).mutate(plan, ctx, Random(0))
    assert {leg.mode for leg in new.legs} == {"walk"}
    rerouted = Reroute().mutate(new, ctx, Random(0))
    assert all(leg.route is None for leg in rerouted.legs)  # teleported


def test_innovative_strategy_adds_unscored_selected_plan(ctx, equil_scenario):
    person = routed(equil_scenario).population[0]
    person = person.with_selected_score(1.0)
    out = PlanStrategy(KeepSelectedSelector(), [TimeAllocationMutator()]).apply(person, ctx, Random(0))
    assert len(out.plans) == 2 and out.selected == 1 and out.scores == (1.0, None)
    assert len(person.plans) == 1  # pure


def test_strategy_manager_respects_memory_limit(ctx, equil_scenario):
    person = routed(equil_scenario).population[0]
    person = person.with_selected_score(-1.0)
    manager = StrategyManager([(1.0, PlanStrategy(RandomSelector(), [TimeAllocationMutator()]))], max_plans=3)
    for i in range(10):
        person = manager.replan(person, ctx, Random(i))
        person = person.with_selected_score(float(i))
    assert len(person.plans) == 3
    assert min(person.scores) == 7.0  # worst plans were removed


def test_innovation_switch_off():
    manager = StrategyManager.default(disable_innovation_at=5)
    rng = Random(0)
    assert any(manager.choose(1, rng).is_innovative for _ in range(100))
    assert not any(manager.choose(5, rng).is_innovative for _ in range(100))


def test_person_rng_is_stable():
    assert person_rng(1, 2, 3).random() == person_rng(1, 2, 3).random()
    assert person_rng(1, 2, 3).random() != person_rng(1, 2, 4).random()


def test_parallel_runner_matches_serial(equil_scenario):
    scenario = routed(equil_scenario)
    pop = scenario.population.with_selected_scores([0.0] * len(scenario.population))
    ctx = ReplanningContext.from_scenario(Scenario(scenario.network, pop),
                                          TravelTimeTable.free_speed(scenario.network), iteration=3)
    manager = StrategyManager.default()
    serial = SerialRunner().run(pop, manager, ctx, seed=42)
    with ProcessPoolRunner(workers=2) as runner:
        parallel = runner.run(pop, manager, ctx, seed=42)
    assert list(serial) == list(parallel)
    assert list(serial) != list(pop)  # something was replanned


def test_parallel_runner_picks_up_a_new_context_each_call(equil_scenario):
    """Workers cache the context per call; a second call must not reuse the first."""
    scenario = routed(equil_scenario)
    pop = scenario.population.with_selected_scores([0.0] * len(scenario.population))
    sc = Scenario(scenario.network, pop)
    free = TravelTimeTable.free_speed(scenario.network)
    slow = TravelTimeTable(free.times * 3, free.bin_size)
    manager = StrategyManager([(1.0, PlanStrategy(RandomSelector(), [Reroute()]))])
    with ProcessPoolRunner(workers=2, chunks_per_worker=1) as runner:
        for it, table in ((1, free), (2, slow)):
            ctx = ReplanningContext.from_scenario(sc, table, iteration=it)
            serial = SerialRunner().run(pop, manager, ctx, seed=7)
            assert list(runner.run(pop, manager, ctx, seed=7)) == list(serial)


def test_strategy_manager_is_immutable_and_validates():
    import dataclasses

    manager = StrategyManager.default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        manager.max_plans = 2
    only_innovative = StrategyManager([(1.0, PlanStrategy(RandomSelector(), [Reroute()]))],
                                      disable_innovation_at=1)
    with pytest.raises(ValueError):
        only_innovative.choose(1, Random(0))
