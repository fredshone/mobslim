import pickle

import numpy as np
import pytest

from mobslim.planbatch import PlanBatch
from mobslim.planstore import PlanStore
from mobslim.population import (
    Activity,
    Leg,
    Person,
    Plan,
    Population,
    PtRoute,
    Route,
    validate_batch,
)


def sample_plans():
    return [
        Plan((Activity("home", 0, end_time=100.0, x=1.5, y=2.5), Leg("car", Route([1, 2], 30.0, 200.0)),
              Activity("work", 3, duration=50.0, link=2), Leg("walk", travel_time=60.0),
              Activity("home", 0))),
        Plan((Activity("home", 4),)),
        Plan((Activity("school", 1, end_time=7.0), Leg("pt", PtRoute([(3, 5), (9, 11)], 99.0, 1000.0)),
              Activity("home", 2, end_time=8.0, duration=1.0), Leg("car"),
              Activity("shop", 2))),
        Plan((Activity("home", 1, end_time=5.0), Leg("car", Route([], 0.0, 0.0)), Activity("home", 1))),
    ]


def batches_equal(a: PlanBatch, b: PlanBatch) -> bool:
    return (a.act_types == b.act_types and a.modes == b.modes
            and all(np.array_equal(getattr(a, k), getattr(b, k), equal_nan=True)
                    for k in PlanBatch._ARRAYS))


def test_store_round_trips_plans():
    plans = sample_plans()
    store = PlanStore.from_plans(plans)
    assert len(store) == 4
    assert list(store) == plans
    assert store.to_plans() == plans
    assert store[-1] == plans[-1]
    # a view builds its elements once, from the columns
    view = store[2]
    assert view.elements is view.elements
    assert view.legs[0].route.rides == ((3, 5), (9, 11)) and view.legs[0].route.travel_time == 99.0


@pytest.mark.parametrize("rows", [[0, 1, 2, 3], [2, 0], [3, 3, 1], []])
def test_take_equals_encoding_the_plans(rows):
    plans = sample_plans()
    store = PlanStore.from_plans(plans)
    assert batches_equal(store.take(rows), PlanBatch.from_plans([plans[r] for r in rows]))


def test_append_at_the_tip_shares_buffers_and_keeps_old_snapshots():
    plans = sample_plans()
    a = PlanStore.from_plans(plans[:2])
    b = a.append(PlanBatch.from_plans(plans[2:3]))
    c = b.append(PlanBatch.from_plans(plans[3:]))  # grows: same lineage
    assert a.token == b.token == c.token
    assert len(a) == 2 and list(a) == plans[:2]  # never sees the rows past its end
    assert list(c) == plans
    assert c.modes == ("car", "walk", "pt")  # vocab only grows


def test_append_to_an_older_snapshot_branches():
    plans = sample_plans()
    a = PlanStore.from_plans(plans[:2])
    tip = a.append(PlanBatch.from_plans(plans[2:3]))
    branch = a.append(PlanBatch.from_plans(plans[3:]))
    assert branch.token != a.token and tip.token == a.token
    assert list(tip) == plans[:3]
    assert list(branch) == plans[:2] + plans[3:]


def test_store_pickles_its_rows_only():
    store = PlanStore.from_plans(sample_plans()).append(PlanBatch.from_plans(sample_plans()[:1]))
    copy = pickle.loads(pickle.dumps(store))
    assert list(copy) == list(store) and copy.token != store.token
    assert len(copy._arrays["act_type"]) == store.n_acts  # no spare capacity


def test_plan_views_are_frozen_and_hashable():
    from dataclasses import FrozenInstanceError

    store = PlanStore.from_plans(sample_plans())
    with pytest.raises(FrozenInstanceError):
        store[0].elements = ()
    assert hash(store[0]) == hash(sample_plans()[0])
    assert store[0] == store[0] and store[0] != store[1]


def test_replace_appends_only_new_plans():
    plans = sample_plans()
    pop = Population.from_plans({i: p for i, p in enumerate(plans)})
    same = pop.replace(list(pop))
    assert same.plans.token == pop.plans.token and len(same.plans) == 4
    new = Plan((Activity("home", 9),))
    persons = list(pop)
    persons[1] = persons[1].add_plan(new)
    grown = pop.replace(persons)
    assert len(grown.plans) == 5 and grown.plan_ids.tolist()[1] == [1, 4]
    assert grown[1].selected_plan == new
    # the plan just stored is recognised next time (no re-append)
    again = grown.replace([p if i != 1 else persons[1] for i, p in enumerate(grown)])
    assert len(again.plans) == 5


def test_pickled_views_are_recognised():
    pop = Population.from_plans({i: p for i, p in enumerate(sample_plans())})
    persons = pickle.loads(pickle.dumps(list(pop)))  # as the process-pool runner does
    assert len(pop.replace(persons).plans) == 4


def test_views_of_another_population_are_stored_once():
    pop = Population.from_plans({"a": sample_plans()[0]})
    other = Population([Person("x", pop[0].plans), Person("y", pop[0].plans)])
    assert len(other.plans) == 1 and other.plan_ids.tolist() == [[0], [0]]


def test_compaction_drops_unused_rows_and_keeps_plans():
    plans = sample_plans()
    pop = Population.from_plans({i: p for i, p in enumerate(plans)})
    for k in range(10):  # each round every person swaps in a fresh plan
        persons = [Person(p.id, (Plan((Activity("home", k),)),)) if p.id == 0 else p
                   for p in pop]
        pop = pop.replace(persons)
        assert len(pop.plans) <= 2 * 4
    assert pop[0].selected_plan == Plan((Activity("home", 9),))
    assert pop.selected_plans()[1:] == plans[1:]
    assert sorted(pop.live_plan_rows().tolist()) == sorted(pop.plan_ids.ravel().tolist())


def test_map_plan_store_with_unused_rows():
    plans = sample_plans()
    pop = Population.from_plans({i: p for i, p in enumerate(plans)})
    persons = list(pop)
    persons[0] = Person(0, (Plan((Activity("home", 7),)),))
    pop = pop.replace(persons)
    assert len(pop.plans) == 5 and pop.live_plan_rows().tolist() == [1, 2, 3, 4]
    renamed = pop.map_plans(lambda p: p.with_elements(
        [Activity("x", e.node, e.end_time, e.duration) if isinstance(e, Activity) else e
         for e in p.elements]))
    assert len(renamed.plans) == 4
    assert [p.activities[0].node for p in renamed.selected_plans()] == [7, 4, 1, 1]
    assert renamed.live_batch().act_types == ("x",)


def test_from_tables_checks_shape():
    store = PlanStore.from_plans(sample_plans())
    pop = Population.from_tables(["a", "b"], store, [[0, 2], [1, -1]], selected=[1, 0])
    assert pop.selected_plans() == [sample_plans()[2], sample_plans()[1]]
    assert np.isnan(pop.selected_scores()).all()
    with pytest.raises(ValueError):
        Population.from_tables(["a"], store, [[-1, 0]])  # gap before a plan
    with pytest.raises(ValueError):
        Population.from_tables(["a"], store, [[0]], selected=[1])


def test_validate_batch():
    validate_batch(PlanBatch.from_plans(sample_plans()))
    b = PlanBatch.from_plans(sample_plans())
    bad = PlanBatch(**{**{k: getattr(b, k) for k in PlanBatch._ARRAYS},
                       "act_types": b.act_types, "modes": b.modes,
                       "act_end_time": np.full(b.n_acts, np.nan),
                       "act_duration": np.full(b.n_acts, np.nan)})
    with pytest.raises(ValueError, match="needs an end_time or duration"):
        validate_batch(bad)
