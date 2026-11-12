"""Weave: parallel strands of innovation (mobslim.weave)."""

import numpy as np
import pytest

from mobslim import (
    Controller,
    ControllerListener,
    ControllerParams,
    EventRecorder,
    Population,
    StrategyManager,
)
from mobslim.cli import main
from mobslim.events import EventsManager
from mobslim.mobsim import QSim
from mobslim.replanning import VectorRunner
from mobslim.replanning.context import ReplanningContext
from mobslim.routing.traveltime import (
    SparseAggregate,
    TravelTimeEstimator,
    TravelTimeTable,
    TraversalAggregate,
    sum_aggregates,
    travel_time_table,
)
from mobslim.weave import (
    BACKGROUND,
    CANDIDATE,
    AsyncWeave,
    ConstantRate,
    Ledger,
    LinearRate,
    MeanReducer,
    Observations,
    PrecisionReducer,
    ProcessLoom,
    SerialLoom,
    StepRate,
    Weave,
    WeightedStalenessScheduler,
    merge,
    weave_rng,
)
from tests.conftest import routed
from tests.test_cli import SCENARIOS


def _strip_timing(metrics):
    """Without timings and transfer sizes (which depend on the loom)."""
    return [{k: v for k, v in m.items() if not k.endswith(("_s", "_mb"))} for m in metrics]


@pytest.fixture
def scenario(equil_scenario):
    return routed(equil_scenario)


def _ledger(n: int) -> Ledger:
    from mobslim import Plan

    plan = Plan.builder().act("h", 0, end_time=0).leg("car").act("h", 0).build()
    return Ledger.start(Population.from_plans({i: plan for i in range(n)}))


# ------------------------------------------------------------------ allocation


@pytest.mark.parametrize("k, rate", [(1, 0.2), (4, 0.1), (4, 0.5), (8, 0.35), (3, 1.0)])
def test_allocation_counts_and_balance(k, rate):
    ledger = _ledger(200)
    cohorts = WeightedStalenessScheduler().allocate(ledger, k, rate, 3, np.random.default_rng(1))
    assert len(cohorts) == k
    for c in cohorts:  # at most one copy of a person per strand
        assert len(np.unique(c)) == len(c)
    sizes = [len(c) for c in cohorts]
    assert max(sizes) - min(sizes) <= 1
    counts = np.bincount(np.concatenate(cohorts), minlength=200)
    coverage = rate * k
    assert set(counts.tolist()) <= {int(np.floor(coverage)), int(np.ceil(coverage))}
    assert counts.sum() == int(np.floor(coverage)) * 200 + round((coverage % 1) * 200)


def test_allocation_skips_persons_in_flight():
    ledger = _ledger(50)
    ledger.in_flight[:10] = True
    cohorts = WeightedStalenessScheduler().allocate(ledger, 2, 1.0, 1, np.random.default_rng(0))
    everyone = np.concatenate(cohorts)
    assert not np.isin(np.arange(10), everyone).any()
    assert np.bincount(everyone, minlength=50)[10:].tolist() == [2] * 40


def test_staleness_priority_leaves_nobody_out():
    ledger = _ledger(100)
    rng = np.random.default_rng(3)
    strict = WeightedStalenessScheduler(alpha=50)
    for it in range(1, 6):  # 20% per iteration, oldest first: each person exactly once
        (cohort,) = strict.allocate(ledger, 1, 0.2, it, rng)
        assert (ledger.last_innovated[cohort] == -1).all()
        ledger.last_innovated[cohort] = it
    assert (ledger.last_innovated > 0).all()


def test_staleness_priority_prefers_stale_persons():
    ledger = _ledger(1000)
    ledger.last_innovated[:500] = 9  # innovated last iteration
    chosen = np.zeros(1000)
    sched = WeightedStalenessScheduler(alpha=1.0)
    for s in range(20):
        (cohort,) = sched.allocate(ledger, 1, 0.2, 10, np.random.default_rng(s))
        chosen[cohort] += 1
    # weight 2 (staleness 1) vs 12 (never): the never-innovated half dominates
    assert chosen[500:].sum() > 3 * chosen[:500].sum()
    uniform = np.zeros(1000)
    for s in range(20):
        (cohort,) = WeightedStalenessScheduler(alpha=0).allocate(
            ledger, 1, 0.2, 10, np.random.default_rng(s))
        uniform[cohort] += 1
    assert 0.8 < uniform[500:].sum() / uniform[:500].sum() < 1.25


def test_schedules():
    assert ConstantRate(0.1).rate(50) == 0.1
    step = StepRate(0.2, until=10)
    assert (step.rate(9), step.rate(10)) == (0.2, 0.0)
    lin = LinearRate(0.4, 0.0, 10, 20)
    assert (lin.rate(0), lin.rate(15), lin.rate(30)) == (0.4, pytest.approx(0.2), 0.0)
    with pytest.raises(ValueError):
        ConstantRate(1.5)


# ------------------------------------------------------------------ pieces


def test_traversal_aggregates_add(equil_scenario):
    net = equil_scenario.network
    rng = np.random.default_rng(0)
    links = rng.integers(0, net.n_links, 400).astype(np.int32)
    enter = rng.uniform(0, 3600 * 10, 400)
    leave = enter + rng.uniform(1, 600, 400)
    leave[::17] = np.nan
    whole = travel_time_table(net, links, enter, leave)
    a = TraversalAggregate.from_traversals(net.n_links, links[:150], enter[:150], leave[:150])
    b = TraversalAggregate.from_traversals(net.n_links, links[150:], enter[150:], leave[150:])
    assert np.array_equal((a + b).table(net).times, whole.times)
    assert (a + b).count.sum() == (~np.isnan(leave)).sum()


def test_sparse_aggregates_add(equil_scenario):
    net = equil_scenario.network
    rng = np.random.default_rng(1)
    links = rng.integers(0, net.n_links, 300).astype(np.int32)
    enter = rng.uniform(0, 3600 * 10, 300)
    leave = enter + rng.uniform(1, 600, 300)
    parts = [TraversalAggregate.from_traversals(net.n_links, links[s:s + 100], enter[s:s + 100],
                                                leave[s:s + 100]) for s in (0, 100, 200)]
    dense = parts[0] + parts[1] + parts[2]
    sparse = [p.to_sparse() for p in parts]
    assert isinstance(sparse[0], SparseAggregate) and len(sparse[0].cells) <= 100

    def same(got, want):
        return (np.array_equal(got.total, want.total) and np.array_equal(got.count, want.count)
                and np.array_equal(got.sum_sq, want.sum_sq))

    one = TraversalAggregate.empty(net.n_links, parts[0].n_bins)
    sparse[0].add_into(one)
    assert same(one, parts[0]) and same(sparse[0].to_dense(), parts[0])
    assert same(sum_aggregates(sparse), dense)
    summed = sum_aggregates([parts[0], sparse[1], sparse[2]])
    assert summed is parts[0] and same(summed, dense)  # in place, into the dense one
    with pytest.raises(ValueError):
        sparse[0].add_into(TraversalAggregate.empty(net.n_links, 3))


def test_estimator_smoothing(equil_scenario):
    net = equil_scenario.network
    agg = TraversalAggregate.from_traversals(net.n_links, np.array([0, 0], np.int32),
                                             np.array([0.0, 10.0]), np.array([1000.0, 1010.0]))
    free = TravelTimeTable.free_speed(net)
    latest = TravelTimeEstimator(net).observe(agg)
    assert latest.times[0, 0] == 1000.0
    smooth = TravelTimeEstimator(net, smoothing=0.5)
    # whole seconds, rounded up, from the unrounded smoothed estimate
    expected = 0.5 * free.times[0, 0] + 500.0
    assert smooth.observe(agg).times[0, 0] == np.ceil(expected)
    assert smooth.observe(agg).times[0, 0] == np.ceil(0.5 * expected + 500.0)  # no drift
    assert latest.times.dtype == np.uint16


def test_population_add_plans_and_with_scores(scenario):
    pop = scenario.population
    rows = pop.selected_plan_ids()[:3].astype(np.int64)
    out = pop.add_plans([0, 0, 2], rows, [1.0, 2.0, np.nan], select=True)
    assert out.n_plans[0] == pop.n_plans[0] + 2 and out.n_plans[2] == pop.n_plans[2] + 1
    assert out.selected[0] == out.n_plans[0] - 1
    assert out.scores[0, out.n_plans[0] - 1] == 2.0
    assert np.isnan(out.scores[2, out.n_plans[2] - 1])
    out = out.with_scores([2], [out.n_plans[2] - 1], [5.0])
    assert out.scores[2, out.n_plans[2] - 1] == 5.0


def test_vector_runner_innovate_and_select(scenario):
    pop = scenario.population
    ctx = ReplanningContext.from_scenario(scenario, TravelTimeTable.free_speed(scenario.network), 1)
    runner = VectorRunner()
    persons = np.array([0, 0, 5, 7])
    store, rows = runner.innovate(pop, StrategyManager.default(), ctx, persons,
                                  np.random.default_rng(0))
    assert len(store) == len(pop.plans) + 4
    assert sorted(rows.tolist()) == list(range(len(pop.plans), len(pop.plans) + 4))
    batch = store.take(rows)
    assert batch.leg_routed[np.isin(batch.leg_mode, [batch.modes.index("car")])].all()
    slots = runner.select(pop, StrategyManager.default(), ctx, np.random.default_rng(0))
    assert ((slots >= 0) & (slots < pop.n_plans)).all()


def test_merge_reduces_adds_and_prunes(scenario):
    pop = scenario.population
    ctx = ReplanningContext.from_scenario(scenario, TravelTimeTable.free_speed(scenario.network), 1)
    store, cand = VectorRunner().innovate(pop, StrategyManager.default(), ctx,
                                          np.array([0, 0, 1]), np.random.default_rng(0))
    ledger = Ledger.start(pop.with_plan_store(store))
    ledger.population = ledger.population.with_selected_scores(np.full(len(pop), 10.0))
    ledger.score_n[:, 0] = 1
    old = int(pop.plan_ids[0, 0])
    obs = Observations(
        person=np.array([0, 0, 0, 1, 0], np.int32),
        plan=np.array([old, old, cand[0], cand[2], cand[1]], np.int64),
        score=np.array([20.0, 30.0, 50.0, 1.0, 5.0]),
        role=np.array([BACKGROUND, BACKGROUND, CANDIDATE, CANDIDATE, CANDIDATE], np.int8),
        strand=np.array([0, 1, 0, 0, 1], np.int16),
        basis=np.zeros(5, np.int32),
    )
    max_plans = np.full(len(pop), 2)
    stats = merge(ledger, obs, MeanReducer(0.5), max_plans,
                  [(StrategyManager.default().removal_selector, np.arange(len(pop)))],
                  np.random.default_rng(0))
    p = ledger.population
    # person 0: old plan 0.5 * 10 + 0.5 * mean(20, 30) = 17.5; candidates 50 and 5;
    # pruned to 2: the 5 goes
    assert p.n_plans[0] == 2
    assert sorted(p.scores[0, :2].tolist()) == [17.5, 50.0]
    assert ledger.score_n[0, :2].tolist() == [3, 1] or ledger.score_n[0, :2].tolist() == [1, 3]
    assert p.scores[0, p.selected[0]] == 50.0  # what person 0 ran in strand 0
    assert p.n_plans[1] == 2 and p.scores[1, p.selected[1]] == 1.0
    assert p.scores[2, 0] == 10.0  # no samples: the score stays
    assert stats.new_plans == 3 and stats.pruned == 1
    assert ledger.version == 1


def test_precision_reducer_weighs_by_count(scenario):
    ledger = Ledger.start(scenario.population)
    ledger.population = ledger.population.with_selected_scores(np.full(len(ledger), 10.0))
    ledger.score_n[:, 0] = 3
    row = int(ledger.population.plan_ids[0, 0])
    obs = Observations(np.array([0], np.int32), np.array([row]), np.array([30.0]),
                       np.array([BACKGROUND], np.int8), np.zeros(1, np.int16), np.zeros(1, np.int32))
    upd = PrecisionReducer().reduce(obs, ledger)
    assert upd.value[0] == pytest.approx((3 * 10 + 30) / 4) and upd.n[0] == 4


def test_weave_rng_keys():
    a = weave_rng(1, 2, "background", 0).random()
    assert a == weave_rng(1, 2, "background", 0).random()
    assert a != weave_rng(1, 2, "background", 1).random()
    assert a != weave_rng(1, 2, "innovate", 0).random()
    assert a != weave_rng(1, 3, "background", 0).random()


# ------------------------------------------------------------------ runs


def test_one_strand_behaves_like_the_controller(equil_scenario):
    params = ControllerParams(iterations=15, seed=3)
    ctrl = [m["score_executed"] for m in Controller(equil_scenario, params=params).run().metrics]
    w = Weave(equil_scenario, strands=1, params=params).run()
    weave = [m["score_executed"] for m in w.metrics]
    assert weave[0] == ctrl[0]  # iteration 0: everyone runs the same initial plans
    assert w.metrics[-1]["coverage"] == pytest.approx(0.2)  # the default strategies' share
    # same dynamics: comparable gains, nothing like the 4-strand run's
    assert abs((weave[-1] - weave[0]) - (ctrl[-1] - ctrl[0])) < 1.0
    assert all(len(p.plans) <= 5 for p in w.scenario.population)


def test_more_strands_cover_more_persons(equil_scenario):
    params = ControllerParams(iterations=8, seed=3)
    one = Weave(equil_scenario, strands=1, innovation_rate=0.1, params=params).run()
    four = Weave(equil_scenario, strands=4, innovation_rate=0.1, loom=SerialLoom(),
                 params=params).run()
    assert four.metrics[-1]["coverage"] == pytest.approx(0.4)
    assert four.metrics[-1]["strands"] == 4 and four.metrics[0]["strands"] == 1
    assert len(four.strand_metrics[-1]) == 4
    assert four.metrics[-1]["score_executed"] > one.metrics[-1]["score_executed"]


def test_coverage_above_one(equil_scenario):
    res = Weave(equil_scenario, strands=4, innovation_rate=0.5, loom=SerialLoom(),
                params=ControllerParams(iterations=3, seed=1)).run()
    assert res.metrics[-1]["coverage"] == pytest.approx(2.0)
    assert all(m["innovators"] == 50 for m in res.strand_metrics[-1])


def test_weave_is_deterministic(equil_scenario):
    def run(seed):
        return Weave(equil_scenario, strands=3, innovation_rate=0.2, loom=SerialLoom(),
                     params=ControllerParams(iterations=4, seed=seed)).run()

    a, b, c = run(7), run(7), run(8)
    assert _strip_timing(a.metrics) == _strip_timing(b.metrics)
    assert _strip_timing(a.metrics) != _strip_timing(c.metrics)


def test_process_loom_matches_serial(equil_scenario):
    params = ControllerParams(iterations=3, seed=5)
    serial = Weave(equil_scenario, strands=3, loom=SerialLoom(), params=params).run()
    parallel = Weave(equil_scenario, strands=3, loom=ProcessLoom(2, start_method="fork"),
                     params=params).run()
    assert _strip_timing(serial.metrics) == _strip_timing(parallel.metrics)
    assert serial.strand_metrics == parallel.strand_metrics or all(
        _strip_timing(a) == _strip_timing(b)
        for a, b in zip(serial.strand_metrics, parallel.strand_metrics))


def test_strands_route_their_own_innovations(equil_scenario):
    """Innovations are made in the strands: every one comes back as a new
    plan, and fewer workers than strands (jobs queue) changes nothing."""
    params = ControllerParams(iterations=3, seed=2)
    serial = Weave(equil_scenario, strands=4, innovation_rate=0.2, loom=SerialLoom(),
                   params=params).run()
    queued = Weave(equil_scenario, strands=4, innovation_rate=0.2,
                   loom=ProcessLoom(1, start_method="fork"), params=params).run()
    assert _strip_timing(serial.metrics) == _strip_timing(queued.metrics)
    last = serial.metrics[-1]
    assert last["new_plans"] == last["innovations"] == 80
    assert all(m["strand_replanning_s"] > 0 for m in serial.strand_metrics[-1])


def test_default_workers_respect_physical_cores(monkeypatch):
    import mobslim.weave.loom as loom

    monkeypatch.setattr(loom, "physical_cores", lambda: 4)
    assert [loom.default_workers(k) for k in (2, 4, 8, 32)] == [1, 3, 3, 3]


def test_innovation_stops_at_disable_innovation_at(equil_scenario):
    res = Weave(equil_scenario, strands=2, loom=SerialLoom(),
                strategies=StrategyManager.default(disable_innovation_at=2),
                params=ControllerParams(iterations=4)).run()
    assert [m["innovations"] for m in res.metrics] == [0, 40, 0, 0]


def test_listeners_see_strand_zero(equil_scenario):
    calls = []

    class Spy(ControllerListener):
        def on_after_mobsim(self, state):
            calls.append((state.iteration, state.strand, len(state.trajectories.stuck)))

        def on_scoring(self, state):
            assert len(state.strands) == (1 if state.iteration == 0 else 2)

    rec = EventRecorder()
    Weave(equil_scenario, strands=2, loom=SerialLoom(), listeners=[Spy(), rec],
          mobsim=QSim(), params=ControllerParams(iterations=2)).run()
    assert calls == [(0, 0, 100), (1, 0, 100)]
    assert rec.last  # events of the last iteration, strand 0


def test_strand_population_is_a_view(scenario):
    from mobslim.weave import strand_population

    pop = scenario.population
    rows = pop.selected_plan_ids()[::-1].copy()
    view = strand_population(pop, pop.plans, rows)
    assert view.plans is pop.plans
    assert np.array_equal(view.selected_plan_ids(), rows)
    QSim().run(scenario.with_population(view), EventsManager())


def test_yaml_weave_section_and_cli(tmp_path, capsys):
    from mobslim.config.mobslim import MobslimConfig

    cfg = MobslimConfig.load(SCENARIOS / "equil" / "mobslim.yaml")
    assert not cfg.weaves
    assert main(["run", str(SCENARIOS / "equil" / "mobslim.yaml"), "-n", "3", "--strands", "2",
                 "-o", str(tmp_path), "-q"]) == 0
    assert "done: 3 iterations" in capsys.readouterr().out
    out = (tmp_path / "output_config.yaml").read_text()
    assert "strands: 2" in out
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0]
    assert "coverage" in header and "score_performed" in header


def test_weave_from_mobslim_config_kwargs():
    from mobslim.config.mobslim import MobslimConfig

    cfg = MobslimConfig.load(SCENARIOS / "equil" / "mobslim.yaml")
    cfg.values["weave"].update(strands=4, innovation_rate=0.05, workers=0, reducer="precision")
    w = Weave.from_mobslim_config(cfg)
    assert w.strands == 4 and isinstance(w.loom, SerialLoom)
    assert w.schedule.rate(1) == 0.05 and isinstance(w.reducer, PrecisionReducer)


# ------------------------------------------------------------------ async


def test_async_weave_is_deterministic_and_covers(equil_scenario):
    def run(seed, loom):
        return AsyncWeave(equil_scenario, strands=3, innovation_rate=0.2, loom=loom,
                          params=ControllerParams(iterations=5, seed=seed)).run()

    a, b, c = run(7, SerialLoom()), run(7, SerialLoom()), run(8, SerialLoom())
    assert _strip_timing(a.metrics) == _strip_timing(b.metrics)
    assert _strip_timing(a.metrics) != _strip_timing(c.metrics)
    assert [m["strands"] for m in a.metrics] == [1, 3, 3, 3, 3]
    assert a.metrics[-1]["coverage"] == pytest.approx(0.6)
    assert [len(s) for s in a.strand_metrics] == [1, 3, 3, 3, 3]
    assert a.metrics[-1]["score_executed"] > a.metrics[0]["score_executed"]
    assert all(len(p.plans) <= 5 for p in a.scenario.population)


def test_async_process_loom_matches_serial(equil_scenario):
    """Merges and builds happen in a fixed order: completion order (two
    workers for three slots) changes nothing."""
    params = ControllerParams(iterations=4, seed=5)
    serial = AsyncWeave(equil_scenario, strands=3, innovation_rate=0.2, loom=SerialLoom(),
                        params=params).run()
    parallel = AsyncWeave(equil_scenario, strands=3, innovation_rate=0.2,
                          loom=ProcessLoom(2, start_method="fork"), params=params).run()
    assert _strip_timing(serial.metrics) == _strip_timing(parallel.metrics)


def test_async_renumber():
    from mobslim.weave.asynchronous import _renumber

    live = np.array([2, 5, 9])
    assert _renumber(np.array([2, 3, 5, 9, -1, 10]), live).tolist() == [0, -1, 1, 2, -1, -1]
    assert _renumber(np.array([1, 2]), np.zeros(0, dtype=np.int64)).tolist() == [-1, -1]


def test_async_merges_stale_strands_onto_the_right_plans(equil_scenario, monkeypatch):
    """Strands in flight across other merges (which prune and compact the
    store) still score the plans they executed, or drop them if pruned."""
    import mobslim.weave.asynchronous as aw

    compacted, checked = [], [0]
    real_merge = aw.merge

    def spy_merge(*args, **kwargs):
        stats = real_merge(*args, **kwargs)
        compacted.append(stats.live is not None)
        return stats

    class Checked(AsyncWeave):
        def _submit(self, ledger, *args):
            slot = super()._submit(ledger, *args)
            slot.built = (ledger.population.plans, slot.rows.copy())  # what the strand runs
            return slot

        def _remapped(self, ledger, result, slot):
            obs = super()._remapped(ledger, result, slot)
            bg = obs.role == BACKGROUND
            now = ledger.population.plans.take(obs.plan[bg])
            store, rows = slot.built
            then = store.take(rows[obs.person[bg]])
            assert np.array_equal(now.route_links, then.route_links)
            assert np.array_equal(now.act_end_time, then.act_end_time, equal_nan=True)
            checked[0] += int(bg.sum())
            return obs

    monkeypatch.setattr(aw, "merge", spy_merge)
    res = Checked(equil_scenario, strands=4, innovation_rate=0.3, loom=SerialLoom(),
                  params=ControllerParams(iterations=12, seed=3)).run()
    assert any(compacted) and checked[0] > 0
    assert sum(m.get("dropped_observations", 0) for m in res.metrics) > 0


def test_async_pool_is_the_sum_of_the_last_k(equil_scenario):
    from mobslim.weave.asynchronous import _Pool

    net = equil_scenario.network
    rng = np.random.default_rng(4)
    parts = []
    for _ in range(7):
        links = rng.integers(0, net.n_links, 50).astype(np.int32)
        enter = np.floor(rng.uniform(0, 3600 * 10, 50))
        parts.append(TraversalAggregate.from_traversals(net.n_links, links, enter,
                                                        enter + np.floor(rng.uniform(1, 600, 50))))
    pool = _Pool(3)
    for i, part in enumerate(parts):
        got = pool.add(part)
        want = sum_aggregates([p.to_sparse() for p in parts[max(0, i - 2):i + 1]])
        assert np.array_equal(got.count, want.count) and np.array_equal(got.total, want.total)
        assert np.array_equal(got.sum_sq, want.sum_sq)


# ------------------------------------------------------------------ validation


def test_weave_validates_every_plan_up_front():
    """An invalid route is found before the first iteration, even in a plan
    that is not selected (strands may select it later)."""
    from mobslim import Person, Plan, Route, Scenario
    from mobslim.mobsim import SimulationError
    from tests.conftest import chain_network

    net = chain_network([100, 100])
    good = Plan.builder().act("h", 0, end_time=0).leg("car", Route((0, 1), 20, 200)).act("w", 2).build()
    bad = Plan.builder().act("h", 0, end_time=0).leg("car", Route((1,), 10, 100)).act("w", 2).build()
    pop = Population([Person("a", (good,)), Person("b", (good, bad), selected=0)])
    weave = Weave(Scenario(net, pop), strands=2, innovation_rate=0.5, loom=SerialLoom(),
                  params=ControllerParams(iterations=2))
    with pytest.raises(SimulationError, match="'b': invalid 'car' route"):
        weave.run()


def test_strands_validate_only_their_new_plans(scenario, monkeypatch):
    from mobslim.weave import strand

    seen, validate = [], strand.validate

    def spy(sc, plans, persons=None):
        seen.append((plans.n_plans, len(sc.population)))
        validate(sc, plans, persons)

    monkeypatch.setattr(strand, "validate", spy)
    result = Weave(scenario, strands=3, innovation_rate=0.2, loom=SerialLoom(),
                   params=ControllerParams(iterations=4, seed=1)).run()
    innovators = [m["innovators"] for ms in result.strand_metrics[1:] for m in ms]
    assert [n for n, _ in seen] == innovators
    assert all(n < people for n, people in seen)


def test_without_validation_copies_the_mobsim():
    from mobslim.mobsim._common import without_validation

    mobsim = QSim()
    quiet, validated = without_validation(mobsim)
    assert validated and quiet is not mobsim
    assert mobsim.params.validate_routes and not quiet.params.validate_routes
    again, validated = without_validation(quiet)
    assert again is quiet and not validated

