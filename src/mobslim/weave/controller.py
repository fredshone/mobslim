"""Weave: parallel strands of innovation.

:class:`Weave` is an alternative to :class:`~mobslim.controller.Controller`.
Each iteration runs ``strands`` simulations of the same population, each
with a different cohort of persons innovating, and merges their results
into one global record, the :class:`~mobslim.weave.ledger.Ledger`::

    1. allocation  scheduler.allocate(ledger, K, rate)    -> K cohorts
    2. planning    per cohort: each innovator's strategy and source plan
    3. background  strand j: ChangeExpBeta (the selection-only strategies)
                   over the ledger, drawn with strand j's random stream
    4. loom        strand j, in its worker: mutate and route cohort j's new
                   plans, swap them into background j, simulate and score
                   -> observations + aggregates + new plans
    5. merge       add the new plans, reduce observations into the ledger, prune
    6. travel time sum the strands' aggregates, smooth into the estimate

Routing, the costly part of innovation, so runs inside the strands, in
parallel, rather than in this process before any strand starts.

With innovation rate ``r`` per strand, coverage is ``r * K`` innovations per
person per iteration, while each simulated world still has only a share
``r`` innovating. Iteration 0 runs a single strand: nobody innovates yet, so
every strand would simulate the same plans.

Each person's selected plan after a merge is the one they executed in strand
0 (whose events and trajectories the listeners see); ChangeExpBeta in the
next iteration's backgrounds starts from it. ``Weave(strands=1)`` therefore
behaves like ``Controller`` statistically (not draw for draw).

Random numbers are keyed by ``(seed, iteration, purpose, strand)``, so
results do not depend on the loom or on completion order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Union

import numpy as np

from mobslim.controller import Controller, RunResult
from mobslim.events import EventsManager
from mobslim.listeners import IterationState
from mobslim.metrics import iteration_metrics
from mobslim.mobsim._common import validate_population, without_validation
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.strategy import PersonStrategyManager, StrategyManager
from mobslim.replanning.vectorised import VectorRunner, manager_split
from mobslim.routing.compiled import router_threads
from mobslim.routing.traveltime import TravelTimeEstimator, sum_aggregates
from mobslim.weave.ledger import Ledger
from mobslim.weave.loom import Loom, ProcessLoom, SerialLoom, default_workers
from mobslim.weave.merge import MeanReducer, Observations, ScoreReducer, merge
from mobslim.weave.scheduler import (
    ConstantRate,
    InnovationSchedule,
    InnovationScheduler,
    StepRate,
    WeightedStalenessScheduler,
)
from mobslim.weave.strand import (
    StrandContext,
    StrandJob,
    StrandResult,
    run_strand,
    slim_jobs,
    strand_scenario,
)

logger = logging.getLogger("mobslim")


def weave_rng(seed: int, iteration: int, purpose: str, strand: int = 0) -> np.random.Generator:
    """The generator for one purpose of one strand of one iteration."""
    code = int.from_bytes(purpose.encode()[:8].ljust(8, b"\0"), "little")
    entropy = [seed % 2**63, iteration, code, strand]
    return np.random.Generator(np.random.Philox(np.random.SeedSequence(entropy)))


def innovation_share(strategies) -> float:
    """Share of the strategy weight on innovative strategies (persons
    weighted by their strategy set)."""
    def share(m: StrategyManager) -> float:
        total = sum(w for w, _ in m.strategies)
        return sum(w for w, s in m.strategies if s.is_innovative) / total if total else 0.0

    if isinstance(strategies, PersonStrategyManager):
        counts = np.bincount(strategies.codes, minlength=len(strategies.managers))
        return float(sum(c * share(m) for c, m in zip(counts, strategies.managers)) / max(counts.sum(), 1))
    return share(strategies)


def _disable_at(strategies) -> Optional[int]:
    managers = strategies.managers if isinstance(strategies, PersonStrategyManager) else (strategies,)
    ats = [m.disable_innovation_at for m in managers if m.disable_innovation_at is not None]
    return min(ats) if ats else None


@dataclass
class WeaveResult(RunResult):
    """:class:`~mobslim.controller.RunResult` plus each strand's metrics."""

    strand_metrics: list[list[dict]] = field(default_factory=list)  # per iteration, per strand


class Weave(Controller):
    """Runs the co-evolutionary loop with ``strands`` parallel simulations
    per iteration (see the module docs).

    ``Weave(scenario, strands=10, innovation_rate=0.1)`` is all that is
    needed; every :class:`~mobslim.controller.Controller` argument is
    accepted too. Specific to weaving:

    Args:
        strands: simulations per iteration (``K``).
        innovation_rate: share of persons innovating in each strand
            (default: the innovative strategies' share of the strategy
            weights, so existing configs carry over). Ignored with
            ``schedule``.
        schedule: the rate per iteration (an :class:`InnovationSchedule`);
            default constant, stepping to 0 at the strategies'
            ``disable_innovation_at``.
        scheduler: who innovates where (default
            :class:`WeightedStalenessScheduler`, ``alpha=1``).
        reducer: how strands' scores update the ledger (default
            :class:`MeanReducer` with ``params.learning_rate``).
        loom: where strands run. Default: in this process for one strand,
            else a :class:`ProcessLoom` with ``strands - 1`` workers, at most
            one fewer than the physical cores (strand 0 always runs in this
            process; strands beyond the workers queue).
        travel_time_smoothing: weight of each iteration's merged travel
            times in the estimate (1: only the latest, as ``Controller``).
    """

    def __init__(self, scenario, strands: int = 1, innovation_rate: Optional[float] = None, *,
                 schedule: Optional[InnovationSchedule] = None,
                 scheduler: Optional[InnovationScheduler] = None,
                 reducer: Optional[ScoreReducer] = None,
                 loom: Optional[Union[Loom, str]] = None,
                 travel_time_smoothing: float = 1.0,
                 **kwargs):
        runner = kwargs.pop("runner", None)
        if runner is not None and not isinstance(runner, VectorRunner):
            raise TypeError("Weave innovates with a VectorRunner.")
        super().__init__(scenario, runner=runner, **kwargs)
        if strands < 1:
            raise ValueError("strands must be >= 1.")
        if manager_split(self.strategies, len(scenario.population)) is None:
            raise TypeError("Weave needs a StrategyManager or PersonStrategyManager.")
        self.strands = strands
        if schedule is None:
            rate = innovation_share(self.strategies) if innovation_rate is None else innovation_rate
            at = _disable_at(self.strategies)
            schedule = ConstantRate(rate) if at is None else StepRate(rate, at)
        self.schedule = schedule
        self.scheduler = scheduler or WeightedStalenessScheduler()
        self.reducer = reducer or MeanReducer(self.params.learning_rate)
        if loom is None or loom == "auto":
            loom = SerialLoom() if strands == 1 else ProcessLoom(workers=default_workers(strands))
        elif loom == "serial":
            loom = SerialLoom()
        elif loom == "process":
            loom = ProcessLoom(workers=default_workers(max(strands, 2)))
        self.loom = loom
        self.travel_time_smoothing = travel_time_smoothing
        self.strand_metrics: list[list[dict]] = []
        self.ledger: Optional[Ledger] = None

    @classmethod
    def from_mobslim_config(cls, path, **overrides) -> "Weave":
        """As :meth:`Controller.from_mobslim_config`, plus the config's
        ``weave`` section."""
        from mobslim.config.mobslim import MobslimConfig

        cfg = path if isinstance(path, MobslimConfig) else MobslimConfig.load(path)
        kwargs = cfg.controller_kwargs()
        kwargs.update(cfg.weave_kwargs())
        return cls._from_kwargs(kwargs, overrides)

    # ----------------------------------------------------------------- loop
    def _removal_groups(self, n: int):
        managers, codes = manager_split(self.strategies, n)
        max_plans = np.array([m.max_plans for m in managers], dtype=np.int64)[codes]
        groups = [(m.removal_selector, np.flatnonzero(codes == c)) for c, m in enumerate(managers)]
        return max_plans, [(s, rows) for s, rows in groups if len(rows)]

    def _strand_context(self, scenario, transit_weights) -> StrandContext:
        """What every strand shares. Routes are validated here, every plan
        once, rather than by the mobsim in every strand and iteration (most
        plans are simulated again unchanged): strands then validate only
        their new plans."""
        cfg = self.params
        mobsim, validates = without_validation(self.mobsim)
        if validates:
            validate_population(scenario)
        return StrandContext(scenario, mobsim, self.scorer, cfg.travel_time_bins(),
                             cfg.travel_time_horizon, self.router_factory,
                             self.transit_router_factory, transit_weights,
                             validate_new=validates)

    def run(self) -> WeaveResult:
        cfg = self.params
        scenario = self.prepare(self.scenario)
        transit_weights = self.person_transit_weights(scenario)
        estimator = TravelTimeEstimator(scenario.network, cfg.travel_time_bins(),
                                        cfg.travel_time_horizon, self.travel_time_smoothing)
        travel_time = estimator.table()
        ledger = self.ledger = Ledger.start(scenario.population)
        n = len(ledger)
        max_plans, removal = self._removal_groups(n)
        runner = self.runner if isinstance(self.runner, VectorRunner) else VectorRunner()
        ctx = self._strand_context(scenario, transit_weights)
        self.metrics, self.strand_metrics = [], []
        stats: Optional[dict] = None
        self._notify("on_startup", self)
        try:
            self.loom.start(ctx)
            for it in range(cfg.iterations):
                t_start = time.perf_counter()
                scenario = scenario.with_population(ledger.population)
                state = IterationState(it, it == cfg.iterations - 1, scenario)
                self._notify("on_iteration_start", state)

                # 1-3: cohorts, planned innovations, backgrounds -> one job per strand
                k = self.strands if it > 0 else 1
                rate = self.schedule.rate(it, stats) if it > 0 else 0.0
                jobs, n_innovations = self._build_jobs(
                    ledger, scenario, travel_time, transit_weights, runner, it, k, rate)
                if it > 0:
                    state.scenario = scenario.with_population(ledger.population)
                    self._notify("on_after_replanning", state)
                t_replanned = time.perf_counter()

                # 4: strands 1.. to the loom, strand 0 here (with events and listeners)
                snapshot_s = getattr(self.loom, "snapshot_s", 0.0)
                futures = [self.loom.submit(job) for job in jobs[1:]]
                snapshot_s = getattr(self.loom, "snapshot_s", 0.0) - snapshot_s
                t_submitted = time.perf_counter()
                with router_threads(getattr(self.loom, "router_threads", None)):
                    local = self._run_local(ctx, jobs[0], state)
                t_local = time.perf_counter()
                results = [local] + [f.result() for f in futures]
                del jobs, futures, local
                state.strands = results
                t_simulated = time.perf_counter()

                # 5: merge
                obs = self._add_new_plans(ledger, results)
                merged = merge(ledger, obs, self.reducer, max_plans, removal,
                               weave_rng(cfg.seed, it, "prune"), selected_strand=0)
                ledger.in_flight[:] = False
                t_merged = time.perf_counter()
                scenario = state.scenario = scenario.with_population(ledger.population)
                self._notify("on_scoring", state)

                # 6: travel times (summed into strand 0's aggregate)
                t_scored = time.perf_counter()
                aggregate = sum_aggregates(r.aggregate for r in results)
                t_summed = time.perf_counter()
                travel_time = state.travel_time = estimator.observe(aggregate)
                t_observed = time.perf_counter()

                m = iteration_metrics(it, ledger.population, results[0].trajectories)
                m.update(
                    score_performed=float(obs.score.mean()) if len(obs) else float("nan"),
                    strands=k, innovation_rate=rate, innovations=n_innovations,
                    coverage=n_innovations / n if n else 0.0,
                    new_plans=merged.new_plans, pruned_plans=merged.pruned,
                    best_plan_changed=merged.best_changed,
                    stuck_all_strands=int(sum(r.metrics["stuck"] for r in results)),
                    replanning_s=t_replanned - t_start, simulation_s=t_simulated - t_replanned,
                    strand_replanning_s=float(max(r.metrics.get("strand_replanning_s", 0.0)
                                                  for r in results)),
                    merge_s=t_merged - t_simulated,
                    strand_s=float(max(r.seconds for r in results)),
                    # simulation_s = submit_s + local_strand_s + wait_s; submit_s
                    # includes snapshot_s (and, on a SerialLoom, strands 1..)
                    submit_s=t_submitted - t_replanned, snapshot_s=snapshot_s,
                    local_strand_s=t_local - t_submitted, wait_s=t_simulated - t_local,
                    on_scoring_s=t_scored - t_merged, aggregate_sum_s=t_summed - t_scored,
                    travel_time_s=t_observed - t_summed,
                    metrics_s=time.perf_counter() - t_observed,
                )
                m["iteration_s"] = time.perf_counter() - t_start
                state.metrics = stats = m
                self.metrics.append(m)
                self.strand_metrics.append([r.metrics for r in results])
                self._notify("on_iteration_end", state)
                del results, aggregate, obs  # free before the next iteration's strands
        finally:
            self.loom.close()
            self.runner.close()
            self._notify("on_shutdown", self)
        return WeaveResult(scenario, travel_time, self.metrics, self.strand_metrics)

    def _build_jobs(self, ledger: Ledger, scenario, travel_time, transit_weights,
                    runner: VectorRunner, it: int, k: int, rate: float):
        seed = self.params.seed
        pop = ledger.population
        n = len(pop)
        if k == 1 and rate == 0 and it == 0:
            rows = pop.selected_plan_ids().astype(np.int64)
            return [StrandJob(it, 0, seed, ledger.version, rows, np.zeros(n, dtype=bool),
                              store=pop.plans)], 0
        rctx = ReplanningContext.from_scenario(scenario, travel_time, it, self.router_factory,
                                               self.transit_router_factory, transit_weights)
        cohorts = (self.scheduler.allocate(ledger, k, rate, it, weave_rng(seed, it, "allocate"))
                   if rate > 0 else [np.zeros(0, dtype=np.int64)] * k)
        jobs = [self._strand_job(ledger, rctx, runner, it, j, cohort)
                for j, cohort in enumerate(cohorts)]
        everyone = np.concatenate(cohorts) if cohorts else np.zeros(0, dtype=np.int64)
        ledger.last_innovated[everyone] = it
        ledger.in_flight[everyone] = True
        if len(jobs) > 1 and getattr(self.loom, "workers", 0):  # workers load only what they use
            jobs = slim_jobs(jobs)
        return jobs, len(everyone)

    def _strand_job(self, ledger: Ledger, rctx: ReplanningContext, runner: VectorRunner,
                    it: int, j: int, cohort: np.ndarray) -> StrandJob:
        """Strand ``j``'s job: ``cohort``'s planned innovations, everyone
        else's plan drawn by the selection strategies (with strand ``j``'s
        random streams)."""
        seed = self.params.seed
        pop = ledger.population
        n = len(pop)
        innovations, _ = runner.plan_innovations(pop, self.strategies, rctx, cohort,
                                                 weave_rng(seed, it, "innovate", j))
        slots = runner.select(pop, self.strategies, rctx, weave_rng(seed, it, "background", j), seed)
        rows = pop.plan_ids[np.arange(n), slots].astype(np.int64)
        return StrandJob(it, j, seed, ledger.version, rows, np.zeros(n, dtype=bool),
                         store=pop.plans, innovations=innovations or None,
                         travel_time=rctx.travel_time)

    @staticmethod
    def _add_new_plans(ledger: Ledger, results: list[StrandResult]) -> Observations:
        """Append the strands' new plans to the ledger store, in strand
        order, and return all observations with their plans as ledger rows."""
        store = ledger.population.plans
        observations = []
        for r in results:
            obs = r.observations
            if r.new_plans is not None and r.new_plans.n_plans:
                offset = len(store)
                store = store.append(r.new_plans)
                new = obs.plan >= r.base_rows
                plan = obs.plan.copy()
                plan[new] += offset - r.base_rows
                obs = replace(obs, plan=plan)
            observations.append(obs)
        ledger.population = ledger.population.with_plan_store(store)
        return Observations.concat(observations)

    def _run_local(self, ctx: StrandContext, job: StrandJob, state: IterationState) -> StrandResult:
        """Strand ``job`` in this process, with the listeners' event handlers
        (and link times in the output iterations; workers never record them)."""
        job = replace(job, link_times=self.params.records_link_times(job.iteration))
        scenario = strand_scenario(ctx, job)
        state.scenario, state.strand = scenario, job.strand
        events = state.events = EventsManager()
        self._notify("on_before_mobsim", state)
        events.reset(job.iteration)
        result = run_strand(ctx, job, events, keep_trajectories=True, scenario=scenario)
        events.finish()
        state.trajectories = result.trajectories
        self._notify("on_after_mobsim", state)
        return result
