"""One strand: one simulated world of an iteration.

A strand is a vector of plan-store rows, one per person: the plan that
person executes in this world. :func:`strand_population` turns it into a
one-plan-per-person :class:`~mobslim.population.Population` (a view: the
store is shared, never compacted), which every mobsim reads through
``selected_batch()``. :func:`run_strand` simulates and scores it and returns
flat results: :class:`~mobslim.weave.merge.Observations` and a
:class:`~mobslim.routing.traveltime.TraversalAggregate` (sparse when it
crosses processes). Scoring runs where
the mobsim runs, so trajectories never cross processes.

A job may also carry its cohort's innovations, planned in the main process
(strategy and source plan per innovator, see
:meth:`~mobslim.replanning.VectorRunner.plan_innovations`). The strand then
mutates and routes them itself, before simulating, and returns the new plans
(:attr:`StrandResult.new_plans`) for the merge to add to the ledger. Routing,
the costly part of innovation, so runs in parallel across strands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional, Union

import numpy as np

from mobslim.events import EventsManager
from mobslim.metrics import iteration_metrics
from mobslim.mobsim._common import validate
from mobslim.mobsim.base import Mobsim, run_mobsim
from mobslim.planbatch import PlanBatch
from mobslim.planstore import PlanStore
from mobslim.population import Population
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.vectorised import innovate_batch
from mobslim.routing.compiled import default_router
from mobslim.routing.traveltime import SparseAggregate, TraversalAggregate
from mobslim.scenario import Scenario
from mobslim.scoring import Scorer, score_all
from mobslim.trajectories import Trajectories
from mobslim.weave.merge import BACKGROUND, CANDIDATE, Observations


@dataclass(frozen=True)
class StrandContext:
    """What every strand of a run shares (sent to each worker once).

    ``scenario``: network, modes, transit and the persons (ids, attributes);
    its population's plans are not used.
    """

    scenario: Scenario
    mobsim: Mobsim
    scorer: Scorer
    bin_size: object  # seconds per travel-time bin, or TimeBins
    horizon: float
    # routing innovations in the strand
    router_factory: object = default_router
    transit_router_factory: object = None
    transit_weights: Optional[object] = None
    # the run validated every plan up front and ``mobsim`` skips validation:
    # strands validate only their new plans
    validate_new: bool = False

    def replanning_context(self, travel_time, iteration: int) -> ReplanningContext:
        kw = {} if self.transit_router_factory is None else {
            "transit_router_factory": self.transit_router_factory}
        return ReplanningContext.from_scenario(self.scenario, travel_time, iteration,
                                               self.router_factory,
                                               transit_weights=self.transit_weights, **kw)

    def without_plans(self) -> "StrandContext":
        """This context with a plan-less population (cheap to pickle): one
        empty plan per person is not allowed, so the scenario keeps a
        one-plan store of the first person's plan, shared by everyone."""
        pop = self.scenario.population
        store = PlanStore.from_batch(pop.plans.take([int(pop.plan_ids[0, 0])]))
        tiny = Population.from_tables(pop.ids, store, np.zeros((len(pop), 1), dtype=np.int32),
                                      attributes=pop.attributes)
        return replace(self, scenario=self.scenario.with_population(tiny))


@dataclass(frozen=True)
class StrandJob:
    iteration: int
    strand: int
    seed: int
    basis: int  # ledger version the strand was built from
    rows: np.ndarray  # int64[n_persons]: each person's plan (a ledger store row)
    candidate: np.ndarray  # bool[n_persons]: the row is a new plan (innovation)
    store: Optional[PlanStore] = None  # the ledger store (in-process looms)
    batch: Optional[PlanBatch] = None  # or just the strand's plans, row i = person i
    # innovations to make here (VectorRunner.plan_innovations' first result,
    # source plans are rows of ``store``), with the travel times to route on;
    # their new plans replace ``rows`` at their persons
    innovations: Optional[list] = None
    travel_time: Optional[object] = None
    # with a store of only the rows the strands use (see slim_jobs): the
    # ledger row of each of its rows, and the ledger store's length
    row_map: Optional[np.ndarray] = None
    ledger_rows: int = 0
    # record the persons' link entry times (the controller's output
    # iterations, in its own process only)
    link_times: bool = False


@dataclass
class StrandResult:
    strand: int
    observations: Observations
    aggregate: Union[TraversalAggregate, SparseAggregate]  # sparse from other processes
    metrics: dict
    seconds: float  # innovation + mobsim + scoring + aggregation
    trajectories: Optional[Trajectories] = None  # in-process strands only
    # plans made by the strand's innovations; observation ``plan`` rows from
    # ``base_rows`` (the ledger store's length) on are these (row - base_rows)
    new_plans: Optional[PlanBatch] = None
    base_rows: int = 0


def strand_population(population: Population, store: PlanStore, rows) -> Population:
    """``population``'s persons, each with the single plan ``store[rows[i]]``."""
    rows = np.asarray(rows, dtype=np.int64)
    return Population.from_tables(population.ids, store, rows[:, None],
                                  attributes=population.attributes)


def run_strand(ctx: StrandContext, job: StrandJob, events: Optional[EventsManager] = None,
               keep_trajectories: bool = False, scenario: Optional[Scenario] = None,
               cross_process: bool = False) -> StrandResult:
    """Simulate and score one strand (``scenario``: the strand's scenario,
    if the caller already built it). ``cross_process``: the result will be
    sent to another process, so its aggregate is made sparse."""
    t0 = time.perf_counter()
    new_plans, base_rows, router_setup = None, 0, 0.0
    if job.innovations:
        base_rows = len(job.store)
        sim_job, job, new_plans, router_setup = innovate_strand(ctx, job)
        scenario = strand_scenario(ctx, sim_job)
        if ctx.validate_new:
            persons = np.flatnonzero(job.candidate)  # the innovators
            validate(scenario, sim_job.store.take(sim_job.rows[persons]), persons)
    t_innovated = time.perf_counter()
    if scenario is None:
        scenario = strand_scenario(ctx, job)
    own_events = events is None
    if own_events:
        events = EventsManager()
        events.reset(job.iteration)
    t_mobsim = time.perf_counter()
    trajectories = run_mobsim(ctx.mobsim, scenario, events, (ctx.bin_size, ctx.horizon),
                              link_times=job.link_times)
    if own_events:
        events.finish()
    t_simulated = time.perf_counter()
    pop = scenario.population
    scores = score_all(ctx.scorer, trajectories, pop.attributes)
    t_scored = time.perf_counter()
    aggregate = trajectories.traversal_aggregate(scenario.network, ctx.bin_size, ctx.horizon,
                                                 sparse=cross_process)
    t_aggregated = time.perf_counter()
    n = len(pop)
    plan = np.asarray(job.rows, dtype=np.int64)
    if job.row_map is not None:  # rows of a slimmed store -> ledger rows
        m = len(job.row_map)
        new = plan >= m
        plan = np.where(new, plan - m + job.ledger_rows, job.row_map[np.where(new, 0, plan)])
        if new_plans is not None:
            base_rows = job.ledger_rows
    obs = Observations(
        person=np.arange(n, dtype=np.int32),
        plan=plan,
        score=np.asarray(scores, dtype=float),
        role=np.where(job.candidate, CANDIDATE, BACKGROUND).astype(np.int8),
        strand=np.full(n, job.strand, dtype=np.int16),
        basis=np.full(n, job.basis, dtype=np.int32),
    )
    metrics = iteration_metrics(job.iteration, pop.with_selected_scores(scores), trajectories)
    metrics.update(strand=job.strand, innovators=int(np.count_nonzero(job.candidate)))
    for key in ("score_best", "score_worst", "score_average", "plans_per_person"):
        metrics.pop(key, None)  # one plan per person: these equal score_executed / 1
    seconds = time.perf_counter() - t0
    metrics.update(strand_s=seconds, strand_replanning_s=t_innovated - t0,
                   router_setup_s=router_setup, mobsim_s=t_simulated - t_mobsim,
                   scoring_s=t_scored - t_simulated, aggregate_s=t_aggregated - t_scored,
                   strand_metrics_s=time.perf_counter() - t_aggregated)
    return StrandResult(job.strand, obs, aggregate, metrics, seconds,
                        trajectories if keep_trajectories else None, new_plans, base_rows)


def innovate_strand(ctx: StrandContext, job: StrandJob):
    """Make ``job``'s innovations. Returns ``(sim_job, obs_job, new_plans,
    router_setup_s)``: ``sim_job`` simulates the strand's own small store
    (its plans, then the new ones), so the ledger store is never extended
    here; ``obs_job`` holds the rows to report, with innovator ``k``'s new
    plan numbered ``len(job.store) + k`` (see :attr:`StrandResult.base_rows`);
    ``router_setup_s``: seconds spent building routers."""
    persons = np.concatenate([p for _, p, _ in job.innovations])
    timed = _TimedFactory(ctx.router_factory)
    rctx = replace(ctx, router_factory=timed).replanning_context(job.travel_time, job.iteration)
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(
        [job.seed % 2**63, job.iteration, _MUTATE, job.strand])))
    new = innovate_batch(job.store, job.innovations, rctx, rng)
    n, base = len(job.rows), len(job.store)
    own = PlanStore.empty().append(job.store.take(job.rows)).append(new)  # 2nd: in place
    local = np.arange(n, dtype=np.int64)
    local[persons] = n + np.arange(len(persons))
    rows = np.array(job.rows, dtype=np.int64)
    rows[persons] = base + np.arange(len(persons))
    candidate = np.zeros(n, dtype=bool)
    candidate[persons] = True
    sim_job = replace(job, store=own, rows=local, innovations=None, travel_time=None)
    obs_job = replace(sim_job, store=None, rows=rows, candidate=candidate)
    return sim_job, obs_job, new, timed.seconds


_MUTATE = int.from_bytes(b"mutate", "little")


class _TimedFactory:
    """A router factory that adds up the time spent building routers."""

    def __init__(self, factory):
        self.factory, self.seconds = factory, 0.0

    def __call__(self, *args, **kwargs):
        t = time.perf_counter()
        router = self.factory(*args, **kwargs)
        self.seconds += time.perf_counter() - t
        return router


def slim_jobs(jobs: list[StrandJob]) -> list[StrandJob]:
    """``jobs``, all on the same ledger store, on one store of just the rows
    they use: every strand's plans and its innovations' source plans (about
    one plan per person, against up to ``max_plans`` in the ledger). Worker
    processes then load that instead of the whole ledger; results still
    report ledger rows."""
    store = jobs[0].store
    used = [np.asarray(j.rows, dtype=np.int64) for j in jobs]
    used += [np.asarray(src, dtype=np.int64) for j in jobs for s, _, src in j.innovations or ()
             if s is not None]
    row_map = np.unique(np.concatenate(used))
    sub = PlanStore.from_batch(store.take(row_map))

    def local(rows):
        return np.searchsorted(row_map, rows).astype(np.int64)

    return [replace(j, store=sub, rows=local(j.rows), row_map=row_map, ledger_rows=len(store),
                    innovations=None if j.innovations is None else [
                        (s, p, src if s is None else local(src)) for s, p, src in j.innovations])
            for j in jobs]


def strand_scenario(ctx: StrandContext, job: StrandJob) -> Scenario:
    """The scenario a strand simulates: its plans as the population."""
    if job.batch is not None:
        store = PlanStore.from_batch(job.batch)
        rows = np.arange(job.batch.n_plans)
    else:
        store, rows = job.store, job.rows
    base = ctx.scenario
    return base.with_population(strand_population(base.population, store, rows))
