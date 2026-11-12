"""AsyncWeave: the weaving loop without waiting for the slowest strand.

:class:`~mobslim.weave.Weave` waits for all its strands each iteration, so
cores idle while the slowest one finishes, and again while the main process
merges and plans. :class:`AsyncWeave` keeps ``K`` strand slots busy instead.
Slots ``0..K-1`` cycle in a fixed order; the main process waits for slot
``j``'s result only, merges it into the ledger, updates the travel times,
builds slot ``j``'s next job from the ledger as it now stands, submits it,
and moves to slot ``j + 1``::

    round r:  wait j=0 -> merge -> travel times -> build + submit (r+1, 0)
              wait j=1 -> merge -> travel times -> build + submit (r+1, 1)
              ...

Because merges and builds happen in a fixed order, the ledger at each build
does not depend on completion order: the same seed gives the same result on
any loom. A round of ``K`` merges is one "iteration" for metrics and
listeners.

What differs from :class:`~mobslim.weave.Weave`:

* A strand's results are merged after ``K - 1`` other strands' merges since
  its job was built. Its background plans may have been pruned meanwhile
  (those observations are dropped: metric ``dropped_observations``), and
  the store may have been compacted (its rows are renumbered). Its observations' ``basis`` records how stale
  they are.
* Travel times: each merge pools the last ``K`` strands' traversals (as an
  iteration of ``Weave`` pools its ``K`` strands) and smooths that into the
  estimate, so ``travel_time_smoothing`` applies per merge.
* Every strand from iteration 1 on runs in the loom (by default a
  :class:`~mobslim.weave.ProcessLoom` with one worker per slot, up to the
  physical cores), so listeners see events and trajectories in iteration 0
  only (``on_before_mobsim`` / ``on_after_mobsim``); later iterations give
  them the merged ledger (``on_scoring``, ``on_iteration_end``) and the
  round's strand results (``state.strands``, without trajectories).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from mobslim.listeners import IterationState
from mobslim.metrics import iteration_metrics, score_metrics
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.vectorised import VectorRunner
from mobslim.routing.compiled import router_threads
from mobslim.routing.traveltime import (
    TravelTimeEstimator,
    TraversalAggregate,
    sum_aggregates,
)
from mobslim.weave.controller import Weave, WeaveResult, weave_rng
from mobslim.weave.ledger import Ledger
from mobslim.weave.loom import ProcessLoom, logical_cpus, physical_cores
from mobslim.weave.merge import CANDIDATE, Observations, merge
from mobslim.weave.strand import StrandResult, slim_jobs

_STRAND_ONLY = ("strand", "innovators")  # strand metrics not averaged over a round


class _InFlight:
    """A submitted strand: its future, its cohort, and its background plans
    as current ledger rows (renumbered when a merge compacts the store;
    ``-1``: pruned)."""

    def __init__(self, future, cohort: np.ndarray, rows: np.ndarray):
        self.future, self.cohort, self.rows = future, cohort, rows


class AsyncWeave(Weave):
    """The ordered asynchronous weaving loop (see the module docs). Takes
    every :class:`~mobslim.weave.Weave` argument; the default loom is a
    :class:`~mobslim.weave.ProcessLoom` with ``min(strands, physical
    cores)`` workers (the main process only merges and plans)."""

    def __init__(self, scenario, strands: int = 2, innovation_rate: Optional[float] = None,
                 **kwargs):
        loom = kwargs.pop("loom", None)
        if loom is None or loom == "auto" or loom == "process":
            workers = max(1, min(strands, physical_cores()))
            # the main process runs no strand: the workers share every core
            loom = ProcessLoom(workers=workers, router_threads=max(1, logical_cpus() // workers))
        super().__init__(scenario, strands, innovation_rate, loom=loom, **kwargs)

    def run(self) -> WeaveResult:
        cfg = self.params
        k = self.strands
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
        pool = _Pool(k)  # the last K strands' traversals
        slots: list[Optional[_InFlight]] = [None] * k
        stats: Optional[dict] = None
        self._notify("on_startup", self)
        try:
            self.loom.start(ctx)
            # iteration 0: everyone's initial plans, one strand, here (as Weave)
            t_start = time.perf_counter()
            state = IterationState(0, cfg.iterations == 1, scenario)
            self._notify("on_iteration_start", state)
            jobs, _ = self._build_jobs(ledger, scenario, travel_time, transit_weights, runner,
                                       0, 1, 0.0)
            with router_threads(getattr(self.loom, "router_threads", None)):
                result = self._run_local(ctx, jobs[0], state)
            state.strands = [result]
            obs = self._add_new_plans(ledger, [result])
            merged = merge(ledger, obs, self.reducer, max_plans, removal,
                           weave_rng(cfg.seed, 0, "prune"), selected_strand=0)
            scenario = state.scenario = scenario.with_population(ledger.population)
            self._notify("on_scoring", state)
            travel_time = state.travel_time = estimator.observe(pool.add(result.aggregate))
            m = iteration_metrics(0, ledger.population, result.trajectories)
            m.update(score_performed=float(obs.score.mean()) if len(obs) else float("nan"),
                     strands=1, innovation_rate=0.0, innovations=0, coverage=0.0,
                     new_plans=merged.new_plans, pruned_plans=merged.pruned,
                     best_plan_changed=merged.best_changed,
                     stuck_all_strands=int(result.metrics["stuck"]),
                     iteration_s=time.perf_counter() - t_start)
            state.metrics = stats = m
            self.metrics.append(m)
            self.strand_metrics.append([result.metrics])
            self._notify("on_iteration_end", state)
            del result, obs, jobs

            # iterations 1..: K slots, merged and refilled in order
            rate = self.schedule.rate(1, stats) if cfg.iterations > 1 else 0.0
            for j in range(k if cfg.iterations > 1 else 0):
                slots[j] = self._submit(ledger, scenario, travel_time, transit_weights, runner,
                                        1, j, rate)
            for it in range(1, cfg.iterations):
                t_start = time.perf_counter()
                state = IterationState(it, it == cfg.iterations - 1,
                                       scenario.with_population(ledger.population))
                self._notify("on_iteration_start", state)
                last = it == cfg.iterations - 1
                next_rate = 0.0 if last else self.schedule.rate(it + 1, stats)
                results, round_obs, wait_s, merge_s, build_s = [], [], 0.0, 0.0, 0.0
                new_plans = pruned = innovations = dropped = 0
                snapshot_s = getattr(self.loom, "snapshot_s", 0.0)
                best_changed = []
                for j in range(k):
                    t0 = time.perf_counter()
                    slot = slots[j]
                    result = slot.future.result()
                    t1 = time.perf_counter()
                    obs = self._remapped(ledger, result, slot)
                    dropped += len(result.observations) - len(obs)
                    merged = merge(ledger, obs, self.reducer, max_plans, removal,
                                   weave_rng(cfg.seed, it, "prune", j), selected_strand=j)
                    ledger.in_flight[slot.cohort] = False
                    innovations += len(slot.cohort)
                    slots[j] = None
                    if merged.live is not None:  # compacted: renumber the strands in flight
                        for other in slots:
                            if other is not None:
                                other.rows = _renumber(other.rows, merged.live)
                    travel_time = estimator.observe(pool.add(result.aggregate))
                    t2 = time.perf_counter()
                    if not last:
                        scenario = scenario.with_population(ledger.population)
                        slots[j] = self._submit(ledger, scenario, travel_time, transit_weights,
                                                runner, it + 1, j, next_rate)
                    t3 = time.perf_counter()
                    wait_s, merge_s, build_s = wait_s + t1 - t0, merge_s + t2 - t1, build_s + t3 - t2
                    new_plans, pruned = new_plans + merged.new_plans, pruned + merged.pruned
                    best_changed.append(merged.best_changed)
                    round_obs.append(obs.score)
                    result.aggregate = None  # held in the pool
                    results.append(result)
                scenario = state.scenario = scenario.with_population(ledger.population)
                state.strands = results
                state.travel_time = travel_time
                self._notify("on_scoring", state)

                scores = np.concatenate(round_obs) if round_obs else np.zeros(0)
                m = score_metrics(it, ledger.population)
                m.update(_mean_strand_metrics(results))
                m.update(score_performed=float(scores.mean()) if len(scores) else float("nan"),
                         strands=k, innovation_rate=rate, innovations=innovations,
                         coverage=innovations / n if n else 0.0, new_plans=new_plans,
                         pruned_plans=pruned, best_plan_changed=float(np.mean(best_changed)),
                         dropped_observations=dropped,
                         stuck_all_strands=int(sum(r.metrics["stuck"] for r in results)),
                         # main process: waiting for results, merging (with the
                         # travel times), building and submitting the next jobs
                         # (of which snapshot_s: writing their snapshots)
                         wait_s=wait_s, merge_s=merge_s, replanning_s=build_s,
                         snapshot_s=getattr(self.loom, "snapshot_s", 0.0) - snapshot_s,
                         strand_replanning_s=float(max(r.metrics.get("strand_replanning_s", 0.0)
                                                       for r in results)),
                         strand_s=float(max(r.seconds for r in results)),
                         iteration_s=time.perf_counter() - t_start)
                state.metrics = stats = m
                rate = next_rate
                self.metrics.append(m)
                self.strand_metrics.append([r.metrics for r in results])
                self._notify("on_iteration_end", state)
                del results, round_obs
        finally:
            for slot in slots:  # an error: let the strands in flight finish
                if slot is not None:
                    slot.future.exception()
            self.loom.close()
            self.runner.close()
            self._notify("on_shutdown", self)
        return WeaveResult(scenario, travel_time, self.metrics, self.strand_metrics)

    def _submit(self, ledger: Ledger, scenario, travel_time, transit_weights,
                runner: VectorRunner, it: int, j: int, rate: float) -> _InFlight:
        """Builds slot ``j``'s job for iteration ``it`` from the ledger as it
        stands, and submits it."""
        seed = self.params.seed
        rctx = ReplanningContext.from_scenario(scenario, travel_time, it, self.router_factory,
                                               self.transit_router_factory, transit_weights)
        # a share ``rate`` of everyone, drawn from those not in flight elsewhere
        eligible = int(np.count_nonzero(~ledger.in_flight))
        share = min(1.0, rate * len(ledger) / eligible) if eligible else 0.0
        cohort = (self.scheduler.allocate(ledger, 1, share, it, weave_rng(seed, it, "allocate", j))[0]
                  if share > 0 else np.zeros(0, dtype=np.int64))
        job = self._strand_job(ledger, rctx, runner, it, j, cohort)
        ledger.last_innovated[cohort] = it
        ledger.in_flight[cohort] = True
        rows = np.asarray(job.rows, dtype=np.int64)
        if getattr(self.loom, "workers", 0):  # workers load only what they use
            job = slim_jobs([job])[0]
        return _InFlight(self.loom.submit(job), cohort, rows)

    def _remapped(self, ledger: Ledger, result: StrandResult, slot: _InFlight) -> Observations:
        """``result``'s observations with its new plans added to the ledger
        store and its background plans as current ledger rows; observations
        of background plans the ledger no longer holds are dropped."""
        obs = self._add_new_plans(ledger, [result])
        cand = obs.role == CANDIDATE
        plan = obs.plan.copy()
        bg = ~cand
        plan[bg] = slot.rows[obs.person[bg]]
        keep = cand.copy()
        live = bg & (plan >= 0)
        keep[live] = ledger.slots_of(obs.person[live], plan[live]) >= 0
        return Observations(obs.person[keep], plan[keep], obs.score[keep], obs.role[keep],
                            obs.strand[keep], obs.basis[keep])


def _sparse(aggregate):
    return aggregate.to_sparse() if isinstance(aggregate, TraversalAggregate) else aggregate


class _Pool:
    """The sum of the last ``k`` strands' traversal aggregates, kept up to
    date in place: each :meth:`add` adds the newest and subtracts the one it
    evicts (only the cells they use), and every ``k`` adds the sum is rebuilt
    from the window, so rounding cannot accumulate."""

    def __init__(self, k: int):
        self.window: deque = deque(maxlen=k)
        self.total: Optional[TraversalAggregate] = None
        self.adds = 0

    def add(self, aggregate) -> TraversalAggregate:
        new = _sparse(aggregate)
        self.adds += 1
        if self.total is None or self.adds % self.window.maxlen == 0:
            self.window.append(new)
            self.total = sum_aggregates(self.window)
            return self.total
        if len(self.window) == self.window.maxlen:
            old = self.window[0]
            d = self.total
            d.total.reshape(-1)[old.cells] -= old.total
            d.count.reshape(-1)[old.cells] -= old.count
            d.sum_sq.reshape(-1)[old.cells] -= old.sum_sq
        self.window.append(new)
        new.add_into(self.total)
        return self.total


def _renumber(rows: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Store rows after compacting to the sorted rows ``live`` (``-1``: gone)."""
    if not len(live):
        return np.full_like(rows, -1)
    at = np.minimum(np.searchsorted(live, rows), len(live) - 1)
    return np.where((rows >= 0) & (live[at] == rows), at, -1)


def _mean_strand_metrics(results: list[StrandResult]) -> dict:
    """The round's strand metrics (leg statistics), averaged over strands."""
    keys = set.intersection(*(set(r.metrics) for r in results)) if results else set()
    return {key: float(np.mean([r.metrics[key] for r in results])) for key in sorted(keys)
            if key not in _STRAND_ONLY and not key.endswith(("_s", "_mb"))
            and not key.startswith("score_")
            and all(isinstance(r.metrics[key], (int, float)) for r in results)}
