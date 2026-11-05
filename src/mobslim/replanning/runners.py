"""Per-person replanning runners: ``StrategyManager.replan`` applied to
every :class:`~mobslim.population.Person`.

Because replanning is a pure function of ``(person, context, rng)`` and every
person gets its own seeded RNG (:func:`person_rng`), these runners produce
identical populations for the same seed.

The controller's default is the vectorised
:class:`~mobslim.replanning.vectorised.VectorRunner`, which applies the same
rules on the population's tables (with numpy random numbers, so its results
differ from these runners' in the draws, not in the rules). These runners
remain the reference, and the way to run custom Python strategies in
parallel (:class:`ProcessPoolRunner`).

With ``batch_routing=True`` (the default) the runners replan with
``ctx.batched()``: :class:`~mobslim.replanning.modules.ReRoute` only clears
routes, and the runner then routes every new plan of the batch at once
(:func:`~mobslim.replanning.modules.route_plans`), letting the router share
searches and run them in parallel. The routes are the same as routing during
replanning, so the result does not depend on the setting.
"""

import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Optional, Protocol, runtime_checkable

from mobslim.population import Person, Population
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.modules import (
    needs_routing,
    route_plans,
    routing_groups,
)
from mobslim.replanning.strategy import StrategyManager, person_rng
from mobslim.routing.compiled import THREADS_ENV


@runtime_checkable
class ReplanningRunner(Protocol):
    def run(self, population: Population, manager: StrategyManager,
            ctx: ReplanningContext, seed: int) -> Population: ...

    def close(self) -> None: ...


def route_persons(persons: list[Person], ctx: ReplanningContext,
                  before: Optional[list[Person]] = None, start: int = 0,
                  indices=None) -> list[Person]:
    """Route the unrouted legs of every plan of ``persons`` (person indices
    ``start, start + 1, ...``, or ``indices``) in one batch.

    With ``before`` (the same persons before replanning) only plans that are
    new since then are checked. With ``ctx.transit_weights`` each person's
    transit legs are routed with that person's weights (a plan shared by
    persons with different weights is routed once per weights).
    """
    groups = routing_groups(ctx)
    codes = None if groups is None else groups[0]

    def code(k: int) -> int:
        if codes is None:
            return -1
        return int(codes[start + k if indices is None else indices[k]])

    todo: dict[tuple, object] = {}  # (id(plan), weights code) -> plan, each once
    for k, person in enumerate(persons):
        if before is not None:
            old = before[k]
            if person is old:
                continue
            known = {id(p) for p in old.plans}
        else:
            known = ()
        c = code(k)
        for plan in person.plans:
            key = (id(plan), c)
            if key[0] not in known and key not in todo and needs_routing(plan, ctx):
                todo[key] = plan
    if not todo:
        return persons
    weights = persons_of = None
    if groups is not None:
        weights = [groups[1][c] for _, c in todo]
        persons_of = [groups[2][c] for _, c in todo]
    routed = dict(zip(todo, route_plans(list(todo.values()), ctx, weights, persons_of)))
    out = []
    for k, person in enumerate(persons):
        c = code(k)
        if any((id(plan), c) in routed for plan in person.plans):
            person = person.with_plans([routed.get((id(plan), c), plan) for plan in person.plans])
        out.append(person)
    return out


def replan_persons(persons, manager: StrategyManager, ctx: ReplanningContext, seed: int,
                   start: int = 0, batch_routing: bool = True) -> list[Person]:
    """Replan ``persons`` (person indices ``start, start + 1, ...``)."""
    persons = list(persons)
    rctx = ctx.batched() if batch_routing else ctx
    out = [
        manager.replan(person, rctx.for_person(start + k), person_rng(seed, ctx.iteration, start + k))
        for k, person in enumerate(persons)
    ]
    return route_persons(out, ctx, persons, start) if batch_routing else out


class SerialRunner:
    def __init__(self, batch_routing: bool = True):
        self.batch_routing = batch_routing

    def run(self, population, manager, ctx, seed):
        return population.replace(
            replan_persons(list(population), manager, ctx, seed, 0, self.batch_routing)
        )

    def close(self) -> None:
        pass


# Per worker process: the (manager, context) of the current replanning call.
_shared: Optional[tuple[str, StrategyManager, ReplanningContext]] = None


def _replan_chunk(args):
    """Replan one contiguous chunk of persons inside a worker process.

    The manager and context are not part of ``args``: they are pickled once
    per call into a file, and each worker loads them once (keyed by ``token``)
    and reuses them for every chunk it gets. That keeps the per-chunk payload
    to the persons only, however large the network and travel-time table are.
    """
    global _shared
    token, path, seed, start, persons, batch_routing = args
    if _shared is None or _shared[0] != token:
        with open(path, "rb") as f:
            manager, ctx = pickle.load(f)
        _shared = (token, manager, ctx)
    _, manager, ctx = _shared
    return replan_persons(persons, manager, ctx, seed, start, batch_routing)


def _init_worker():
    # the processes already run in parallel: one routing thread each, unless set
    os.environ.setdefault(THREADS_ENV, "1")


class ProcessPoolRunner:
    """Replan in parallel worker processes.

    The population is split into ``workers * chunks_per_worker`` contiguous
    chunks. The (picklable) strategy manager and context are written once per
    call to a temporary file that each worker reads once; chunks carry only
    persons. Each chunk's new plans are routed in one batch inside its worker.
    Workers route with one thread each (``$MOBSLIM_ROUTER_THREADS``, unless
    already set). The pool is created on first use and reused until :meth:`close`.
    Strategies, modules and router factories must therefore be picklable
    and importable by the workers (defined in a module, not a notebook cell).
    With the default ``spawn`` start method, scripts need an
    ``if __name__ == "__main__":`` guard; ``start_method="fork"`` (Linux)
    avoids both constraints.
    """

    def __init__(self, workers: Optional[int] = None, chunks_per_worker: int = 4,
                 start_method: str = "spawn", batch_routing: bool = True):
        self.batch_routing = batch_routing
        self.workers = workers or os.cpu_count() or 1
        self.chunks_per_worker = chunks_per_worker
        self.start_method = start_method
        self._pool: Optional[ProcessPoolExecutor] = None
        self._calls = 0

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=get_context(self.start_method),
                initializer=_init_worker,
            )
        return self._pool

    def run(self, population, manager, ctx, seed):
        persons = list(population)
        if not persons:
            return population
        pool = self._executor()
        self._calls += 1
        token = f"{os.getpid()}:{id(self)}:{self._calls}"
        n_chunks = max(1, min(len(persons), self.workers * self.chunks_per_worker))
        size = -(-len(persons) // n_chunks)
        fd, path = tempfile.mkstemp(prefix="mobslim-replan-", suffix=".pkl")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump((manager, ctx), f, protocol=pickle.HIGHEST_PROTOCOL)
            jobs = [
                (token, path, seed, start, persons[start:start + size], self.batch_routing)
                for start in range(0, len(persons), size)
            ]
            results = []
            for chunk in pool.map(_replan_chunk, jobs):
                results.extend(chunk)
        finally:
            os.unlink(path)
        return population.replace(results)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
