"""Vectorised replanning: the whole population in array operations.

:class:`VectorRunner` applies a :class:`~mobslim.replanning.StrategyManager`
(or a :class:`~mobslim.replanning.PersonStrategyManager`) to every person at
once, on the population's tables instead of :class:`~mobslim.population.Person`
values:

1. **Strategy choice**: one uniform draw per person against each person's
   manager's cumulative weights.
2. **Plan memory**: before an innovative strategy, plans are removed while a
   person holds ``max_plans`` (removal selector, vectorised); after the
   strategy, while they hold more.
3. **Selection**: each strategy's selector picks a slot for its persons
   (``select_many``).
4. **Innovation**: the selected plans of an innovative strategy's persons are
   gathered from the store as one :class:`~mobslim.planbatch.PlanBatch`, sent
   through its modules (``mutate_batch``), routed together
   (:func:`~mobslim.replanning.modules.route_batch`) and appended to the store
   as new, unscored, selected plans.

The rules are those of the per-person path (:meth:`StrategyManager.replan`
and :class:`~mobslim.replanning.SerialRunner`); only the random numbers
differ: one ``numpy`` generator keyed by ``(seed, iteration)`` replaces a
``random.Random`` per person. Results are deterministic and do not depend on
how the work is split.

Anything without a vectorised form (a custom :class:`Strategy`, or a
selector without ``select_many``, or a module without ``mutate_batch``) is
applied per person, exactly as :class:`SerialRunner` does, for the persons
that drew it; everyone else stays vectorised.
"""

from __future__ import annotations

import numpy as np

from mobslim.planbatch import PlanBatch
from mobslim.planstore import PlanStore
from mobslim.population import Population
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.modules import route_batch
from mobslim.replanning.strategy import (
    PersonStrategyManager,
    PlanStrategy,
    StrategyManager,
    person_rng,
)


def replanning_rng(seed: int, iteration: int) -> np.random.Generator:
    """The generator for one replanning step: counter-based (Philox), keyed by
    ``(seed, iteration)``."""
    key = np.array([seed % 2**64, iteration % 2**64], dtype=np.uint64)
    return np.random.Generator(np.random.Philox(key=key))


def _vectorisable(manager: StrategyManager, strategy) -> bool:
    return (isinstance(strategy, PlanStrategy)
            and hasattr(strategy.selector, "select_many")
            and all(hasattr(m, "mutate_batch") for m in strategy.modules)
            and hasattr(manager.removal_selector, "select_many"))


class _Tables:
    """Writable copies of a population's plan tables, one spare slot wide."""

    def __init__(self, population: Population, spare: int):
        k = population.plan_ids.shape[1] + spare
        n = len(population)
        self.plan_ids = np.full((n, k), -1, dtype=np.int32)
        self.plan_ids[:, :population.plan_ids.shape[1]] = population.plan_ids
        self.scores = np.full((n, k), np.nan)
        self.scores[:, :population.scores.shape[1]] = population.scores
        self.selected = population.selected.astype(np.int64)
        self.n_plans = population.n_plans.astype(np.int64)

    def remove(self, rows: np.ndarray, slots: np.ndarray) -> None:
        """Remove slot ``slots[i]`` of person ``rows[i]`` (as ``Person.remove_plan``)."""
        k = self.plan_ids.shape[1]
        cols = np.arange(k - 1)[None, :]
        src = cols + (cols >= slots[:, None])
        for table, empty in ((self.plan_ids, -1), (self.scores, np.nan)):
            block = table[rows]
            table[rows, :-1] = np.take_along_axis(block, src, axis=1)
            table[rows, -1] = empty
        sel = self.selected[rows]
        self.selected[rows] = np.where(sel == slots, 0, sel - (slots < sel))
        self.n_plans[rows] -= 1

    def remove_while(self, rows: np.ndarray, condition, selector, rng) -> None:
        """Remove plans of ``rows`` chosen by ``selector`` while ``condition(n_plans)``."""
        while len(rows):
            rows = rows[condition(self.n_plans[rows])]
            if not len(rows):
                return
            slots = selector.select_many(self.scores[rows], self.n_plans[rows],
                                         self.selected[rows], rng)
            self.remove(rows, np.asarray(slots, dtype=np.int64))


class VectorRunner:
    """Replan everyone at once with array operations (see the module docs).

    ``batch_routing`` is accepted for symmetry with the other runners; new
    plans are always routed together after the modules ran (the routes do
    not depend on it).
    """

    def __init__(self, batch_routing: bool = True):
        self.batch_routing = batch_routing

    def run(self, population: Population, manager, ctx: ReplanningContext, seed: int) -> Population:
        n = len(population)
        if n == 0:
            return population
        split = manager_split(manager, n)
        if split is None:  # some other manager: the per-person path
            from mobslim.replanning.runners import SerialRunner

            return SerialRunner(self.batch_routing).run(population, manager, ctx, seed)
        managers, codes = split
        it = ctx.iteration
        rng = replanning_rng(seed, it)
        u = rng.random(n)  # strategy choice
        groups = _choose(managers, codes, np.arange(n), u, lambda m: m._table(it)[0])

        vector = [g for g in groups if _vectorisable(g[0], g[1])]
        fallback = [g for g in groups if not _vectorisable(g[0], g[1])]
        tables = _Tables(population, spare=1 if any(g[1].is_innovative for g in vector) else 0)
        store = population.plans

        # selection, plan memory, and the new plans' sources
        innovations: list[tuple[PlanStrategy, np.ndarray, np.ndarray]] = []
        next_row = len(store)
        for m, strategy, rows in vector:
            removal = m.removal_selector
            if strategy.is_innovative:
                tables.remove_while(rows, lambda k, mp=m.max_plans: (k >= mp) & (k > 1), removal, rng)
            slots = strategy.selector.select_many(tables.scores[rows], tables.n_plans[rows],
                                                  tables.selected[rows], rng)
            tables.selected[rows] = slots
            if strategy.is_innovative:
                sources = tables.plan_ids[rows, tables.selected[rows]].astype(np.int64)
                innovations.append((strategy, rows, sources))
                new_slot = tables.n_plans[rows]
                tables.plan_ids[rows, new_slot] = np.arange(next_row, next_row + len(rows))
                tables.scores[rows, new_slot] = np.nan
                tables.selected[rows] = new_slot
                tables.n_plans[rows] += 1
                next_row += len(rows)
            tables.remove_while(rows, lambda k, mp=m.max_plans: k > mp, removal, rng)

        # innovation: modules per strategy, then all new plans routed at once
        if innovations:
            store = store.append(innovate_batch(store, innovations, ctx, rng))
        result = population.with_tables(store, tables.plan_ids, tables.scores, tables.selected,
                                        tables.n_plans)
        if fallback:
            result = self._per_person(result, fallback, ctx, seed)
        return result

    # -- the two halves of replanning, for callers that schedule them (Weave) --
    def select(self, population: Population, manager, ctx: ReplanningContext,
               rng: np.random.Generator, seed: int = 0) -> np.ndarray:
        """The slot each person would execute under the *selection-only*
        strategies (chosen by their relative weights), without changing the
        population. One uniform draw per person picks the strategy, then each
        strategy's selector draws. ``seed`` keys the per-person fallback for
        selectors without ``select_many``."""
        n = len(population)
        split = manager_split(manager, n)
        if split is None:
            raise TypeError("select needs a StrategyManager or PersonStrategyManager.")
        managers, codes = split
        u = rng.random(n)
        groups = _choose(managers, codes, np.arange(n), u, _selection_options)
        slots = population.selected.astype(np.int64).copy()
        for m, strategy, rows in groups:
            if isinstance(strategy, PlanStrategy) and hasattr(strategy.selector, "select_many"):
                slots[rows] = strategy.selector.select_many(
                    population.scores[rows], population.n_plans[rows],
                    population.selected[rows], rng)
            else:
                for i in rows.tolist():
                    person = strategy.apply(population[i], ctx.for_person(i),
                                            person_rng(seed, ctx.iteration, i))
                    slots[i] = person.selected
        return slots

    def innovate(self, population: Population, manager, ctx: ReplanningContext, persons,
                 rng: np.random.Generator):
        """New plans for ``persons`` (person indices; a person may repeat, and
        gets one independent new plan per copy), without attaching them to
        anyone. Each copy draws an innovative strategy by the relative
        weights of its manager's innovative strategies; the strategy's
        selector picks the source plan and its modules mutate a copy. All new
        plans are routed together and appended to the population's store.

        Returns ``(store, rows)``: the extended store and each copy's new row.
        """
        persons = np.asarray(persons, dtype=np.int64)
        store = population.plans
        if not len(persons):
            return store, np.zeros(0, dtype=np.int64)
        innovations, order = self.plan_innovations(population, manager, ctx, persons, rng)
        new = innovate_batch(store, innovations, ctx, rng)
        first = len(store)
        store = store.append(new)
        rows = np.empty(len(persons), dtype=np.int64)
        rows[order] = np.arange(first, first + len(persons))
        return store, rows

    def plan_innovations(self, population: Population, manager, ctx: ReplanningContext,
                         persons, rng: np.random.Generator):
        """The cheap half of :meth:`innovate`: each copy's innovative
        strategy and source plan. Non-vectorisable strategies are applied
        here, person by person.

        Returns ``(innovations, order)``. ``innovations`` goes to
        :func:`innovate_batch` (with the population's store), whose new plan
        ``i`` belongs to copy ``order[i]`` of ``persons``.
        """
        persons = np.asarray(persons, dtype=np.int64)
        if not len(persons):
            return [], np.zeros(0, dtype=np.int64)
        split = manager_split(manager, len(population))
        if split is None:
            raise TypeError("innovate needs a StrategyManager or PersonStrategyManager.")
        managers, codes = split
        u = rng.random(len(persons))
        picks = _choose(managers, codes[persons], np.arange(len(persons)), u,
                        _innovation_options, missing="innovative")
        innovations, order, fallback = [], [], []
        for m, strategy, copies in picks:
            rows = persons[copies]
            if _vectorisable(m, strategy):
                slots = strategy.selector.select_many(
                    population.scores[rows], population.n_plans[rows],
                    population.selected[rows], rng)
                sources = population.plan_ids[rows, slots].astype(np.int64)
                innovations.append((strategy, rows, sources))
                order.append(copies)
            else:
                fallback.append((strategy, copies))
        if fallback:  # per copy: the strategy on the person, keep its new plan
            rctx = ctx.batched()
            plans, copies_fb = [], []
            for strategy, copies in fallback:
                for k in copies.tolist():
                    i = int(persons[k])
                    person = strategy.apply(population[i], rctx.for_person(i),
                                            person_rng(int(rng.integers(2**63)), ctx.iteration, i))
                    plans.append(person.selected_plan)
                    copies_fb.append(k)
            batch = PlanBatch.from_plans(plans)
            innovations.append((None, persons[copies_fb], batch))
            order.append(np.asarray(copies_fb, dtype=np.int64))
        return innovations, np.concatenate(order)

    def _per_person(self, population: Population, groups, ctx: ReplanningContext,
                    seed: int) -> Population:
        """Apply non-vectorisable strategies person by person, as
        :class:`SerialRunner` does (with each person's ``person_rng``)."""
        from mobslim.replanning.runners import route_persons

        rctx = ctx.batched() if self.batch_routing else ctx
        persons = list(population)
        index = np.concatenate([rows for _, _, rows in groups])
        before = [persons[i] for i in index.tolist()]
        after = []
        for (m, strategy, rows) in groups:
            for i in rows.tolist():
                rng = person_rng(seed, ctx.iteration, i)
                rng.random()  # the draw the per-person path spends choosing the strategy
                after.append(_apply(m, strategy, persons[i], rctx.for_person(i), rng))
        if self.batch_routing:
            after = route_persons(after, ctx, before, indices=index.tolist())
        for i, person in zip(index.tolist(), after):
            persons[i] = person
        return population.replace(persons)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def manager_split(manager, n: int):
    """``(managers, codes)``: the strategy managers and each person's code,
    or ``None`` for a manager the vectorised path does not know."""
    if isinstance(manager, PersonStrategyManager):
        return manager.managers, np.asarray(manager.codes, dtype=np.int64)
    if isinstance(manager, StrategyManager):
        return (manager,), np.zeros(n, dtype=np.int64)
    return None


def _selection_options(m: StrategyManager):
    return [(w, s) for w, s in m.strategies if not s.is_innovative]


def _innovation_options(m: StrategyManager):
    return [(w, s) for w, s in m.strategies if s.is_innovative]


def _choose(managers, codes, items, u, options, missing: str = "selection-only"):
    """``[(manager, strategy, items)]``: item ``k`` (manager ``codes[k]``)
    takes the strategy its draw ``u[k]`` falls on, by ``options(manager)``'s
    cumulative weights. Groups come in manager, then strategy order."""
    groups = []
    for c, m in enumerate(managers):
        mine = items[codes == c]
        if not len(mine):
            continue
        opts = options(m)
        strategies = [s for _, s in opts]
        cum = np.cumsum([w for w, _ in opts], dtype=float)
        if not len(cum) or cum[-1] <= 0:
            raise ValueError(f"No {missing} strategy with positive weight.")
        pick = np.searchsorted(cum, u[mine] * cum[-1], side="right")
        pick = np.minimum(pick, len(strategies) - 1)
        for k, strategy in enumerate(strategies):
            rows = mine[pick == k]
            if len(rows):
                groups.append((m, strategy, rows))
    return groups


def innovate_batch(store: PlanStore, innovations, ctx: ReplanningContext, rng) -> PlanBatch:
    """The new plans of ``innovations`` (``(strategy, persons, sources)``,
    with ``sources`` store rows to send through the strategy's modules, or
    ``(None, persons, batch)`` for plans already made), routed in one batch."""
    rctx = ctx.batched()
    chunk = PlanStore.empty()
    persons_of_new = []
    for strategy, rows, sources in innovations:
        if strategy is None:
            batch = sources
        else:
            batch = store.take(sources)
            for module in strategy.modules:
                batch = module.mutate_batch(batch, rows, rctx, rng)
        chunk = chunk.append(batch)
        persons_of_new.append(rows)
    persons_of_new = np.concatenate(persons_of_new)
    return route_batch(chunk.take(np.arange(len(chunk))), ctx, persons_of_new)


def _apply(manager: StrategyManager, strategy, person, ctx, rng):
    """``manager.replan`` with the strategy already chosen."""
    if strategy.is_innovative:
        while len(person.plans) >= manager.max_plans and len(person.plans) > 1:
            person = person.remove_plan(manager.removal_selector.select(person, rng))
    person = strategy.apply(person, ctx, rng)
    while len(person.plans) > manager.max_plans:
        person = person.remove_plan(manager.removal_selector.select(person, rng))
    return person
