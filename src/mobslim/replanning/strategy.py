"""Strategies and the strategy manager.

A :class:`PlanStrategy` is a selector plus an optional chain of innovation
modules (MATSim's ``PlanStrategyImpl``):

* no modules: *selection* only. Pick one of the existing plans.
* with modules: *innovation*. Select a plan, copy it through the modules,
  and add the result as a new, unscored, selected plan.

The :class:`StrategyManager` picks one strategy per person per iteration by
weight, enforces the plan memory limit, and switches innovation off after a
given iteration.

Everything here is a pure function of ``(person, context, rng)``. That is what
lets :mod:`mobslim.replanning.runners` run it serially or across processes
with identical results. A learned policy plugs in as just another strategy.
"""

from dataclasses import dataclass, field
from itertools import accumulate
from random import Random
from typing import Optional, Protocol, Sequence, runtime_checkable

from mobslim.population import Person
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.modules import (
    PlanInnovationModule,
    Reroute,
    TimeAllocationMutator,
)
from mobslim.replanning.selectors import (
    ChangeExpBetaSelector,
    PlanSelector,
    RandomSelector,
    WorstScoreSelector,
)


@runtime_checkable
class Strategy(Protocol):
    """Anything that maps a person to an updated person."""

    is_innovative: bool

    def apply(self, person: Person, ctx: ReplanningContext, rng: Random) -> Person: ...


class PlanStrategy:
    def __init__(
        self,
        selector: PlanSelector,
        modules: Sequence[PlanInnovationModule] = (),
        name: Optional[str] = None,
    ):
        self.selector = selector
        self.modules = tuple(modules)
        self.name = name or (
            "+".join(type(m).__name__ for m in self.modules) or type(selector).__name__
        )

    @property
    def is_innovative(self) -> bool:
        return bool(self.modules)

    def apply(self, person: Person, ctx: ReplanningContext, rng: Random) -> Person:
        person = person.select(self.selector.select(person, rng))
        if not self.modules:
            return person
        plan = person.selected_plan
        for module in self.modules:
            plan = module.mutate(plan, ctx, rng)
        return person.add_plan(plan, select=True)  # new plans start unscored

    def __repr__(self) -> str:
        return f"PlanStrategy({self.name})"


@dataclass(frozen=True)
class StrategyManager:
    """Weighted choice of strategies plus plan-memory management.

    ``disable_innovation_at``: from this iteration on, innovative strategies
    are no longer chosen (MATSim's ``fractionOfIterationsToDisableInnovation``).
    Immutable: the choice tables are built once, not per person.
    """

    strategies: Sequence[tuple[float, Strategy]]
    max_plans: int = 5
    removal_selector: PlanSelector = field(default_factory=WorstScoreSelector)
    disable_innovation_at: Optional[int] = None
    # (strategies, cumulative weights) with and without innovative strategies
    _tables: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        strategies = tuple((float(w), s) for w, s in self.strategies)
        object.__setattr__(self, "strategies", strategies)
        if not strategies:
            raise ValueError("StrategyManager needs at least one strategy.")
        if any(w < 0 for w, _ in strategies):
            raise ValueError("Strategy weights must be non-negative.")
        if self.max_plans < 1:
            raise ValueError("max_plans must be >= 1.")
        tables = {}
        for innovate in (True, False):
            options = [(w, s) for w, s in strategies if innovate or not s.is_innovative]
            cum = list(accumulate(w for w, _ in options))
            tables[innovate] = (options, [s for _, s in options], cum)
        object.__setattr__(self, "_tables", tables)

    @classmethod
    def default(cls, max_plans: int = 5, disable_innovation_at: Optional[int] = None) -> "StrategyManager":
        """MATSim-like defaults: 80% ChangeExpBeta, 10% Reroute, 10% TimeAllocationMutator."""
        return cls(
            [
                (0.8, PlanStrategy(ChangeExpBetaSelector())),
                (0.1, PlanStrategy(RandomSelector(), [Reroute()])),
                (0.1, PlanStrategy(RandomSelector(), [TimeAllocationMutator(), Reroute()])),
            ],
            max_plans=max_plans,
            disable_innovation_at=disable_innovation_at,
        )

    def _table(self, iteration: int):
        innovate = self.disable_innovation_at is None or iteration < self.disable_innovation_at
        table = self._tables[innovate]
        if not table[2] or table[2][-1] <= 0:
            raise ValueError(f"No strategy with positive weight available at iteration {iteration}.")
        return table

    def available(self, iteration: int) -> list[tuple[float, Strategy]]:
        """``(weight, strategy)`` pairs that may be chosen at ``iteration``."""
        return list(self._table(iteration)[0])

    def choose(self, iteration: int, rng: Random) -> Strategy:
        _, strategies, cum = self._table(iteration)
        return rng.choices(strategies, cum_weights=cum)[0]

    def replan(self, person: Person, ctx: ReplanningContext, rng: Random) -> Person:
        strategy = self.choose(ctx.iteration, rng)
        if strategy.is_innovative:
            # make room for the plan the strategy will add
            while len(person.plans) >= self.max_plans and len(person.plans) > 1:
                person = person.remove_plan(self.removal_selector.select(person, rng))
        person = strategy.apply(person, ctx, rng)
        while len(person.plans) > self.max_plans:
            person = person.remove_plan(self.removal_selector.select(person, rng))
        return person


class PersonStrategyManager:
    """Per-person strategy sets: person ``i`` is replanned by
    ``managers[codes[i]]`` (each a :class:`StrategyManager` with its own
    weighted, parameterised strategies). ``names[c]`` names set ``c``.
    Same interface as :class:`StrategyManager` for the runners."""

    def __init__(self, managers: Sequence[StrategyManager], codes, names: Sequence[str] = ()):
        import numpy as np

        self.managers = tuple(managers)
        self.codes = np.asarray(codes, dtype=np.int32)
        self.codes.flags.writeable = False
        self.names = tuple(names) or tuple(f"set{i}" for i in range(len(self.managers)))
        if len(self.names) != len(self.managers):
            raise ValueError("PersonStrategyManager: one name per strategy set.")
        if len(self.codes) and (self.codes.min() < 0 or self.codes.max() >= len(self.managers)):
            raise ValueError("PersonStrategyManager: strategy set code out of range.")

    def manager_of(self, person_index: int) -> StrategyManager:
        return self.managers[int(self.codes[person_index])]

    def replan(self, person: Person, ctx: ReplanningContext, rng: Random) -> Person:
        if ctx.person_index is None:
            raise RuntimeError("PersonStrategyManager needs a context bound to a person.")
        return self.manager_of(ctx.person_index).replan(person, ctx, rng)

    def __repr__(self) -> str:
        counts = [int((self.codes == c).sum()) for c in range(len(self.managers))]
        sets = ", ".join(f"{n}: {k} persons" for n, k in zip(self.names, counts))
        return f"PersonStrategyManager({sets})"


def person_rng(seed: int, iteration: int, person_index: int) -> Random:
    """Deterministic per-person RNG, independent of execution order or process."""
    return Random(f"{seed}:{iteration}:{person_index}")
