"""The ledger: the global record of every person's plans across strands.

The ledger is a :class:`~mobslim.population.Population` (plans, scores,
selected plan) plus what weaving needs on top:

==================  ===========================  ==============================================
``score_n``         ``int32[n_persons, slots]``  samples behind each stored score
``last_innovated``  ``int64[n_persons]``         iteration of the last innovation (-1: never)
``in_flight``       ``bool[n_persons]``          innovating in a launched, unmerged strand
``version``         ``int``                      merges so far (an observation's ``basis``)
==================  ===========================  ==============================================

``score_n`` is aligned with ``population.scores`` slot by slot. Strands hold
store row numbers, so the store is only compacted at a merge (by
:meth:`~mobslim.population.Population.with_tables`), never while strands
are in flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mobslim.population import Population


@dataclass
class Ledger:
    population: Population
    score_n: np.ndarray
    last_innovated: np.ndarray
    in_flight: np.ndarray
    version: int = 0
    # each person's best-scored plan row at the last merge (for metrics)
    best_rows: np.ndarray = field(default=None, repr=False)

    @classmethod
    def start(cls, population: Population) -> "Ledger":
        n = len(population)
        score_n = np.where(np.isnan(population.scores), 0, 1).astype(np.int32)
        return cls(population, score_n, np.full(n, -1, dtype=np.int64),
                   np.zeros(n, dtype=bool), 0, best_plan_rows(population))

    def __len__(self) -> int:
        return len(self.population)

    def staleness(self, iteration: int) -> np.ndarray:
        """Iterations since each person last innovated (never: ``iteration + 1``)."""
        return iteration - self.last_innovated

    def slots_of(self, persons, plan_rows) -> np.ndarray:
        """The slot holding store row ``plan_rows[i]`` for person
        ``persons[i]`` (``-1``: the person does not hold it)."""
        persons = np.asarray(persons, dtype=np.int64)
        plan_rows = np.asarray(plan_rows, dtype=np.int64)
        held = self.population.plan_ids[persons] == plan_rows[:, None]
        return np.where(held.any(axis=1), held.argmax(axis=1), -1)


def best_plan_rows(population: Population) -> np.ndarray:
    """Each person's best-scored plan's store row (unscored: the selected one)."""
    scores = population.scores
    valid = np.arange(scores.shape[1])[None, :] < population.n_plans[:, None]
    filled = np.where(valid & ~np.isnan(scores), scores, -np.inf)
    slot = np.where(np.isfinite(filled).any(axis=1), filled.argmax(axis=1), population.selected)
    return population.plan_ids[np.arange(len(population)), slot].astype(np.int64)
