"""Plan selectors: choose which of a person's plans to execute (or remove).

All selectors return a plan *index*. They pick an unscored plan first if one
exists (a new plan must be tried before it can be judged), mirroring
MATSim's behaviour.

Two forms of the same rule:

* ``select(person, rng)``: one :class:`Person`, a ``random.Random``
  (the per-person path: :class:`~mobslim.replanning.runners.SerialRunner`).
* ``select_many(scores, n_plans, selected, rng)``: many persons at once
  (:class:`~mobslim.replanning.vectorised.VectorRunner`). ``scores`` is
  their ``(m, n_slots)`` score table (``NaN``: unscored or empty slot),
  ``n_plans`` and ``selected`` their plan counts and selected slots, ``rng``
  a ``numpy.random.Generator``. Returns one slot per person. A selector
  without it is run per person by the vectorised runner.
"""

import math
from random import Random
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from mobslim.population import Person


@runtime_checkable
class PlanSelector(Protocol):
    def select(self, person: Person, rng: Random) -> int: ...


def _valid(scores: np.ndarray, n_plans: np.ndarray) -> np.ndarray:
    return np.arange(scores.shape[1])[None, :] < n_plans[:, None]


def _first_unscored_many(scores: np.ndarray, n_plans: np.ndarray) -> np.ndarray:
    """Per row, the first used slot without a score, or ``-1``."""
    unscored = _valid(scores, n_plans) & np.isnan(scores)
    return np.where(unscored.any(axis=1), unscored.argmax(axis=1), -1)


def _uniform_slot(u: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.minimum((u * n).astype(np.int64), n - 1)


def _first_unscored(person: Person) -> Optional[int]:
    for i, score in enumerate(person.scores):
        if score is None:
            return i
    return None


class KeepSelectedSelector:
    """Keep the currently selected plan."""

    def select(self, person: Person, rng: Random) -> int:
        return person.selected

    def select_many(self, scores, n_plans, selected, rng) -> np.ndarray:
        return np.asarray(selected, dtype=np.int64).copy()


class BestScoreSelector:
    def select(self, person: Person, rng: Random) -> int:
        unscored = _first_unscored(person)
        if unscored is not None:
            return unscored
        return max(range(len(person.scores)), key=person.scores.__getitem__)

    def select_many(self, scores, n_plans, selected, rng) -> np.ndarray:
        first = _first_unscored_many(scores, n_plans)
        best = np.where(_valid(scores, n_plans), scores, -np.inf).argmax(axis=1)
        return np.where(first >= 0, first, best)


class RandomSelector:
    def select(self, person: Person, rng: Random) -> int:
        unscored = _first_unscored(person)
        if unscored is not None:
            return unscored
        return rng.randrange(len(person.plans))

    def select_many(self, scores, n_plans, selected, rng) -> np.ndarray:
        first = _first_unscored_many(scores, n_plans)
        return np.where(first >= 0, first, _uniform_slot(rng.random(len(n_plans)), n_plans))


class ExpBetaSelector:
    """Multinomial logit over plan scores: ``P(i) ∝ exp(beta * score_i)``."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def select(self, person: Person, rng: Random) -> int:
        unscored = _first_unscored(person)
        if unscored is not None:
            return unscored
        scores = person.scores
        top = max(scores)
        weights = [math.exp(self.beta * (s - top)) for s in scores]
        return rng.choices(range(len(scores)), weights=weights)[0]

    def select_many(self, scores, n_plans, selected, rng) -> np.ndarray:
        first = _first_unscored_many(scores, n_plans)
        valid = _valid(scores, n_plans)
        filled = np.where(valid, scores, -np.inf)
        top = filled.max(axis=1, keepdims=True)
        weights = np.where(valid, np.exp(self.beta * (np.where(valid, scores, 0.0) - top)), 0.0)
        cum = np.cumsum(weights, axis=1)
        x = rng.random(len(n_plans)) * cum[:, -1]
        pick = np.minimum((cum <= x[:, None]).sum(axis=1), n_plans - 1)
        return np.where(first >= 0, first, pick)


class ChangeExpBetaSelector:
    """MATSim's ``ChangeExpBeta``: compare the current plan with a random other.

    Switches with probability ``0.01 * exp(0.5 * beta * (s_other - s_current))``.
    In equilibrium this yields logit choice probabilities, while moving
    agents gradually.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def select(self, person: Person, rng: Random) -> int:
        unscored = _first_unscored(person)
        if unscored is not None:
            return unscored
        n = len(person.plans)
        current = person.selected
        if n == 1:
            return current
        other = rng.randrange(n - 1)
        if other >= current:
            other += 1
        delta = person.scores[other] - person.scores[current]
        weight = math.exp(min(0.5 * self.beta * delta, 700.0))
        return other if rng.random() < 0.01 * weight else current

    def select_many(self, scores, n_plans, selected, rng) -> np.ndarray:
        first = _first_unscored_many(scores, n_plans)
        m = len(n_plans)
        current = np.asarray(selected, dtype=np.int64)
        u_other, u_switch = rng.random(m), rng.random(m)
        many = n_plans > 1
        other = _uniform_slot(u_other, np.maximum(n_plans - 1, 1))
        other += other >= current
        other = np.where(many, other, current)
        rows = np.arange(m)
        delta = scores[rows, other] - scores[rows, current]
        with np.errstate(invalid="ignore"):
            weight = np.exp(np.minimum(0.5 * self.beta * np.where(many, delta, 0.0), 700.0))
        switch = many & (u_switch < 0.01 * weight)
        return np.where(first >= 0, first, np.where(switch, other, current))


class WorstScoreSelector:
    """Removal selector: the worst-scored plan (unscored plans are kept)."""

    def select(self, person: Person, rng: Random) -> int:
        scored = [i for i, s in enumerate(person.scores) if s is not None]
        if not scored:
            return rng.randrange(len(person.plans))
        return min(scored, key=person.scores.__getitem__)

    def select_many(self, scores, n_plans, selected, rng) -> np.ndarray:
        scored = _valid(scores, n_plans) & ~np.isnan(scores)
        worst = np.where(scored, scores, np.inf).argmin(axis=1)
        random = _uniform_slot(rng.random(len(n_plans)), n_plans)
        return np.where(scored.any(axis=1), worst, random)
