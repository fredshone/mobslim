"""Who innovates, and in which strand.

**Coverage.** With ``K`` strands and innovation rate ``r`` (the share of
persons innovating in each strand), coverage is ``c = r * K``, which may
exceed 1. Every person innovates ``floor(c)`` times per iteration; a share
``frac(c)`` innovate once more, chosen by priority. A person innovates at
most once per strand (``r <= 1``, so nobody needs more than ``K`` strands).

**Priority** (:class:`WeightedStalenessScheduler`): weighted random sampling
without replacement (Efraimidis-Spirakis keys ``u ** (1 / w)``) with weight
``w = (1 + staleness) ** alpha``, where staleness is the number of iterations
since the person last innovated. ``alpha = 0`` is uniform random;
``alpha -> inf`` strict oldest-first; ``alpha`` around 1-2 is random across
iterations but leaves nobody out for long.

**Spreading.** The chosen persons are shuffled and expanded into their
copies (copies adjacent), and slot ``i`` is dealt to strand ``i mod K``:
strand sizes differ by at most one and a person's copies land in different
strands.

**Rate.** An :class:`InnovationSchedule` gives the rate per iteration
(:class:`ConstantRate`, :class:`StepRate`, :class:`LinearRate`); the number
of strands stays fixed while the cohorts shrink.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from mobslim.weave.ledger import Ledger


@runtime_checkable
class InnovationScheduler(Protocol):
    def allocate(self, ledger: Ledger, n_strands: int, rate: float, iteration: int,
                 rng: np.random.Generator) -> list[np.ndarray]:
        """One array of person indices per strand: who innovates there."""
        ...


def staleness(ledger: Ledger, iteration: int) -> np.ndarray:
    """Iterations since each person last innovated (never: ``iteration + 1``)."""
    return ledger.staleness(iteration).astype(float)


class WeightedStalenessScheduler:
    """Weighted sampling by ``(1 + priority) ** alpha`` (see the module docs).

    ``priority(ledger, iteration) -> float[n_persons]`` defaults to
    :func:`staleness`; another priority (score variance, recent gain, ...)
    replaces only that function. Persons ``in_flight`` are skipped.
    """

    def __init__(self, alpha: float = 1.0,
                 priority: Callable[[Ledger, int], np.ndarray] = staleness):
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = alpha
        self.priority = priority

    def weights(self, ledger: Ledger, iteration: int) -> np.ndarray:
        p = np.asarray(self.priority(ledger, iteration), dtype=float)
        with np.errstate(over="ignore"):
            return (1.0 + np.maximum(p, 0.0)) ** self.alpha

    def allocate(self, ledger, n_strands, rate, iteration, rng):
        if n_strands < 1:
            raise ValueError("n_strands must be >= 1.")
        if not 0 <= rate <= 1:
            raise ValueError("rate must be in [0, 1].")
        eligible = np.flatnonzero(~ledger.in_flight)
        n = len(eligible)
        coverage = rate * n_strands
        whole = math.floor(coverage + 1e-12)
        extra = int(round((coverage - whole) * n))
        copies = np.full(n, whole, dtype=np.int64)
        if extra:
            w = self.weights(ledger, iteration)[eligible]
            u = rng.random(n)
            with np.errstate(divide="ignore"):
                keys = np.log(u) / w  # log of u ** (1 / w): larger is chosen first
            chosen = np.argpartition(-keys, extra - 1)[:extra]
            copies[chosen] += 1
        return deal(eligible, copies, n_strands, rng)


def deal(persons: np.ndarray, copies: np.ndarray, n_strands: int,
         rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle ``persons``, expand each into ``copies`` adjacent copies and
    deal slot ``i`` to strand ``i mod n_strands``."""
    if (copies > n_strands).any():
        raise ValueError("A person cannot innovate more than once per strand.")
    order = rng.permutation(len(persons))
    slots = np.repeat(persons[order], copies[order])
    return [np.sort(slots[j::n_strands]) for j in range(n_strands)]


# ------------------------------------------------------------------ rates


@runtime_checkable
class InnovationSchedule(Protocol):
    def rate(self, iteration: int, stats: Optional[dict] = None) -> float:
        """Share of persons innovating per strand at ``iteration``
        (``stats``: the previous iteration's metrics, if any)."""
        ...


class ConstantRate:
    def __init__(self, rate: float):
        _check_rate(rate)
        self.value = rate

    def rate(self, iteration, stats=None) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantRate({self.value})"


class StepRate:
    """``rate`` until iteration ``until`` (exclusive), then ``after`` (MATSim's
    ``fractionOfIterationsToDisableInnovation`` with ``after = 0``)."""

    def __init__(self, rate: float, until: int, after: float = 0.0):
        _check_rate(rate)
        _check_rate(after)
        self.value, self.until, self.after = rate, until, after

    def rate(self, iteration, stats=None) -> float:
        return self.value if iteration < self.until else self.after

    def __repr__(self) -> str:
        return f"StepRate({self.value}, until={self.until}, after={self.after})"


class LinearRate:
    """Anneals linearly from ``start`` (at iteration ``begin``) to ``end``
    (at iteration ``finish``), constant outside."""

    def __init__(self, start: float, end: float, begin: int, finish: int):
        _check_rate(start)
        _check_rate(end)
        if finish <= begin:
            raise ValueError("finish must be after begin.")
        self.start, self.end, self.begin, self.finish = start, end, begin, finish

    def rate(self, iteration, stats=None) -> float:
        t = min(max((iteration - self.begin) / (self.finish - self.begin), 0.0), 1.0)
        return self.start + t * (self.end - self.start)

    def __repr__(self) -> str:
        return f"LinearRate({self.start} -> {self.end}, {self.begin}..{self.finish})"


def _check_rate(rate: float) -> None:
    if not 0 <= rate <= 1:
        raise ValueError("innovation rate must be in [0, 1].")
