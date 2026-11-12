"""Merging strands: observations, score reducers and the ledger update.

Every strand returns flat :class:`Observations`, one row per person it
simulated. The merge concatenates every strand's rows, groups them by
``(person, plan)`` (the plan's store row identifies it whichever strand ran
it) and lets a :class:`ScoreReducer` turn each group into a stored score.
Plans the person did not hold yet (candidates) are added; the removal
selector then prunes each person back to ``max_plans``, so a bad candidate
is dropped at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from mobslim.population import compact_plans
from mobslim.weave.ledger import Ledger, best_plan_rows

BACKGROUND = 0
CANDIDATE = 1


@dataclass(frozen=True)
class Observations:
    """One row per ``(strand, person)`` simulated."""

    person: np.ndarray  # int32
    plan: np.ndarray  # int64: store row of the executed plan
    score: np.ndarray  # float64
    role: np.ndarray  # int8: BACKGROUND | CANDIDATE
    strand: np.ndarray  # int16
    basis: np.ndarray  # int32: ledger version the strand was built from

    @classmethod
    def concat(cls, parts: Sequence["Observations"]) -> "Observations":
        return cls(*(np.concatenate([getattr(p, f) for p in parts])
                     for f in ("person", "plan", "score", "role", "strand", "basis")))

    def __len__(self) -> int:
        return len(self.person)


@dataclass(frozen=True)
class ScoreUpdate:
    """Per ``(person, plan)``: the score to store, the sample count behind
    it, and the sample variance this iteration (NaN with one sample)."""

    person: np.ndarray
    plan: np.ndarray
    value: np.ndarray
    n: np.ndarray
    var: np.ndarray


@dataclass(frozen=True)
class _Groups:
    person: np.ndarray
    plan: np.ndarray
    mean: np.ndarray
    count: np.ndarray
    var: np.ndarray
    old: np.ndarray  # stored score (NaN: unscored or new plan)
    old_n: np.ndarray  # stored sample count (0: new plan)


def _group(obs: Observations, ledger: Ledger) -> _Groups:
    # one int64 key per (person, plan): sorted by person, then plan
    key = (obs.person.astype(np.int64) << 32) | obs.plan.astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)
    m = len(uniq)
    count = np.bincount(inverse, minlength=m)
    total = np.bincount(inverse, weights=obs.score, minlength=m)
    sq = np.bincount(inverse, weights=obs.score * obs.score, minlength=m)
    mean = total / count
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(count > 1, (sq - total * mean) / (count - 1), np.nan)
    person, plan = uniq >> 32, uniq & 0xFFFFFFFF
    slots = ledger.slots_of(person, plan)
    held = slots >= 0
    pop = ledger.population
    old = np.full(m, np.nan)
    old_n = np.zeros(m, dtype=np.int64)
    old[held] = pop.scores[person[held], slots[held]]
    old_n[held] = ledger.score_n[person[held], slots[held]]
    return _Groups(person, plan, mean, count, var, old, old_n)


@runtime_checkable
class ScoreReducer(Protocol):
    def reduce(self, obs: Observations, ledger: Ledger) -> ScoreUpdate: ...


class MeanReducer:
    """The mean of a plan's samples this iteration, blended into its stored
    score with ``learning_rate`` (``(1 - lr) * old + lr * mean``, as
    :func:`~mobslim.scoring.apply_scores`); an unscored or new plan takes the
    mean. ``score_n`` accumulates the samples."""

    def __init__(self, learning_rate: float = 1.0):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        self.learning_rate = learning_rate

    def reduce(self, obs, ledger):
        g = _group(obs, ledger)
        lr = self.learning_rate
        value = np.where(np.isnan(g.old), g.mean, (1 - lr) * g.old + lr * g.mean)
        return ScoreUpdate(g.person, g.plan, value, g.old_n + g.count, g.var)


class PrecisionReducer:
    """Count-weighted: the running mean of every sample a plan received,
    ``(n_old * old + k * mean) / (n_old + k)``. ``max_count`` caps the weight
    of the past (``n_old`` counts at most that many samples), so scores keep
    tracking a changing network."""

    def __init__(self, max_count: int = 10):
        if max_count < 1:
            raise ValueError("max_count must be >= 1.")
        self.max_count = max_count

    def reduce(self, obs, ledger):
        g = _group(obs, ledger)
        n_old = np.where(np.isnan(g.old), 0, np.minimum(g.old_n, self.max_count))
        old = np.where(np.isnan(g.old), 0.0, g.old)
        value = (n_old * old + g.count * g.mean) / (n_old + g.count)
        return ScoreUpdate(g.person, g.plan, value, np.minimum(g.old_n + g.count, 2**31 - 1), g.var)


# ------------------------------------------------------------------ merge


@dataclass(frozen=True)
class MergeStats:
    new_plans: int  # candidates added (before pruning)
    pruned: int  # plans removed to respect max_plans
    best_changed: float  # share of persons whose best-scored plan changed
    # the store was compacted: the old rows kept (old row live[i] is now row i)
    live: Optional[np.ndarray] = None


def merge(ledger: Ledger, obs: Observations, reducer: ScoreReducer, max_plans: np.ndarray,
          removal, rng: np.random.Generator, selected_strand: int = 0) -> MergeStats:
    """Fold ``obs`` into ``ledger`` (in place: the ledger gets a new
    population and arrays).

    ``max_plans``: per person. ``removal``: ``[(selector, persons)]``, the
    removal selector of each group of persons. Each person's selected plan
    becomes the one they executed in strand ``selected_strand``.
    """
    pop = ledger.population
    n = len(pop)
    upd = reducer.reduce(obs, ledger)
    slots = ledger.slots_of(upd.person, upd.plan)
    held = slots >= 0
    new = ~held

    # writable tables, wide enough for every new plan
    n_plans = pop.n_plans.astype(np.int64)
    add = np.bincount(upd.person[new], minlength=n)
    width = max(int((n_plans + add).max(initial=1)), pop.plan_ids.shape[1])
    plan_ids = np.full((n, width), -1, dtype=np.int32)
    plan_ids[:, :pop.plan_ids.shape[1]] = pop.plan_ids
    scores = np.full((n, width), np.nan)
    scores[:, :pop.scores.shape[1]] = pop.scores
    score_n = np.zeros((n, width), dtype=np.int32)
    score_n[:, :ledger.score_n.shape[1]] = ledger.score_n
    selected = pop.selected.astype(np.int64).copy()

    # scores of held plans
    p, s = upd.person[held], slots[held]
    scores[p, s] = upd.value[held]
    score_n[p, s] = upd.n[held]

    # new plans: appended after each person's plans, in (person, row) order
    p_new = upd.person[new]
    if len(p_new):
        start = np.r_[0, np.flatnonzero(p_new[1:] != p_new[:-1]) + 1]
        rank = np.arange(len(p_new)) - np.repeat(start, np.diff(np.r_[start, len(p_new)]))
        s_new = n_plans[p_new] + rank
        plan_ids[p_new, s_new] = upd.plan[new]
        scores[p_new, s_new] = upd.value[new]
        score_n[p_new, s_new] = upd.n[new]
        n_plans += add

    # selected: the plan each person executed in the chosen strand
    mine = obs.strand == selected_strand
    if mine.any():
        who = obs.person[mine].astype(np.int64)
        rows = plan_ids[who] == obs.plan[mine][:, None]
        selected[who] = rows.argmax(axis=1)

    # prune each person back to max_plans
    pruned = 0
    for selector, persons in removal:
        rows = persons
        while len(rows):
            rows = rows[n_plans[rows] > max_plans[rows]]
            if not len(rows):
                break
            drop = np.asarray(selector.select_many(scores[rows], n_plans[rows], selected[rows], rng),
                              dtype=np.int64)
            _remove(rows, drop, plan_ids, scores, score_n, selected, n_plans)
            pruned += len(rows)

    best_before = ledger.best_rows
    plans, compacted, live = compact_plans(pop.plans, plan_ids, n_plans)
    population = pop.with_tables(plans, compacted, scores, selected, n_plans)
    # with_tables keeps the used width and may compact the store (renumbering rows)
    score_n = np.ascontiguousarray(score_n[:, :population.plan_ids.shape[1]])
    best_after_rows = plan_ids[np.arange(n), _best_slots(scores, n_plans, selected)]
    changed = float((best_after_rows != best_before).mean()) if n else 0.0
    ledger.population = population
    ledger.score_n = score_n
    ledger.version += 1
    ledger.best_rows = best_plan_rows(population)
    return MergeStats(int(new.sum()), pruned, changed, live)


def _best_slots(scores, n_plans, selected) -> np.ndarray:
    valid = np.arange(scores.shape[1])[None, :] < n_plans[:, None]
    filled = np.where(valid & ~np.isnan(scores), scores, -np.inf)
    return np.where(np.isfinite(filled).any(axis=1), filled.argmax(axis=1), selected)


def _remove(rows, slots, plan_ids, scores, score_n, selected, n_plans) -> None:
    """Remove slot ``slots[i]`` of person ``rows[i]``, shifting later slots
    left (as ``Person.remove_plan``)."""
    k = plan_ids.shape[1]
    cols = np.arange(k - 1)[None, :]
    src = cols + (cols >= slots[:, None])
    for table, empty in ((plan_ids, -1), (scores, np.nan), (score_n, 0)):
        block = table[rows]
        table[rows, :-1] = np.take_along_axis(block, src, axis=1)
        table[rows, -1] = empty
    sel = selected[rows]
    selected[rows] = np.where(sel == slots, 0, sel - (slots < sel))
    n_plans[rows] -= 1
