"""Scoring: turning an experienced day into a utility.

:class:`CharyparNagelScorer` follows the structure of MATSim's default
scoring function:

* Activities earn ``performing * t_typ * ln(dur / t_0)`` (hours), with
  ``t_0 = t_typ * exp(-10 h / (t_typ * priority))`` (``"uniform"``) or
  ``t_typ * exp(-1 / priority)`` (``"relative"``, MATSim's default). Below
  ``t_0`` the utility continues linearly (so very short activities are
  penalised smoothly). As in MATSim's ``calcActScore``, the activity is only
  performed while open (``opening_time``/``closing_time``; the rest is
  ``waiting``), a start after ``latest_start_time`` pays ``late_arrival``, and
  an end before ``earliest_end_time`` or a duration below ``minimal_duration``
  pays ``early_departure``.
* Overnight wrap: if the first and last activity have the same type, they
  are scored as one activity from ``last.start`` to ``first.end + day``;
  otherwise from ``0`` to ``first.end`` and from ``last.start`` to ``day``.
* Legs earn ``constant + marginal_utility_of_traveling * hours +
  (marginal_utility_of_distance + monetary_distance_rate *
  marginal_utility_of_money) * meters`` per mode, plus each used mode's
  daily constants once per day.
* A person who got stuck receives ``stuck_penalty`` on top.
* A pt leg (one leg per trip, walks implied) earns, with ``pt`` its mode's
  parameters and ``walk`` the ``"walk"`` mode's::

      pt.constant + pt.beta_travel * in-vehicle h + waiting_pt * wait h
      + walk.beta_travel * walk h + walk.beta_distance * walk m
      + pt.beta_distance * ride m + line_switch * max(rides - 1, 0)

  where walk time is the leg's travel time minus waiting and riding. This is
  MATSim's sum over the expanded legs, without the walk mode's constant.

Money events (tolls, fares) are not modelled. Add them by writing another
:class:`Scorer`.

A scorer may also implement ``score_trajectories(trajectories) -> array``,
which scores every person at once from the mobsim's arrays. The controller
uses it when present (``CharyparNagelScorer`` does, with numpy, in one pass
over all persons) and falls back to ``score`` on each
:class:`ExperiencedPlan` otherwise.

Person attributes: if ``score_trajectories`` (or ``score``) takes an
``attributes`` keyword, :func:`score_all` passes the population's
:class:`~mobslim.attributes.PersonAttributes` (or, to ``score``, the
person's attributes as a dict). ``CharyparNagelScorer`` resolves them into
:class:`~mobslim.scoring.person.PersonParams`: a parameter set per person
(per-subpopulation :class:`ScoringParams`, MATSim's ``scoringParameters``
sets with a ``subpopulation``) plus per-person overrides of any parameter
from ``person_rules`` (see :mod:`mobslim.scoring.person`).

Modules: :mod:`~mobslim.scoring.params` (parameters, field paths, compiled
tables), :mod:`~mobslim.scoring.person` (person parameters and rules),
:mod:`~mobslim.scoring.charypar_nagel` (the scorer).
"""

import inspect
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from mobslim.attributes import PersonAttributes
from mobslim.experience import ExperiencedPlan
from mobslim.population import Population
from mobslim.scoring.charypar_nagel import CharyparNagelScorer
from mobslim.scoring.params import (
    ACT_FIELDS,
    MODE_FIELDS,
    SCALAR_FIELDS,
    ActivityParams,
    ModeParams,
    ScoringParams,
    parse_field,
)
from mobslim.scoring.person import (
    FromAttributes,
    IncomeDependentMoney,
    ModeConstants,
    PersonParams,
    PersonParamsBuilder,
    PersonScoringRule,
    Subpopulation,
    numeric_column,
    resolve_person_params,
)
from mobslim.trajectories import Trajectories

__all__ = [
    "ACT_FIELDS",
    "MODE_FIELDS",
    "SCALAR_FIELDS",
    "ActivityParams",
    "CharyparNagelScorer",
    "FromAttributes",
    "IncomeDependentMoney",
    "ModeConstants",
    "ModeParams",
    "PersonParams",
    "PersonParamsBuilder",
    "PersonScoringRule",
    "Scorer",
    "ScoringParams",
    "Subpopulation",
    "apply_scores",
    "numeric_column",
    "parse_field",
    "resolve_person_params",
    "score_all",
]


@runtime_checkable
class Scorer(Protocol):
    def score(self, plan: ExperiencedPlan) -> float: ...


def _takes_attributes(fn) -> bool:
    try:
        return "attributes" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def score_all(scorer: Scorer, trajectories: Trajectories,
              attributes: Optional[PersonAttributes] = None) -> np.ndarray:
    """Score every person: vectorised if the scorer supports it.

    ``attributes`` (the population's person attributes) are passed on to
    scorers whose ``score_trajectories``/``score`` take an ``attributes``
    keyword.
    """
    batch = getattr(scorer, "score_trajectories", None)
    if batch is not None:
        if attributes is not None and _takes_attributes(batch):
            return np.asarray(batch(trajectories, attributes=attributes), dtype=float)
        return np.asarray(batch(trajectories), dtype=float)
    plans = trajectories.experienced_plans()
    if attributes is not None and _takes_attributes(scorer.score):
        return np.array([scorer.score(p, attributes=attributes.row(i))
                         for i, p in enumerate(plans)], dtype=float)
    return np.array([scorer.score(p) for p in plans], dtype=float)


def apply_scores(
    population: Population, scores: Sequence[float], learning_rate: float = 1.0
) -> Population:
    """Store each person's observed score for their selected plan.

    With ``learning_rate < 1`` the new score is averaged with the old one:
    ``(1 - learning_rate) * old + learning_rate * observed`` (MATSim's
    ``BrainExpBeta``-style learning). ``scores`` is indexed by person index.
    """
    observed = np.asarray(scores, dtype=float)
    if observed.shape != (len(population),):
        raise ValueError(f"Expected {len(population)} scores, got {len(observed)}.")
    old = population.selected_scores()
    lr = learning_rate
    new = np.where(np.isnan(old), observed, (1 - lr) * old + lr * observed)
    return population.with_selected_scores(new)
