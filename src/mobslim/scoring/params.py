"""Scoring parameters and their compiled array form.

:class:`ScoringParams` (with :class:`ActivityParams` and :class:`ModeParams`)
is one parameter *set*, as MATSim's ``scoringParameters``. Scorers compile
the sets they use into dense tables (:func:`mode_table`, :func:`act_table`,
:func:`scalar_table`) indexed by set, so that every person can be scored in
one vectorised pass whatever set they use.

Every numeric parameter has a *field path*, the name person-specific
overrides use (see :mod:`mobslim.scoring.person`):

* ``"performing"``, ``"marginal_utility_of_money"``, ... (:data:`SCALAR_FIELDS`)
* ``"modes.<mode>.<field>"``, e.g. ``"modes.walk.marginal_utility_of_traveling"``
  (:data:`MODE_FIELDS`)
* ``"activities.<type>.<field>"``, e.g. ``"activities.work.typical_duration"``
  (:data:`ACT_FIELDS`)
"""

import math
from dataclasses import dataclass, field, fields
from typing import Mapping, NamedTuple, Optional, Sequence

import numpy as np

from mobslim.timeutils import DAY, HOUR


@dataclass(frozen=True)
class ActivityParams:
    """Utility parameters of one activity type (MATSim's ``activityParams``).

    Times are seconds of the day; ``None`` means not set (no constraint).
    ``zero_utility`` picks how ``t_0`` (the duration of zero utility) is
    computed: ``"uniform"`` ``t_typ * exp(-10 h / (t_typ * priority))`` or
    ``"relative"`` ``t_typ * exp(-1 / priority)`` (MATSim's default
    ``typicalDurationScoreComputation``).
    """

    typical_duration: float  # seconds
    priority: float = 1.0
    zero_utility: str = "uniform"
    opening_time: Optional[float] = None
    closing_time: Optional[float] = None
    latest_start_time: Optional[float] = None  # later starts pay late_arrival
    earliest_end_time: Optional[float] = None  # earlier ends pay early_departure
    minimal_duration: Optional[float] = None  # shorter durations pay early_departure

    def __post_init__(self):
        if self.zero_utility not in ("uniform", "relative"):
            raise ValueError(f"zero_utility must be 'uniform' or 'relative', got {self.zero_utility!r}.")

    @property
    def zero_utility_duration(self) -> float:
        """``t_0`` in seconds (as :func:`zero_utility_duration`)."""
        t = self.typical_duration
        if self.zero_utility == "relative":
            return t * math.exp(-1.0 / self.priority)
        return t * math.exp(-10.0 * HOUR / (t * self.priority))


@dataclass(frozen=True)
class ModeParams:
    constant: float = 0.0  # per leg
    marginal_utility_of_traveling: float = -6.0  # util / hour
    marginal_utility_of_distance: float = 0.0  # util / meter
    monetary_distance_rate: float = 0.0  # money / meter (times marginal_utility_of_money)
    daily_constant: float = 0.0  # util, once per day if the mode is used
    daily_monetary_constant: float = 0.0  # money, once per day if the mode is used


@dataclass(frozen=True)
class ScoringParams:
    performing: float = 6.0  # util / hour
    activities: Mapping[str, ActivityParams] = field(default_factory=dict)
    modes: Mapping[str, ModeParams] = field(default_factory=dict)
    default_activity: Optional[ActivityParams] = ActivityParams(8 * HOUR)
    default_mode: ModeParams = ModeParams()
    stuck_penalty: float = -100.0
    day_length: float = DAY
    marginal_utility_of_waiting_pt: float = -6.0  # util / hour (MATSim's waitingPt)
    utility_of_line_switch: float = -1.0  # per pt leg after the first in a trip
    walk_mode: str = "walk"  # whose parameters score a pt trip's walks
    late_arrival: float = -18.0  # util / hour after an activity's latest_start_time
    early_departure: float = 0.0  # util / hour before earliest_end_time / minimal_duration
    waiting: float = 0.0  # util / hour at an activity outside its opening times
    marginal_utility_of_money: float = 1.0

    def activity(self, act_type: str) -> ActivityParams:
        params = self.activities.get(act_type, self.default_activity)
        if params is None:
            raise KeyError(f"No ActivityParams for activity type '{act_type}'.")
        return params

    def mode(self, mode: str) -> ModeParams:
        return self.modes.get(mode, self.default_mode)


# ------------------------------------------------------------------ fields

SCALAR_FIELDS: tuple[str, ...] = (
    "performing", "stuck_penalty", "day_length", "marginal_utility_of_waiting_pt",
    "utility_of_line_switch", "late_arrival", "early_departure", "waiting",
    "marginal_utility_of_money",
)
MODE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ModeParams))
# numeric ActivityParams fields (zero_utility is per set only)
ACT_FIELDS: tuple[str, ...] = (
    "typical_duration", "priority", "opening_time", "closing_time",
    "latest_start_time", "earliest_end_time", "minimal_duration",
)
# act_table columns: ACT_FIELDS, then 1.0 if zero_utility is "relative", then t_0 [s]
ACT_RELATIVE = len(ACT_FIELDS)
ACT_T0 = ACT_RELATIVE + 1
ACT_COLUMNS = ACT_T0 + 1
# act_table columns that are optional constraints (NaN = not set)
ACT_OPTIONAL = ACT_FIELDS[2:]

S = {name: i for i, name in enumerate(SCALAR_FIELDS)}
M = {name: i for i, name in enumerate(MODE_FIELDS)}
A = {name: i for i, name in enumerate(ACT_FIELDS)}


class FieldPath(NamedTuple):
    """A parsed field path: ``kind`` is ``"scalar"``, ``"modes"`` or
    ``"activities"``; ``name`` the mode or activity type (``None`` for
    scalars); ``field`` the parameter."""

    kind: str
    name: Optional[str]
    field: str

    def __str__(self) -> str:
        return self.field if self.kind == "scalar" else f"{self.kind}.{self.name}.{self.field}"


def parse_field(path: str) -> FieldPath:
    """``"performing"``, ``"modes.car.constant"`` or
    ``"activities.work.typical_duration"`` -> :class:`FieldPath`. Mode and
    activity names may contain dots."""
    if path in S:
        return FieldPath("scalar", None, path)
    kind, _, rest = path.partition(".")
    name, _, fld = rest.rpartition(".")
    if kind == "modes" and name and fld in M:
        return FieldPath("modes", name, fld)
    if kind == "activities" and name and fld in A:
        return FieldPath("activities", name, fld)
    raise KeyError(f"Unknown scoring field {path!r}: expected one of {list(SCALAR_FIELDS)}, "
                   f"'modes.<mode>.<{'|'.join(MODE_FIELDS)}>' or "
                   f"'activities.<type>.<{'|'.join(ACT_FIELDS)}>'.")


def field_value(params: ScoringParams, path: FieldPath) -> float:
    """The value of ``path`` in ``params`` (``NaN``: not set, or an activity
    type without params)."""
    if path.kind == "scalar":
        return float(getattr(params, path.field))
    if path.kind == "modes":
        return float(getattr(params.mode(path.name), path.field))
    act = params.activities.get(path.name, params.default_activity)
    if act is None:
        return math.nan
    v = getattr(act, path.field)
    return math.nan if v is None else float(v)


# ------------------------------------------------------------------ tables

def zero_utility_duration(t_typ: np.ndarray, priority: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """``t_0`` [s] per element (see :class:`ActivityParams`)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(relative, t_typ * np.exp(-1.0 / priority),
                        t_typ * np.exp(-10.0 * HOUR / (t_typ * priority)))


def scalar_table(sets: Sequence[ScoringParams]) -> np.ndarray:
    """``(n_sets, len(SCALAR_FIELDS))``."""
    return np.array([[float(getattr(p, f)) for f in SCALAR_FIELDS] for p in sets],
                    dtype=float).reshape(len(sets), len(SCALAR_FIELDS))


def mode_table(sets: Sequence[ScoringParams], modes: Sequence[str]) -> np.ndarray:
    """``(n_sets, n_modes, len(MODE_FIELDS))``: each set's params of each mode."""
    return np.array([[[float(getattr(p.mode(m), f)) for f in MODE_FIELDS] for m in modes]
                     for p in sets], dtype=float).reshape(len(sets), len(modes), len(MODE_FIELDS))


def walk_table(sets: Sequence[ScoringParams]) -> np.ndarray:
    """``(n_sets, len(MODE_FIELDS))``: each set's params of its ``walk_mode``."""
    return np.array(
        [[float(getattr(p.mode(p.walk_mode), f)) for f in MODE_FIELDS] for p in sets],
        dtype=float).reshape(len(sets), len(MODE_FIELDS))


def act_row(params: Optional[ActivityParams]) -> list[float]:
    """One :func:`act_table` row (all ``NaN`` for ``None``)."""
    if params is None:
        return [math.nan] * ACT_COLUMNS
    row = [math.nan if getattr(params, f) is None else float(getattr(params, f)) for f in ACT_FIELDS]
    return row + [1.0 if params.zero_utility == "relative" else 0.0, params.zero_utility_duration]


def act_table(sets: Sequence[ScoringParams], act_types: Sequence[str]) -> np.ndarray:
    """``(n_sets, n_types, ACT_COLUMNS)``: :data:`ACT_FIELDS`, a ``relative``
    flag and ``t_0``. A type a set has no params for is a row of ``NaN``."""
    return np.array([[act_row(p.activities.get(t, p.default_activity)) for t in act_types]
                     for p in sets], dtype=float).reshape(len(sets), len(act_types), ACT_COLUMNS)
