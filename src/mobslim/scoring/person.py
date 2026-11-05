"""Person-specific scoring parameters.

Each person is scored with one parameter *set* (a :class:`ScoringParams`:
the scorer's default, or a subpopulation's) plus any number of per-person
*overrides*. An override is an absolute value for one field path (see
:mod:`mobslim.scoring.params`), such as ``"marginal_utility_of_money"`` or
``"modes.walk.marginal_utility_of_traveling"``, stored as a column with one
value per person index. ``NaN`` means "the set's value".

:class:`PersonParams` is the resolved form a scorer consumes. It is built
once per run (person attributes do not change) by applying *rules* in
order to a :class:`PersonParamsBuilder`::

    scorer = CharyparNagelScorer(
        params,
        subpopulations={"freight": freight_params},
        person_rules=[IncomeDependentMoney(), ModeConstants()],
    )

A rule is any object with ``apply(builder)``, or a plain callable
``rule(builder)``. It reads attribute columns (``builder.attributes``) and
the values so far (``builder.current(field)``), and writes with
``builder.use_set(name, where)`` and ``builder.set(field, values, where)``.
Rules work on whole columns, never row by row.

Built-in rules:

* :class:`Subpopulation`: pick the set from a categorical attribute.
* :class:`IncomeDependentMoney`: marginal utility of money from income.
* :class:`ModeConstants`: per-person mode constants from ``asc_<mode>`` columns.
* :class:`FromAttributes`: any field from a function of the attributes.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from mobslim.attributes import PersonAttributes
from mobslim.scoring.params import (
    SCALAR_FIELDS,
    FieldPath,
    ScoringParams,
    field_value,
    parse_field,
    scalar_table,
)

FieldLike = Union[str, FieldPath]


def _path(f: FieldLike) -> FieldPath:
    return f if isinstance(f, FieldPath) else parse_field(f)


def numeric_column(attributes: PersonAttributes, name: str) -> np.ndarray:
    """Attribute ``name`` as float64, ``NaN`` where missing (all ``NaN`` if
    there is no such attribute). Raises ``ValueError`` for non-numeric values."""
    col = attributes.get(name)
    if col is None:
        return np.full(attributes.n_persons, np.nan)
    if col.dtype != object:
        return col.astype(float)
    missing = attributes.missing(name)
    try:
        return np.array([math.nan if m else float(v) for v, m in zip(col.tolist(), missing.tolist())],
                        dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"Person attribute {name!r} is not numeric.") from None


class PersonParams:
    """Resolved per-person scoring parameters.

    ``sets[set_index[i]]`` is person ``i``'s parameter set (``sets[0]`` the
    scorer's default, ``set_names[0]`` is ``None``); ``overrides`` maps a
    :class:`FieldPath` to a float64 column (``NaN``: the set's value).
    """

    __slots__ = ("sets", "set_names", "set_index", "overrides", "_scalars", "_constant")

    def __init__(self, sets: Sequence[ScoringParams], set_names: Sequence[Optional[str]],
                 set_index, overrides: Mapping[FieldLike, np.ndarray] = ()):
        self.sets = tuple(sets)
        self.set_names = tuple(set_names)
        self.set_index = np.array(set_index, dtype=np.int32)  # a copy: builders keep theirs
        n = len(self.set_index)
        if len(self.set_names) != len(self.sets):
            raise ValueError("PersonParams: one name per set.")
        if n and (self.set_index.min() < 0 or self.set_index.max() >= len(self.sets)):
            raise ValueError("PersonParams: set index out of range.")
        self.set_index.flags.writeable = False
        out = {}
        for f, col in dict(overrides).items():
            col = np.asarray(col, dtype=float)
            if col.shape != (n,):
                raise ValueError(f"Override {f} has shape {col.shape}, expected ({n},).")
            if np.isnan(col).all():
                continue
            col = col.copy()
            col.flags.writeable = False
            out[_path(f)] = col
        self.overrides: dict[FieldPath, np.ndarray] = out
        self._scalars = None
        self._constant = None

    @classmethod
    def uniform(cls, params: ScoringParams, n_persons: int) -> "PersonParams":
        """Everyone with ``params`` and no overrides."""
        return cls((params,), (None,), np.zeros(n_persons, dtype=np.int32))

    @property
    def n_persons(self) -> int:
        return len(self.set_index)

    def __len__(self) -> int:
        return self.n_persons

    @property
    def is_uniform(self) -> bool:
        """Everyone uses ``sets[0]`` with no overrides."""
        return not self.overrides and not self.set_index.any()

    def value(self, f: FieldLike) -> np.ndarray:
        """Each person's effective value of field ``f`` (``NaN`` for an
        activity type their set has no params for)."""
        path = _path(f)
        base = np.array([field_value(p, path) for p in self.sets], dtype=float)[self.set_index]
        col = self.overrides.get(path)
        if col is not None:
            apply = ~np.isnan(col)
            if path.kind == "activities":  # overrides don't create activity params
                has = np.array([p.activities.get(path.name, p.default_activity) is not None
                                for p in self.sets])
                apply &= has[self.set_index]
            base = np.where(apply, col, base)
        return base

    @property
    def scalars(self) -> np.ndarray:
        """``(n_persons, len(SCALAR_FIELDS))``: effective scalar params."""
        if self._scalars is None:
            out = scalar_table(self.sets)[self.set_index]
            for path, col in self.overrides.items():
                if path.kind == "scalar":
                    j = SCALAR_FIELDS.index(path.field)
                    ok = ~np.isnan(col)
                    out[ok, j] = col[ok]
            out.flags.writeable = False
            self._scalars = out
        return self._scalars

    def scalar(self, f: str, persons):
        """Scalar param ``f`` of ``persons`` (an index array or slice): a
        float if it is the same for everyone, else one value per person."""
        j = SCALAR_FIELDS.index(f)
        if self._constant is None:
            sc = self.scalars
            self._constant = (sc == sc[:1]).all(axis=0) if len(sc) else np.ones(sc.shape[1], dtype=bool)
        if self._constant[j]:
            return float(self.scalars[0, j]) if self.n_persons else float(getattr(self.sets[0], f))
        return self.scalars[persons, j]

    def of_kind(self, kind: str) -> list[tuple[str, str, np.ndarray]]:
        """``[(name, field, column)]`` of the ``"modes"`` or ``"activities"`` overrides."""
        return [(p.name, p.field, col) for p, col in self.overrides.items() if p.kind == kind]

    def params_of(self, index: int) -> ScoringParams:
        """Person ``index``'s parameters as one :class:`ScoringParams`.

        An activity override only applies to types the person's set has
        params for (it does not create params)."""
        p = self.sets[int(self.set_index[index])]
        scalars, modes, acts = {}, dict(p.modes), dict(p.activities)
        for path, col in self.overrides.items():
            v = float(col[index])
            if math.isnan(v):
                continue
            if path.kind == "scalar":
                scalars[path.field] = v
            elif path.kind == "modes":
                modes[path.name] = replace(modes.get(path.name, p.default_mode), **{path.field: v})
            else:
                base = acts.get(path.name, p.default_activity)
                if base is not None:
                    acts[path.name] = replace(base, **{path.field: v})
        if not scalars and modes == p.modes and acts == p.activities:
            return p
        return replace(p, modes=modes, activities=acts, **scalars)

    def __repr__(self) -> str:
        return (f"PersonParams(persons={self.n_persons}, sets={list(self.set_names)}, "
                f"overrides={[str(p) for p in self.overrides]})")


class PersonParamsBuilder:
    """What rules write to; see the module docstring."""

    def __init__(self, attributes: PersonAttributes, sets: Sequence[ScoringParams],
                 set_names: Sequence[Optional[str]]):
        self.attributes = attributes
        self.sets = tuple(sets)
        self.set_names = tuple(set_names)
        self.set_index = np.zeros(attributes.n_persons, dtype=np.int32)
        self._overrides: dict[FieldPath, np.ndarray] = {}

    @property
    def n_persons(self) -> int:
        return self.attributes.n_persons

    def _mask(self, where) -> np.ndarray:
        if where is None:
            return np.ones(self.n_persons, dtype=bool)
        where = np.asarray(where, dtype=bool)
        if where.shape != (self.n_persons,):
            raise ValueError(f"where has shape {where.shape}, expected ({self.n_persons},).")
        return where

    def use_set(self, name: Optional[str], where=None) -> None:
        """Persons in ``where`` use the set called ``name`` (``None``: the default)."""
        try:
            code = self.set_names.index(name)
        except ValueError:
            raise KeyError(f"No ScoringParams for subpopulation {name!r}.") from None
        self.set_index[self._mask(where)] = code

    def current(self, f: FieldLike) -> np.ndarray:
        """Each person's value of ``f`` so far."""
        return self.build().value(f)

    def set(self, f: FieldLike, values, where=None) -> None:
        """Override ``f`` with ``values`` (a scalar or one value per person)
        for persons in ``where``. ``NaN`` values leave a person unchanged."""
        path = _path(f)
        values = np.broadcast_to(np.asarray(values, dtype=float), (self.n_persons,))
        col = self._overrides.get(path)
        if col is None:
            col = self._overrides[path] = np.full(self.n_persons, np.nan)
        ok = self._mask(where) & ~np.isnan(values)
        col[ok] = values[ok]

    def build(self) -> PersonParams:
        return PersonParams(self.sets, self.set_names, self.set_index, self._overrides)


Rule = Union[Callable[[PersonParamsBuilder], None], "PersonScoringRule"]


def apply_rule(rule, builder: PersonParamsBuilder) -> None:
    apply = getattr(rule, "apply", None)
    (apply or rule)(builder)


def resolve_person_params(attributes: PersonAttributes, sets: Sequence[ScoringParams],
                          set_names: Sequence[Optional[str]], rules: Sequence) -> PersonParams:
    """Apply ``rules`` in order, starting from everyone on ``sets[0]``."""
    b = PersonParamsBuilder(attributes, sets, set_names)
    for rule in rules:
        apply_rule(rule, b)
    return b.build()


# ------------------------------------------------------------------- rules

class PersonScoringRule:
    """Base class for rules (any object with ``apply(builder)`` works)."""

    def apply(self, b: PersonParamsBuilder) -> None:  # pragma: no cover
        raise NotImplementedError


class Subpopulation(PersonScoringRule):
    """Persons whose ``attribute`` names a set use that set (MATSim's
    ``scoringParameters`` with a ``subpopulation``). Persons without the
    attribute keep theirs; a value with no set is a ``KeyError``."""

    def __init__(self, attribute: str = "subpopulation"):
        self.attribute = attribute

    def apply(self, b: PersonParamsBuilder) -> None:
        col = b.attributes.get(self.attribute)
        if col is None:
            return
        present = ~b.attributes.missing(self.attribute)
        names = dict.fromkeys(v for v, ok in zip(col.tolist(), present.tolist()) if ok)
        for name in names:  # check all before changing anyone
            if name not in b.set_names:
                raise KeyError(f"No ScoringParams for subpopulation {name!r}.")
        for name in names:
            b.use_set(name, present & (col == name))

    def __repr__(self) -> str:
        return f"Subpopulation({self.attribute!r})"


class IncomeDependentMoney(PersonScoringRule):
    """``marginal_utility_of_money = m * (reference / income) ** elasticity``,
    with ``m`` the person's value so far, after MATSim's
    ``IncomeDependentUtilityOfMoneyPersonScoringParameters``.

    ``reference`` defaults to the mean income of the persons who have one
    (of the attribute table given, so of the whole population when scoring
    trajectories). Persons with a missing or non-positive income keep ``m``.
    """

    def __init__(self, attribute: str = "income", elasticity: float = 1.0,
                 reference: Optional[float] = None):
        self.attribute = attribute
        self.elasticity = float(elasticity)
        self.reference = reference

    def apply(self, b: PersonParamsBuilder) -> None:
        income = numeric_column(b.attributes, self.attribute)
        valid = ~np.isnan(income) & (income > 0)
        if not valid.any():
            return
        ref = float(income[valid].mean()) if self.reference is None else float(self.reference)
        safe = np.where(valid, income, 1.0)
        money = b.current("marginal_utility_of_money") * (ref / safe) ** self.elasticity
        b.set("marginal_utility_of_money", money, where=valid)

    def __repr__(self) -> str:
        return (f"IncomeDependentMoney({self.attribute!r}, elasticity={self.elasticity}, "
                f"reference={self.reference})")


class ModeConstants(PersonScoringRule):
    """Each numeric attribute ``<prefix><mode>`` is added to that mode's
    ``constant`` (per-person taste variations, e.g. from a mixed logit).
    Missing values add nothing."""

    def __init__(self, prefix: str = "asc_"):
        self.prefix = prefix

    def apply(self, b: PersonParamsBuilder) -> None:
        for name in b.attributes.names:
            if not name.startswith(self.prefix) or len(name) == len(self.prefix):
                continue
            delta = numeric_column(b.attributes, name)
            field = f"modes.{name[len(self.prefix):]}.constant"
            b.set(field, b.current(field) + delta)

    def __repr__(self) -> str:
        return f"ModeConstants({self.prefix!r})"


class FromAttributes(PersonScoringRule):
    """``field`` from ``fn(attributes, current) -> values`` (one per person;
    ``NaN`` keeps a person's value). ``current`` is each person's value so
    far, so ``fn`` can scale it::

        FromAttributes("modes.walk.marginal_utility_of_traveling",
                       lambda a, cur: np.where(numeric_column(a, "age") > 70, 2 * cur, np.nan))
    """

    def __init__(self, field: str, fn: Callable[[PersonAttributes, np.ndarray], np.ndarray]):
        self.field = parse_field(field)
        self.fn = fn

    def apply(self, b: PersonParamsBuilder) -> None:
        b.set(self.field, self.fn(b.attributes, b.current(self.field)))

    def __repr__(self) -> str:
        return f"FromAttributes({str(self.field)!r}, {self.fn!r})"
