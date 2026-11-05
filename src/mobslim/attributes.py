"""Person attributes: a column table aligned with the population's person index.

MATSim keeps arbitrary per-person attributes (``subpopulation``, ``age``,
``carAvail``, income, ...) in an ``<attributes>`` block of each ``<person>``
(``population_v6``), as XML attributes of ``<person>`` (``plans_v4``), or in
a separate ``objectAttributes`` file (``plans.inputPersonAttributesFile``).
MobSlim stores them as columns: one numpy array per attribute, row ``i``
belonging to the person with person index ``i``. Code that consumes them
(scorers, mobsims, analysis) reads whole columns, as the rest of MobSlim
reads its arrays.

Column dtypes are inferred from the values:

=======================  ===========  ===================
values                   dtype        missing value
=======================  ===========  ===================
``bool``                 ``bool``     (object if missing)
``int`` (no missing)     ``int64``    --
``int``/``float``        ``float64``  ``NaN``
anything else (``str``)  ``object``   ``None``
=======================  ===========  ===================

A boolean column with missing values is an ``object`` column of
``True``/``False``/``None``. :meth:`PersonAttributes.missing` gives the mask
of missing values for any column.

The table is immutable: columns are read-only and updates return a new table.
"""

from __future__ import annotations

import math
from typing import Any, Hashable, Iterable, Iterator, Mapping, Optional

import numpy as np

from mobslim.ids import IdMap


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def column_array(values: Iterable[Any]) -> np.ndarray:
    """A read-only column from Python values, with the dtype inferred as
    described in the module docstring."""
    values = list(values)
    present = [v for v in values if not _missing(v)]
    has_missing = len(present) < len(values)
    if present and all(isinstance(v, (bool, np.bool_)) for v in present):
        col = (np.array(values, dtype=object) if has_missing
               else np.array(values, dtype=bool))
    elif present and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                         for v in present) and not has_missing:
        col = np.array(values, dtype=np.int64)
    elif present and all(isinstance(v, (int, float, np.integer, np.floating))
                         and not isinstance(v, (bool, np.bool_)) for v in present):
        col = np.array([math.nan if _missing(v) else float(v) for v in values], dtype=float)
    else:
        col = np.empty(len(values), dtype=object)
        col[:] = [None if _missing(v) else v for v in values]
    col.flags.writeable = False
    return col


def _as_column(values, n: int, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype != object:
        col = values.copy() if values.flags.writeable else values
        if col.ndim != 1:
            raise ValueError(f"Attribute {name!r} must be one-dimensional.")
        col.flags.writeable = False
    else:
        col = column_array(values)
    if len(col) != n:
        raise ValueError(f"Attribute {name!r} has {len(col)} values for {n} persons.")
    return col


class PersonAttributes:
    """Per-person attributes: ``{name: column}``, each column one value per
    person index.

    ``attrs["age"]`` (or ``attrs.column("age")``) is the read-only column;
    ``attrs.get("age")`` returns ``None`` if there is no such attribute.
    ``attrs.row(i)`` is person ``i``'s attributes as a dict (missing values
    left out), ``attrs.to_frame(ids)`` a pandas ``DataFrame``.
    """

    __slots__ = ("n_persons", "_columns")

    def __init__(self, n_persons: int, columns: Mapping[str, Any] = ()):
        self.n_persons = int(n_persons)
        self._columns: dict[str, np.ndarray] = {
            str(name): _as_column(values, self.n_persons, name)
            for name, values in dict(columns).items()
        }

    # -- construction -----------------------------------------------------------
    @classmethod
    def empty(cls, n_persons: int) -> "PersonAttributes":
        return cls(n_persons)

    @classmethod
    def from_records(cls, ids: IdMap,
                     records: Mapping[Hashable, Mapping[str, Any]]) -> "PersonAttributes":
        """From ``{person_id: {name: value}}``. Persons without a record, or
        without some attribute, get missing values; records of persons not in
        ``ids`` are ignored (as MATSim ignores them)."""
        names: dict[str, None] = {}
        for rec in records.values():
            names.update(dict.fromkeys(rec))
        rows = [records.get(pid, {}) for pid in ids]
        return cls(len(ids), {name: [r.get(name) for r in rows] for name in names})

    @classmethod
    def from_frame(cls, frame, ids: IdMap) -> "PersonAttributes":
        """From a pandas ``DataFrame`` indexed by person id (each column an
        attribute). Persons missing from the frame get missing values."""
        frame = frame.reindex(list(ids))
        return cls(len(ids), {c: frame[c].tolist() for c in frame.columns})

    @classmethod
    def coerce(cls, value, ids: IdMap) -> "PersonAttributes":
        """``None``, a :class:`PersonAttributes`, ``{person_id: {name: value}}``
        records, or a pandas ``DataFrame`` indexed by person id."""
        if value is None:
            return cls.empty(len(ids))
        if isinstance(value, PersonAttributes):
            if value.n_persons != len(ids):
                raise ValueError(f"PersonAttributes has {value.n_persons} rows for {len(ids)} persons.")
            return value
        if hasattr(value, "to_dict") and hasattr(value, "columns"):
            return cls.from_frame(value, ids)
        return cls.from_records(ids, value)

    # -- updates --------------------------------------------------------------
    def with_column(self, name: str, values) -> "PersonAttributes":
        """A table with ``name`` added or replaced."""
        return PersonAttributes(self.n_persons, {**self._columns, name: values})

    def without(self, *names: str) -> "PersonAttributes":
        return PersonAttributes(self.n_persons,
                                {k: v for k, v in self._columns.items() if k not in names})

    def merged(self, other: "PersonAttributes") -> "PersonAttributes":
        """Both tables' columns; where both have a value, ``other``'s wins."""
        if other.n_persons != self.n_persons:
            raise ValueError("Cannot merge attribute tables of different sizes.")
        cols = dict(self._columns)
        for name, col in other._columns.items():
            if name in cols:
                keep = other.missing(name)
                col = [a if m else b for a, b, m in zip(cols[name].tolist(), col.tolist(), keep.tolist())]
            cols[name] = col
        return PersonAttributes(self.n_persons, cols)

    # -- reading --------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def column(self, name: str) -> np.ndarray:
        try:
            return self._columns[name]
        except KeyError:
            raise KeyError(f"No person attribute {name!r}.") from None

    __getitem__ = column

    def get(self, name: str, default=None) -> Optional[np.ndarray]:
        return self._columns.get(name, default)

    def missing(self, name: str) -> np.ndarray:
        """Boolean mask: persons without a value for ``name`` (all ``True``
        if there is no such attribute)."""
        col = self._columns.get(name)
        if col is None:
            return np.ones(self.n_persons, dtype=bool)
        if col.dtype == object:
            return np.array([_missing(v) for v in col.tolist()], dtype=bool)
        if col.dtype.kind == "f":
            return np.isnan(col)
        return np.zeros(self.n_persons, dtype=bool)

    def value(self, index: int, name: str, default=None):
        """Person ``index``'s value of ``name`` (``default`` if missing)."""
        col = self._columns.get(name)
        if col is None:
            return default
        v = col[index].item() if col.dtype != object else col[index]
        return default if _missing(v) else v

    def row(self, index: int) -> dict[str, Any]:
        """Person ``index``'s attributes (missing values left out)."""
        out = {}
        for name, col in self._columns.items():
            v = col[index].item() if col.dtype != object else col[index]
            if not _missing(v):
                out[name] = v
        return out

    def to_frame(self, ids: Optional[IdMap] = None):
        """A pandas ``DataFrame``: one row per person (indexed by person id if
        ``ids`` is given, else by person index), one column per attribute."""
        import pandas as pd

        index = pd.Index(list(ids), name="person") if ids is not None else None
        return pd.DataFrame(dict(self._columns), index=index)

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __eq__(self, other):
        if not isinstance(other, PersonAttributes):
            return NotImplemented
        def same(a, b):
            if a.dtype.kind == "f" and b.dtype.kind == "f":
                return np.array_equal(a, b, equal_nan=True)
            return a.dtype == b.dtype and a.tolist() == b.tolist()

        return (self.n_persons == other.n_persons and self.names == other.names
                and all(same(self[n], other[n]) for n in self.names))

    __hash__ = None  # type: ignore[assignment]

    def __getstate__(self):
        return (self.n_persons, self._columns)

    def __setstate__(self, state):
        self.n_persons, columns = state
        for col in columns.values():
            col.flags.writeable = False
        self._columns = columns

    def __repr__(self) -> str:
        return f"PersonAttributes(persons={self.n_persons}, columns={list(self._columns)})"


# ------------------------------------------------------------------- MATSim
