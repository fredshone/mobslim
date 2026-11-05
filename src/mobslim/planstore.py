"""Every stored plan of a population, as flat columns.

A :class:`PlanStore` holds plans in the same CSR layout as a
:class:`~mobslim.planbatch.PlanBatch` (plan ``p``'s activities are
``act_offsets[p]:act_offsets[p + 1]``, its legs the slice of the implied
``leg_offsets``, a leg's links ``route_offsets[l]:route_offsets[l + 1]``,
its pt rides ``ride_offsets[l]:ride_offsets[l + 1]``) plus each route's
expected travel time and distance. A population's ``plan_ids`` index its rows.

**Immutable, append-only.** A store is a snapshot: its columns are read-only
views of the first ``n`` rows of shared buffers. :meth:`append` returns a new
snapshot. If the store is the newest snapshot of its *lineage* (the tip), the
new rows are written into spare buffer capacity past its end, which older
snapshots never read, so appending costs only the new rows (buffers grow by
doubling). Appending to an older snapshot copies it into a new lineage first.
Rows never change once written, so any snapshot of a lineage sees a prefix of
the same rows: ``(lineage token, row)`` identifies a plan across snapshots
(and processes), which is how :class:`~mobslim.population.Population` knows
which plans of updated persons are already stored.

**Compaction.** Plans that no person holds any more stay in the store until
:meth:`compact` copies the live rows (a vectorised gather) into a new lineage.

**Views.** ``store[p]`` is a :class:`~mobslim.population.Plan` that builds its
activities and legs from the columns on first access; :meth:`take` gathers
rows into a :class:`~mobslim.planbatch.PlanBatch` without building any
Python objects (how a mobsim gets the selected plans).

Activity types and modes are stored as codes into ``act_types`` and
``modes``; a lineage only ever extends them, so codes stay valid.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from mobslim.planbatch import _I32, PlanBatch

# column -> (unit it has one value per, dtype, trailing shape)
_COLUMNS = {
    "act_type": ("act", np.int32, ()),
    "act_node": ("act", np.int32, ()),
    "act_link": ("act", np.int32, ()),
    "act_xy": ("act", np.float64, (2,)),
    "act_end_time": ("act", np.float64, ()),
    "act_duration": ("act", np.float64, ()),
    "leg_mode": ("leg", np.int32, ()),
    "leg_travel_time": ("leg", np.float64, ()),
    "leg_routed": ("leg", np.bool_, ()),
    "leg_pt_route": ("leg", np.bool_, ()),
    "leg_route_time": ("leg", np.float64, ()),
    "leg_route_distance": ("leg", np.float64, ()),
    "route_links": ("link", np.int32, ()),
    "ride_board_stop": ("ride", np.int32, ()),
    "ride_alight_stop": ("ride", np.int32, ()),
}
# offset column -> (unit it has n + 1 values per, unit its values index)
_OFFSETS = {
    "act_offsets": ("plan", "act"),
    "route_offsets": ("leg", "link"),
    "ride_offsets": ("leg", "ride"),
}
_UNITS = ("plan", "act", "leg", "link", "ride")
_ALL = (*_OFFSETS, *_COLUMNS)


def _allocate(capacity: dict) -> dict:
    arrays = {}
    for name, (unit, dtype, shape) in _COLUMNS.items():
        arrays[name] = np.empty((capacity[unit], *shape), dtype=dtype)
    for name, (unit, _) in _OFFSETS.items():
        arrays[name] = np.empty(capacity[unit] + 1, dtype=np.int64)
        arrays[name][0] = 0
    return arrays


def _capacity(arrays: dict) -> dict:
    cap = {}
    for name, (unit, _, _) in _COLUMNS.items():
        cap[unit] = len(arrays[name])
    cap["plan"] = len(arrays["act_offsets"]) - 1
    return cap


class _Lineage:
    """The buffers a chain of appends shares, and how far they are filled."""

    __slots__ = ("token", "arrays", "tip")

    def __init__(self, arrays: dict, tip: tuple):
        self.token = uuid.uuid4().int  # unique across processes
        self.arrays = arrays
        self.tip = tip  # counts of the newest snapshot


def csr_gather(offsets: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather CSR rows: ``(element indices, new offsets)``.

    ``offsets`` has one more entry than there are rows; ``rows`` may repeat
    and be in any order."""
    rows = np.asarray(rows, dtype=np.int64)
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    new = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=new[1:])
    idx = np.arange(new[-1], dtype=np.int64) + np.repeat(starts - new[:-1], counts)
    return idx, new


def first_seen_codes(codes: np.ndarray, vocab: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Recode ``codes`` to ``0..k-1`` in order of first appearance, keeping
    only the vocabulary entries they use (as ``PlanBatch.from_plans`` codes)."""
    if len(codes) == 0:
        return codes.astype(np.int32), ()
    used, first = np.unique(codes, return_index=True)
    order = used[np.argsort(first, kind="stable")]
    table = np.full(max(len(vocab), 1), -1, dtype=np.int32)
    table[order] = np.arange(len(order), dtype=np.int32)
    return table[codes], tuple(vocab[i] for i in order.tolist())


def _vocab_table(src: Sequence[str], vocab: list[str], index: dict[str, int]) -> np.ndarray:
    """Codes of ``src``'s entries in ``vocab`` (extended in place as needed)."""
    out = np.empty(len(src), dtype=np.int32)
    for i, name in enumerate(src):
        code = index.get(name)
        if code is None:
            code = index[name] = len(vocab)
            vocab.append(name)
        out[i] = code
    return out


class PlanStore:
    """An immutable, append-only table of plans (see the module docstring).

    Columns (read-only numpy views) have the names and meaning of the
    :class:`~mobslim.planbatch.PlanBatch` columns, plus ``leg_route_time``
    and ``leg_route_distance`` (``NaN``: no route). ``len(store)`` is the
    number of plans; ``store[p]`` and iteration give
    :class:`~mobslim.population.Plan` views.
    """

    __slots__ = ("_lin", "_arrays", "_n", "act_types", "modes", "_views")

    # -- construction -------------------------------------------------------
    @classmethod
    def _make(cls, lin: _Lineage, arrays: dict, n: tuple, act_types, modes) -> "PlanStore":
        store = object.__new__(cls)
        store._lin = lin
        store._arrays = arrays
        store._n = n
        store.act_types = tuple(act_types)
        store.modes = tuple(modes)
        store._views = {}
        return store

    @classmethod
    def empty(cls) -> "PlanStore":
        arrays = _allocate(dict.fromkeys(_UNITS, 0))
        n = (0,) * len(_UNITS)
        return cls._make(_Lineage(arrays, n), arrays, n, (), ())

    @classmethod
    def from_batch(cls, batch: PlanBatch, headroom: float = 0.0) -> "PlanStore":
        """A new lineage holding ``batch``'s plans (``headroom``: spare
        capacity for appends, as a fraction of the size)."""
        return cls.empty().append(batch, headroom=headroom)

    @classmethod
    def from_plans(cls, plans: Iterable) -> "PlanStore":
        return cls.from_batch(PlanBatch.from_plans(list(plans)))

    # -- sizes ----------------------------------------------------------------
    @property
    def token(self) -> int:
        """The lineage token: equal for snapshots sharing rows (see module docs)."""
        return self._lin.token

    @property
    def n_plans(self) -> int:
        return self._n[0]

    @property
    def n_acts(self) -> int:
        return self._n[1]

    @property
    def n_legs(self) -> int:
        return self._n[2]

    @property
    def n_links(self) -> int:
        return self._n[3]

    @property
    def n_rides(self) -> int:
        return self._n[4]

    def _count(self, unit: str) -> int:
        return self._n[_UNITS.index(unit)]

    def _view(self, name: str) -> np.ndarray:
        v = self._views.get(name)
        if v is None:
            if name in _OFFSETS:
                n = self._count(_OFFSETS[name][0]) + 1
            else:
                n = self._count(_COLUMNS[name][0])
            v = self._arrays[name][:n].view()
            v.flags.writeable = False
            self._views[name] = v
        return v

    @property
    def leg_offsets(self) -> np.ndarray:
        return self.act_offsets - np.arange(self.n_plans + 1)

    @property
    def nbytes(self) -> int:
        """Bytes used by the rows of this snapshot (not spare capacity)."""
        return sum(self._view(name).nbytes for name in _ALL)

    # -- appending --------------------------------------------------------------
    def append(self, batch: PlanBatch, headroom: float = 1.0) -> "PlanStore":
        """A snapshot with ``batch``'s plans appended as rows
        ``n_plans .. n_plans + batch.n_plans - 1``."""
        add = (batch.n_plans, batch.n_acts, batch.n_legs, len(batch.route_links), batch.n_rides)
        if add[0] == 0:
            return self
        n = self._n
        new_n = tuple(a + b for a, b in zip(n, add))
        lin = self._lin
        if lin.tip == n and self._arrays is lin.arrays:
            arrays = lin.arrays
            cap = _capacity(arrays)
            if any(new_n[i] > cap[u] for i, u in enumerate(_UNITS)):
                arrays = self._copy_into({u: max(new_n[i], int(cap[u] * (1 + headroom)))
                                          for i, u in enumerate(_UNITS)})
                lin.arrays = arrays  # same lineage: rows are unchanged
        else:  # not the tip: branch into a new lineage
            arrays = self._copy_into({u: int(new_n[i] * (1 + headroom)) or new_n[i]
                                      for i, u in enumerate(_UNITS)})
            lin = _Lineage(arrays, n)

        act_types, modes = list(self.act_types), list(self.modes)
        type_codes = _vocab_table(batch.act_types, act_types, {t: i for i, t in enumerate(act_types)})
        mode_codes = _vocab_table(batch.modes, modes, {m: i for i, m in enumerate(modes)})
        p, a, leg, k, r = n
        arrays["act_offsets"][p + 1:p + 1 + add[0]] = batch.act_offsets[1:] + a
        arrays["route_offsets"][leg + 1:leg + 1 + add[2]] = batch.route_offsets[1:] + k
        arrays["ride_offsets"][leg + 1:leg + 1 + add[2]] = batch.ride_offsets[1:] + r
        start = {"act": a, "leg": leg, "link": k, "ride": r}
        size = {"act": add[1], "leg": add[2], "link": add[3], "ride": add[4]}
        for name, (unit, _, _) in _COLUMNS.items():
            if name == "act_type":
                values = type_codes[batch.act_type] if size[unit] else batch.act_type
            elif name == "leg_mode":
                values = mode_codes[batch.leg_mode] if size[unit] else batch.leg_mode
            else:
                values = getattr(batch, name)
            arrays[name][start[unit]:start[unit] + size[unit]] = values
        lin.tip = new_n
        return PlanStore._make(lin, arrays, new_n, act_types, modes)

    def append_plans(self, plans: Sequence) -> "PlanStore":
        return self.append(PlanBatch.from_plans(plans))

    def _copy_into(self, capacity: dict) -> dict:
        arrays = _allocate(capacity)
        for name in _ALL:
            v = self._view(name)
            arrays[name][:len(v)] = v
        return arrays

    # -- reading ----------------------------------------------------------------
    def take(self, rows) -> PlanBatch:
        """Rows ``rows`` (any order, may repeat) as a :class:`PlanBatch`.

        Activity types and modes are recoded in order of first appearance,
        so the batch equals ``PlanBatch.from_plans([store[p] for p in rows])``.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) and (rows.min() < 0 or rows.max() >= self.n_plans):
            raise IndexError("PlanStore.take: row out of range.")
        if len(rows) == self.n_plans and (rows == np.arange(self.n_plans)).all():
            return self._all_rows()
        a_idx, act_offsets = csr_gather(self.act_offsets, rows)
        l_idx, _ = csr_gather(self.leg_offsets, rows)
        k_idx, route_offsets = csr_gather(self.route_offsets, l_idx)
        r_idx, ride_offsets = csr_gather(self.ride_offsets, l_idx)
        act_type, act_types = first_seen_codes(self.act_type[a_idx], self.act_types)
        leg_mode, modes = first_seen_codes(self.leg_mode[l_idx], self.modes)
        return PlanBatch(
            act_offsets=act_offsets,
            act_type=act_type,
            act_node=self.act_node[a_idx],
            act_end_time=self.act_end_time[a_idx],
            act_duration=self.act_duration[a_idx],
            leg_mode=leg_mode,
            leg_travel_time=self.leg_travel_time[l_idx],
            leg_routed=self.leg_routed[l_idx],
            route_offsets=route_offsets,
            route_links=self.route_links[k_idx],
            act_types=act_types,
            modes=modes,
            act_xy=self.act_xy[a_idx],
            leg_pt_route=self.leg_pt_route[l_idx],
            ride_offsets=ride_offsets,
            ride_board_stop=self.ride_board_stop[r_idx],
            ride_alight_stop=self.ride_alight_stop[r_idx],
            act_link=self.act_link[a_idx],
            leg_route_time=self.leg_route_time[l_idx],
            leg_route_distance=self.leg_route_distance[l_idx],
        )

    def _all_rows(self) -> PlanBatch:
        """Every row as a :class:`PlanBatch` of views (no gather)."""
        act_type, act_types = first_seen_codes(self.act_type, self.act_types)
        leg_mode, modes = first_seen_codes(self.leg_mode, self.modes)
        cols = {name: self._view(name) for name in _ALL if name not in ("act_type", "leg_mode")}
        return PlanBatch(act_type=act_type, leg_mode=leg_mode, act_types=act_types, modes=modes, **cols)

    def compact(self, rows) -> "PlanStore":
        """A new lineage holding only ``rows``, in that order (row ``rows[i]``
        becomes row ``i``)."""
        return PlanStore.from_batch(self.take(rows))

    def plan_elements(self, row: int) -> tuple:
        """Row ``row`` as ``Activity, Leg, ..., Activity`` objects."""
        from mobslim.population import (
            Activity,
            Leg,
            Route,
            TransitPassengerRoute,
        )

        if not 0 <= row < self.n_plans:
            raise IndexError("plan row out of range")
        arr = self._arrays
        a0, a1 = arr["act_offsets"][row:row + 2].tolist()
        l0, l1 = a0 - row, a1 - row - 1
        types, modes = self.act_types, self.modes
        acts = []
        for t, node, link, (x, y), end, dur in zip(
            arr["act_type"][a0:a1].tolist(), arr["act_node"][a0:a1].tolist(),
            arr["act_link"][a0:a1].tolist(), arr["act_xy"][a0:a1].tolist(),
            arr["act_end_time"][a0:a1].tolist(), arr["act_duration"][a0:a1].tolist(),
        ):
            acts.append(Activity(
                types[t], node,
                None if end != end else end,
                None if dur != dur else dur,
                None if x != x else x,
                None if x != x else y,
                None if link < 0 else link,
            ))
        if l1 == l0:
            return tuple(acts)
        out = [acts[0]]
        ro = arr["route_offsets"][l0:l1 + 1].tolist()
        rio = arr["ride_offsets"][l0:l1 + 1].tolist()
        links, board, alight = arr["route_links"], arr["ride_board_stop"], arr["ride_alight_stop"]
        for i, (m, tt, routed, pt, rtt, rdist) in enumerate(zip(
            arr["leg_mode"][l0:l1].tolist(), arr["leg_travel_time"][l0:l1].tolist(),
            arr["leg_routed"][l0:l1].tolist(), arr["leg_pt_route"][l0:l1].tolist(),
            arr["leg_route_time"][l0:l1].tolist(), arr["leg_route_distance"][l0:l1].tolist(),
        )):
            if not routed:
                route = None
            elif pt:
                pairs = np.empty((rio[i + 1] - rio[i], 2), dtype=_I32)
                pairs[:, 0] = board[rio[i]:rio[i + 1]]
                pairs[:, 1] = alight[rio[i]:rio[i + 1]]
                route = TransitPassengerRoute.packed(pairs.tobytes(), rtt, rdist)
            else:
                route = Route.packed(links[ro[i]:ro[i + 1]].astype(_I32, copy=False).tobytes(),
                                     rtt, rdist)
            out.append(Leg(modes[m], route, None if tt != tt else tt))
            out.append(acts[i + 1])
        return tuple(out)

    def __len__(self) -> int:
        return self.n_plans

    def __getitem__(self, row: int):
        from mobslim.population import Plan

        n = self.n_plans
        if not -n <= row < n:
            raise IndexError("plan row out of range")
        return Plan._view(self, row % n)

    def __iter__(self) -> Iterator:
        from mobslim.population import Plan

        view = Plan._view
        for row in range(self.n_plans):
            yield view(self, row)

    def to_plans(self, rows: Optional[Iterable[int]] = None) -> list:
        """Fully built plans (all rows by default)."""
        from mobslim.population import Plan

        rows = range(self.n_plans) if rows is None else rows
        return [Plan(self.plan_elements(int(r))) for r in rows]

    # -- pickling: the rows only, as a new lineage -------------------------------
    def __reduce__(self):
        return (_unpickle, ({name: np.array(self._view(name)) for name in _ALL},
                            self.act_types, self.modes, self._n))

    def __repr__(self) -> str:
        return (f"PlanStore(plans={self.n_plans}, acts={self.n_acts}, legs={self.n_legs}, "
                f"links={self.n_links}, rides={self.n_rides})")


def _store_column(name: str):
    return property(lambda self: self._view(name), doc=f"Read-only ``{name}`` column.")


for _name in _ALL:
    setattr(PlanStore, _name, _store_column(_name))
del _name


def _unpickle(arrays, act_types, modes, n) -> PlanStore:
    for name in _ALL:
        arrays[name] = np.ascontiguousarray(arrays[name])
    return PlanStore._make(_Lineage(arrays, tuple(n)), arrays, tuple(n), act_types, modes)
