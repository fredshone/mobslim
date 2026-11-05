"""Populations as Parquet: plans and person attributes in four column tables.

A population is a directory of four Parquet files::

    persons.parquet     person, <attribute columns ...>
    plans.parquet       person, plan, score, selected
    activities.parquet  person, plan, seq, type, node, link, x, y, end_time, duration
    legs.parquet        person, plan, seq, mode, travel_time, route_time,
                        route_distance, route, rides

* ``person`` is the person id, ``plan`` the plan's slot in the person's plan
  memory (0, 1, ...), ``seq`` the element's position among its plan's
  activities (or legs). Rows are written ordered by ``(person, plan, seq)``
  in person order; the reader sorts if they are not.
* Ids are external (the network's node and link ids, the person ids), so a
  file stays valid if the network is re-read in another order. String
  columns are dictionary-encoded, so repeated ids and types cost little.
* Missing values are nulls: ``score`` (unscored), ``link``, ``x``/``y``,
  ``end_time``/``duration`` (seconds from midnight), ``travel_time`` (fixed
  teleport time), ``route_*``.
* ``route`` is a network leg's route, a list of link ids: the links it
  drives, from the departure link's to-node up to and including the arrival
  link (MobSlim's convention, not MATSim's: the departure link is not in
  it). Null if the leg is unrouted. ``route_time``/``route_distance`` are the
  router's expectations.
* ``rides`` is a pt leg's route: a list of ``{line, route, board, alight}``
  (transit line and route ids; ``board``/``alight`` are stop positions within
  that transit route). Null for other legs; an empty list is a trip walked
  all the way. Reading rides needs the transit schedule; without it they
  are dropped (the legs are routed again).
* Person attributes are columns of ``persons.parquet`` (bool, int64,
  float64 or string, missing values null).

Needs ``pyarrow`` (``pip install mobslim[parquet]``).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Hashable, Optional

import numpy as np

from mobslim.attributes import PersonAttributes, column_array
from mobslim.ids import IdMap
from mobslim.network import Network
from mobslim.planbatch import PlanBatch
from mobslim.planstore import PlanStore, csr_gather
from mobslim.population import Population, validate_batch

log = logging.getLogger(__name__)

FORMAT = "mobslim.population"
VERSION = 1
TABLES = ("persons", "plans", "activities", "legs")


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("Parquet populations need pyarrow: pip install 'mobslim[parquet]'") from e
    return pa, pc, pq


def is_parquet_population(path) -> bool:
    """Is ``path`` a population directory (it holds ``persons.parquet``)?"""
    return (Path(path) / "persons.parquet").is_file()


# ---------------------------------------------------------------- writing

def _id_values(ids) -> Any:
    """External ids as a pyarrow array: int64 if they are all ints, else strings."""
    pa, _, _ = _pa()
    ids = list(ids)
    if ids and all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in ids):
        return pa.array(ids, type=pa.int64())
    return pa.array([str(i) for i in ids], type=pa.string())


def _dict(indices: np.ndarray, dictionary, missing: Optional[np.ndarray] = None):
    """A dictionary-encoded column: ``dictionary[indices]`` (null where ``missing``)."""
    pa, _, _ = _pa()
    idx = pa.array(np.asarray(indices, dtype=np.int32),
                   mask=None if missing is None or not missing.any() else missing)
    if not isinstance(dictionary, pa.Array):
        dictionary = pa.array(list(dictionary), type=pa.string())
    return pa.DictionaryArray.from_arrays(idx, dictionary)


def _floats(values: np.ndarray):
    pa, _, _ = _pa()
    values = np.asarray(values, dtype=np.float64)
    nan = np.isnan(values)
    return pa.array(values, mask=nan if nan.any() else None)


def _attribute_column(col: np.ndarray):
    pa, _, _ = _pa()
    if col.dtype != object:
        return _floats(col) if col.dtype.kind == "f" else pa.array(col)
    values = col.tolist()
    present = [v for v in values if v is not None]
    if all(isinstance(v, bool) for v in present):
        return pa.array(values, type=pa.bool_())
    if all(isinstance(v, str) for v in present):
        return pa.array(values, type=pa.string())
    return pa.array([None if v is None else str(v) for v in values], type=pa.string())


def _rs_parts(schedule):
    """Per global route stop: (route index, position in the route)."""
    return schedule.rs_route, schedule.rs_index_in_route


def write_population_parquet(population: Population, path, network: Network, *,
                             transit_schedule=None, compression: str = "zstd") -> Path:
    """Write ``population`` as a Parquet population directory (see the module
    docs). ``transit_schedule`` (a :class:`~mobslim.transit.TransitSchedule`
    or a :class:`~mobslim.transit.Transit`) is needed to write pt rides;
    without it pt legs are written unrouted."""
    pa, _, pq = _pa()
    schedule = getattr(transit_schedule, "schedule", transit_schedule)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta = {b"mobslim": json.dumps({"format": FORMAT, "version": VERSION}).encode()}

    def write(name: str, columns: dict):
        table = pa.table(columns).replace_schema_metadata(meta)
        pq.write_table(table, path / f"{name}.parquet", compression=compression)

    person_ids = _id_values(population.ids)

    # persons
    persons = {"person": person_ids}
    attrs = population.attributes
    for name in attrs.names:
        if name in ("person",):
            raise ValueError(f"Person attribute {name!r} clashes with the id column.")
        persons[name] = _attribute_column(attrs[name])
    write("persons", persons)

    # plans: one row per (person, slot), in person then slot order
    used = population.plan_ids >= 0
    person_of_plan, slot = np.nonzero(used)
    rows = population.plan_ids[used].astype(np.int64)
    scores = population.scores[used]
    write("plans", {
        "person": _dict(person_of_plan, person_ids),
        "plan": pa.array(slot.astype(np.int32)),
        "score": _floats(scores),
        "selected": pa.array(population.selected[person_of_plan] == slot),
    })

    batch = population.plans.take(rows)
    node_ids = _id_values(network.node_ids)
    link_ids = _id_values(network.link_ids)

    # activities
    plan_of_act = batch.plan_of_act()
    act_seq = np.arange(batch.n_acts) - batch.act_offsets[plan_of_act]
    xy = batch.act_xy
    write("activities", {
        "person": _dict(person_of_plan[plan_of_act], person_ids),
        "plan": pa.array(slot[plan_of_act].astype(np.int32)),
        "seq": pa.array(act_seq.astype(np.int32)),
        "type": _dict(batch.act_type, batch.act_types),
        "node": _dict(batch.act_node, node_ids),
        "link": _dict(np.maximum(batch.act_link, 0), link_ids, batch.act_link < 0),
        "x": _floats(xy[:, 0]),
        "y": _floats(xy[:, 1]),
        "end_time": _floats(batch.act_end_time),
        "duration": _floats(batch.act_duration),
    })

    # legs
    plan_of_leg = batch.plan_of_leg()
    leg_seq = np.arange(batch.n_legs) - batch.leg_offsets[plan_of_leg]
    pt = batch.leg_pt_route & batch.leg_routed
    drop_pt = schedule is None and bool(pt.any())
    if drop_pt:
        log.warning("write_population_parquet: no transit schedule; %d pt legs written unrouted.",
                    int(pt.sum()))
        pt = np.zeros_like(pt)
    has_route = batch.leg_routed & ~batch.leg_pt_route
    route_offsets = batch.route_offsets
    route_values = _dict(batch.route_links, link_ids)
    route = pa.ListArray.from_arrays(pa.array(route_offsets.astype(np.int32)), route_values,
                                     mask=pa.array(~has_route))
    ride_type = pa.struct([("line", pa.string()), ("route", pa.string()),
                           ("board", pa.int32()), ("alight", pa.int32())])
    if schedule is not None and batch.n_rides:
        rs_route, rs_pos = _rs_parts(schedule)
        board = batch.ride_board_stop.astype(np.int64)
        alight = batch.ride_alight_stop.astype(np.int64)
        if (rs_route[board] != rs_route[alight]).any():
            raise ValueError("A pt ride boards and alights on different transit routes.")
        route_ids = [schedule.route_ids.id(r) for r in range(schedule.n_routes)]
        lines = pa.array([str(r[0]) for r in route_ids], type=pa.string())
        routes = pa.array([str(r[1]) for r in route_ids], type=pa.string())
        ride_route = rs_route[board]
        rides_values = pa.StructArray.from_arrays(
            [_dict(ride_route, lines).cast(pa.string()), _dict(ride_route, routes).cast(pa.string()),
             pa.array(rs_pos[board].astype(np.int32)), pa.array(rs_pos[alight].astype(np.int32))],
            fields=list(ride_type))
    else:
        rides_values = pa.array([], type=ride_type)
    ride_offsets = batch.ride_offsets if not drop_pt else np.zeros(batch.n_legs + 1, np.int64)
    rides = pa.ListArray.from_arrays(pa.array(ride_offsets.astype(np.int32)), rides_values,
                                     mask=pa.array(~pt))
    unrouted = ~(has_route | pt)
    write("legs", {
        "person": _dict(person_of_plan[plan_of_leg], person_ids),
        "plan": pa.array(slot[plan_of_leg].astype(np.int32)),
        "seq": pa.array(leg_seq.astype(np.int32)),
        "mode": _dict(batch.leg_mode, batch.modes),
        "travel_time": _floats(batch.leg_travel_time),
        "route_time": _floats(np.where(unrouted, np.nan, batch.leg_route_time)),
        "route_distance": _floats(np.where(unrouted, np.nan, batch.leg_route_distance)),
        "route": route,
        "rides": rides,
    })
    return path


# ---------------------------------------------------------------- reading

def _index_values(arr, value_set, what: str) -> np.ndarray:
    """Positions of ``arr``'s values in ``value_set`` (``-1`` for nulls);
    ``KeyError`` for a value not in it. Vectorised (``pyarrow.compute.index_in``)."""
    pa, pc, _ = _pa()
    if arr.type != value_set.type:
        try:
            arr_c = arr.cast(value_set.type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            arr_c, value_set = arr.cast(pa.string()), value_set.cast(pa.string())
    else:
        arr_c = arr
    found = pc.index_in(arr_c, value_set=value_set)
    missing = pc.and_(found.is_null(), arr.is_valid())
    if pc.any(missing).as_py():
        first = arr.filter(missing)[0].as_py()
        raise KeyError(f"Unknown {what} id {first!r}.")
    return found.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)


def _index(column, value_set, what: str) -> np.ndarray:
    """``_index_values`` over a chunked column; a dictionary chunk is looked
    up once per distinct value."""
    pa, _, _ = _pa()
    out = []
    for chunk in column.chunks:
        if pa.types.is_dictionary(chunk.type):
            table = _index_values(chunk.dictionary, value_set, what)
            idx = chunk.indices.fill_null(0).to_numpy(zero_copy_only=False)
            codes = table[idx] if len(table) else np.full(len(chunk), -1, dtype=np.int64)
            if chunk.null_count:
                codes[chunk.is_null().to_numpy(zero_copy_only=False)] = -1
        else:
            codes = _index_values(chunk, value_set, what)
        out.append(codes)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _plain(column):
    """A chunked column as one plain (not dictionary-encoded) array."""
    pa, _, _ = _pa()
    if pa.types.is_dictionary(column.type):
        column = column.cast(column.type.value_type)
    return column.combine_chunks()


def _codes(column, lookup) -> np.ndarray:
    """A (chunked) column -> int64 codes via ``lookup`` on its distinct
    values (``-1`` for nulls)."""
    _, pc, _ = _pa()
    import pyarrow as pa

    out = []
    for chunk in column.chunks:
        if not pa.types.is_dictionary(chunk.type):
            chunk = pc.dictionary_encode(chunk)
        values = chunk.dictionary.to_pylist()
        table = np.array([lookup(v) for v in values], dtype=np.int64) if values else np.zeros(0, np.int64)
        idx = chunk.indices.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        codes = table[idx] if len(table) else np.full(len(chunk), -1, dtype=np.int64)
        if chunk.null_count:
            codes[chunk.is_null().to_numpy(zero_copy_only=False)] = -1
        out.append(codes)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _vocab_codes(column) -> tuple[np.ndarray, list[str]]:
    vocab: list[str] = []
    index: dict[str, int] = {}

    def code(v):
        if v is None:
            raise ValueError("Activity types and modes must not be null.")
        c = index.get(v)
        if c is None:
            c = index[v] = len(vocab)
            vocab.append(v)
        return c

    return _codes(column, code), vocab


def _float_array(column) -> np.ndarray:
    _, pc, _ = _pa()
    import pyarrow as pa

    return pc.fill_null(column.cast(pa.float64()), math.nan).to_numpy()


def _int_array(column) -> np.ndarray:
    if column.null_count:
        raise ValueError(f"Column {column!r} must not have nulls.")
    return column.to_numpy().astype(np.int64)


# string columns read dictionary-encoded: ids and types are looked up once per value
_DICTIONARY = ("person", "type", "node", "link", "mode", "route.list.element")


def _read(path: Path, name: str, columns=None):
    pa, _, pq = _pa()
    file = path / f"{name}.parquet"
    if not file.is_file():
        raise FileNotFoundError(f"{file} is missing (a Parquet population needs {', '.join(TABLES)}).")
    schema = pq.read_schema(file)

    def is_string(field_path: str) -> bool:
        head, *rest = field_path.split(".")
        if schema.get_field_index(head) < 0:
            return False
        t = schema.field(head).type
        if rest:
            if not pa.types.is_list(t):
                return False
            t = t.value_type
        return pa.types.is_string(t) or pa.types.is_large_string(t)

    table = pq.read_table(file, columns=columns,
                          read_dictionary=[c for c in _DICTIONARY if is_string(c)])
    meta = (table.schema.metadata or {}).get(b"mobslim")
    if meta is not None:
        info = json.loads(meta)
        if info.get("format") != FORMAT or info.get("version", 0) > VERSION:
            raise ValueError(f"{file}: unsupported format {info}.")
    return table


def _attributes_from_table(table, n: int) -> dict[str, np.ndarray]:
    import pyarrow as pa

    cols = {}
    for name in table.column_names:
        if name == "person":
            continue
        col = table.column(name)
        t = col.type
        if pa.types.is_dictionary(t):
            col = col.cast(t.value_type)
            t = t.value_type
        if col.null_count == 0 and (pa.types.is_integer(t) or pa.types.is_floating(t)
                                    or pa.types.is_boolean(t)):
            values = col.to_numpy()
            if pa.types.is_integer(t):
                values = values.astype(np.int64)
            cols[name] = values
        elif pa.types.is_floating(t):
            cols[name] = _float_array(col)
        else:
            cols[name] = column_array(col.to_pylist())
    return cols


def read_person_attributes_parquet(path) -> dict[Hashable, dict[str, Any]]:
    """``{person_id: {name: value}}`` from a population directory's (or a
    single) ``persons`` Parquet file, as :func:`read_person_attributes_xml`."""
    path = Path(path)
    file = path / "persons.parquet" if path.is_dir() else path
    _, _, pq = _pa()
    table = pq.read_table(file)
    ids = table.column("person").to_pylist()
    attrs = PersonAttributes(len(ids), _attributes_from_table(table, len(ids)))
    return {pid: attrs.row(i) for i, pid in enumerate(ids)}


def _order(*keys: np.ndarray) -> Optional[np.ndarray]:
    """The lexicographic order of ``keys`` (most significant first), or
    ``None`` if the rows are already in order."""
    if len(keys[0]) < 2:
        return None
    tied = np.ones(len(keys[0]) - 1, dtype=bool)
    for k in keys:
        d = np.diff(k)
        if (tied & (d < 0)).any():
            return np.lexsort(keys[::-1])
        tied &= d == 0
    return None


def read_population_parquet(path, network: Network, *, transit_schedule=None,
                            keep_routes: bool = True) -> Population:
    """Read a Parquet population directory (see the module docs).

    Activity location: ``node``, else the to-node of ``link``, else the node
    nearest ``x``/``y``. ``keep_routes=False`` drops every route.
    ``transit_schedule`` (a :class:`~mobslim.transit.TransitSchedule` or a
    :class:`~mobslim.transit.Transit`) is needed to keep pt rides.
    """
    schedule = getattr(transit_schedule, "schedule", transit_schedule)
    path = Path(path)

    persons = _read(path, "persons")
    person_ids = _plain(persons.column("person"))
    ids = IdMap(person_ids.to_pylist())
    n = len(ids)
    node_ids, link_ids = _id_values(network.node_ids), _id_values(network.link_ids)
    attributes = PersonAttributes(n, _attributes_from_table(persons, n))

    # plans -> store rows, ordered by (person, slot)
    plans = _read(path, "plans")
    p_person = _index(plans.column("person"), person_ids, "person")
    p_slot = _int_array(plans.column("plan"))
    order = _order(p_person, p_slot)
    if order is not None:
        p_person, p_slot = p_person[order], p_slot[order]
    p_score = _float_array(plans.column("score"))
    p_selected = plans.column("selected").fill_null(False).to_numpy(zero_copy_only=False)
    if order is not None:
        p_score, p_selected = p_score[order], p_selected[order]
    n_plans = np.bincount(p_person, minlength=n)
    if (n_plans < 1).any():
        raise ValueError(f"Person {ids.id(int(np.argmax(n_plans < 1)))!r} has no plans.")
    first = np.concatenate([[0], np.cumsum(n_plans)[:-1]])
    if (p_slot != np.arange(len(p_slot)) - first[p_person]).any():
        raise ValueError("plans: each person's plan slots must be 0, 1, ... without gaps or repeats.")
    width = int(n_plans.max()) if n else 1
    key_of_row = p_person * width + p_slot  # sorted
    n_rows = len(key_of_row)

    def rows_of(table) -> np.ndarray:
        key = _index(table.column("person"), person_ids, "person") * width + _int_array(table.column("plan"))
        row = np.searchsorted(key_of_row, key)
        bad = (row >= n_rows) | (key_of_row[np.minimum(row, n_rows - 1)] != key)
        if bad.any():
            raise ValueError("A plan element names a (person, plan) that plans.parquet does not have.")
        return row

    # activities
    acts = _read(path, "activities")
    a_row = rows_of(acts)
    a_seq = _int_array(acts.column("seq"))
    a_order = _order(a_row, a_seq)
    take_a = (lambda x: x) if a_order is None else (lambda x: x[a_order])
    a_row, a_seq = take_a(a_row), take_a(a_seq)
    a_count = np.bincount(a_row, minlength=n_rows)
    act_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(a_count, out=act_offsets[1:])
    if (a_count < 1).any():
        raise ValueError("Every plan needs at least one activity.")
    if (a_seq != np.arange(len(a_seq)) - act_offsets[a_row]).any():
        raise ValueError("activities: each plan's seq must be 0, 1, ... without gaps or repeats.")
    act_type, act_types = _vocab_codes(acts.column("type"))
    act_type = take_a(act_type)
    node = take_a(_index(acts.column("node"), node_ids, "node"))
    link = take_a(_index(acts.column("link"), link_ids, "link"))
    x = take_a(_float_array(acts.column("x")))
    y = take_a(_float_array(acts.column("y")))
    no_node = node < 0
    if no_node.any():
        on_link = no_node & (link >= 0)
        node[on_link] = network.to_node[link[on_link]]
        rest = np.flatnonzero(no_node & (link < 0))
        if np.isnan(x[rest]).any() or np.isnan(y[rest]).any():
            raise ValueError("An activity has no node, link or coordinates.")
        for i in rest.tolist():
            node[i] = network.nearest_node(float(x[i]), float(y[i]))
    if ((link >= 0) & (network.to_node[np.maximum(link, 0)] != node)).any():
        raise ValueError("An activity's link does not end at its node.")

    # legs
    legs = _read(path, "legs")
    l_row = rows_of(legs)
    l_seq = _int_array(legs.column("seq"))
    l_order = _order(l_row, l_seq)
    take_l = (lambda x: x) if l_order is None else (lambda x: x[l_order])
    l_row, l_seq = take_l(l_row), take_l(l_seq)
    if (np.bincount(l_row, minlength=n_rows) != a_count - 1).any():
        raise ValueError("Every plan needs one leg fewer than activities.")
    leg_offsets = act_offsets - np.arange(n_rows + 1)
    if (l_seq != np.arange(len(l_seq)) - leg_offsets[l_row]).any():
        raise ValueError("legs: each plan's seq must be 0, 1, ... without gaps or repeats.")
    n_legs = len(l_row)
    leg_mode, modes = _vocab_codes(legs.column("mode"))
    leg_mode = take_l(leg_mode)
    travel_time = take_l(_float_array(legs.column("travel_time")))
    route_time = take_l(_float_array(legs.column("route_time")))
    route_distance = take_l(_float_array(legs.column("route_distance")))

    route_col = legs.column("route").combine_chunks()
    has_route = ~route_col.is_null().to_numpy(zero_copy_only=False)
    offsets, route_values = _list_parts(route_col)
    links = _index(_chunked(route_values), link_ids, "link")
    if (links < 0).any():
        raise ValueError("A route has a null link.")
    rides_col = legs.column("rides").combine_chunks()
    has_rides = ~rides_col.is_null().to_numpy(zero_copy_only=False)
    if has_rides.any() and schedule is None:
        log.warning("read_population_parquet: no transit schedule; pt rides of %d legs dropped "
                    "(they are routed again).", int(has_rides.sum()))
        has_rides[:] = False
    ride_offsets, ride_values = _list_parts(rides_col)
    if has_rides.any():
        board, alight = _ride_stops(ride_values, schedule)
    else:
        board = alight = np.zeros(0, dtype=np.int64)
    if not keep_routes:
        has_route[:] = False
        has_rides[:] = False
    if (has_route & has_rides).any():
        raise ValueError("A leg has both a network route and pt rides.")
    # empty the lists of legs without a route, then put legs in order
    route_offsets, links = _keep_lists(offsets, has_route, links)
    rides_offsets, (board, alight) = _keep_lists(ride_offsets, has_rides, board, alight)
    if l_order is not None:
        link_idx, route_offsets = csr_gather(route_offsets, l_order)
        links = links[link_idx]
        ride_idx, rides_offsets = csr_gather(rides_offsets, l_order)
        board, alight = board[ride_idx], alight[ride_idx]
        has_route, has_rides = has_route[l_order], has_rides[l_order]
    routed = has_route | has_rides
    nan = np.full(n_legs, np.nan)

    batch = PlanBatch(
        act_offsets=act_offsets,
        act_type=act_type.astype(np.int32),
        act_node=node.astype(np.int32),
        act_end_time=take_a(_float_array(acts.column("end_time"))),
        act_duration=take_a(_float_array(acts.column("duration"))),
        leg_mode=leg_mode.astype(np.int32),
        leg_travel_time=travel_time,
        leg_routed=routed,
        route_offsets=route_offsets,
        route_links=links.astype(np.int32),
        act_types=tuple(act_types),
        modes=tuple(modes),
        act_xy=np.column_stack([x, y]).reshape(-1, 2),
        leg_pt_route=has_rides,
        ride_offsets=rides_offsets,
        ride_board_stop=board.astype(np.int32),
        ride_alight_stop=alight.astype(np.int32),
        act_link=link.astype(np.int32),
        leg_route_time=np.where(routed, route_time, nan),
        leg_route_distance=np.where(routed, route_distance, nan),
    )
    validate_batch(batch)
    plan_ids = np.full((n, width), -1, dtype=np.int32)
    plan_ids[p_person, p_slot] = np.arange(n_rows, dtype=np.int32)
    scores = np.full((n, width), np.nan)
    scores[p_person, p_slot] = p_score
    selected = np.zeros(n, dtype=np.int32)
    if p_selected.any():
        if np.bincount(p_person[p_selected], minlength=n).max() > 1:
            raise ValueError("A person has more than one selected plan.")
        selected[p_person[p_selected]] = p_slot[p_selected]
    return Population.from_tables(ids, PlanStore.from_batch(batch), plan_ids, scores, selected,
                                  attributes)


def _keep_lists(offsets: np.ndarray, keep: np.ndarray, *values: np.ndarray):
    """Offsets and values with the lists of rows not in ``keep`` emptied
    (no copy if they already are)."""
    lengths = np.diff(offsets)
    if not keep.any():
        out = tuple(v[:0] for v in values)
        return np.zeros(len(offsets), dtype=np.int64), out if len(out) != 1 else out[0]
    if not lengths[~keep].any():
        return offsets, values if len(values) != 1 else values[0]
    idx, _ = csr_gather(offsets, np.flatnonzero(keep))
    new = np.zeros(len(offsets), dtype=np.int64)
    np.cumsum(np.where(keep, lengths, 0), out=new[1:])
    out = tuple(v[idx] for v in values)
    return new, out if len(out) != 1 else out[0]


def _list_parts(col) -> tuple[np.ndarray, Any]:
    """``(offsets from 0, the values they index)`` of a list array (null
    entries keep whatever sub-lists back them; callers mask them)."""
    offsets = col.offsets.to_numpy().astype(np.int64)
    start = int(offsets[0]) if len(offsets) else 0
    values = col.values.slice(start, int(offsets[-1]) - start if len(offsets) else 0)
    return offsets - start, values


def _chunked(array):
    import pyarrow as pa

    return pa.chunked_array([array])


def _ride_stops(rides, schedule) -> tuple[np.ndarray, np.ndarray]:
    """Global (board, alight) route stops of ``{line, route, board, alight}`` rows."""
    routes = {}

    def route_index(key):
        r = routes.get(key)
        if r is None:
            line, route = key
            r = schedule.route_ids.get((line, route))
            if r is None:  # ids written as strings
                by_str = {(str(a), str(b)): i for i, (a, b) in enumerate(schedule.route_ids)}
                r = by_str.get((str(line), str(route)))
            if r is None:
                raise KeyError(f"Unknown transit route {line!r}/{route!r}.")
            routes[key] = r
        return r

    line = rides.field("line").to_pylist()
    route = rides.field("route").to_pylist()
    r = np.array([route_index(k) for k in zip(line, route)], dtype=np.int64)
    start = schedule.route_stop_offsets[r]
    size = schedule.route_stop_offsets[r + 1] - start
    board = rides.field("board").to_numpy(zero_copy_only=False).astype(np.int64)
    alight = rides.field("alight").to_numpy(zero_copy_only=False).astype(np.int64)
    if ((board < 0) | (alight <= board) | (alight >= size)).any():
        raise ValueError("A pt ride's board/alight positions are out of order or out of its route.")
    return start + board, start + alight
