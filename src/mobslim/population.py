"""Demand: activities, legs, plans, persons and the population.

All plan objects are immutable (frozen dataclasses). Replanning never edits a
plan in place; it builds a new one with :func:`dataclasses.replace`. This makes
plans safe to share between iterations, between the mobsim and replanning, and
across worker processes.

Plans and scores are kept apart. A :class:`Plan` is content only; scores and
the selected plan are per-person state that changes every iteration. A
:class:`Population` stores persons as tables: a plan store plus
``(n_persons, n_slots)`` arrays of plan ids and scores and a ``selected``
array. Scoring updates one array instead of rebuilding persons and plans.

An activity is a location: its coordinates (``Activity.x, y``) and the
network node a network leg starts and ends at. Before simulation every
activity is assigned a link (``Activity.link``, as MATSim's activities have;
its node is then the link's to-node) by
:func:`~mobslim.activity_links.assign_activity_links`, unless it already
has one. A car leg then ends by driving through that link, as in
MATSim, instead of reaching the node by any way in (see
:func:`~mobslim.routing.base.route_many_to_links`). An activity may also keep
the coordinates it had in the plans file (``Activity.x, y``): teleported legs
and the walks of pt trips start and end there, as in MATSim, while the network
sees the node.
A route's link indices are stored packed as int32 bytes (see :class:`Route`),
so routes stay compact and a mobsim's input is assembled by joining buffers,
not by iterating Python ints. A pt leg's route is a :class:`TransitPassengerRoute`: its
rides, packed the same way.

Activity timing follows MATSim's ``tryEndTimeThenDuration``: an activity with
an ``end_time`` ends at that clock time (or immediately, if the agent arrives
late); otherwise it lasts ``duration`` seconds. The last activity of a plan has
neither and lasts until the end of the simulation.
"""

import math
from dataclasses import FrozenInstanceError, dataclass
from typing import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from mobslim.attributes import (
    PersonAttributes,
)
from mobslim.ids import IdMap
from mobslim.network import Network
from mobslim.planbatch import PlanBatch
from mobslim.planstore import PlanStore


@dataclass(frozen=True, slots=True)
class Activity:
    """``x, y`` are optional coordinates (``None``: the node's). Teleported
    legs and pt walks start and end at them; the network uses ``node``.

    ``link`` is the optional link the activity is on; ``node`` must then be
    its to-node. A network leg to the activity is routed to the link's
    from-node and then through the link (MATSim's routes end on the
    activity's link), unless the leg starts on that same link (an empty
    route) or the link does not allow the leg's mode. ``None``: the leg
    reaches ``node`` by whichever link is quickest.
    """

    type: str
    node: int
    end_time: Optional[float] = None
    duration: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    link: Optional[int] = None

    @property
    def has_xy(self) -> bool:
        return self.x is not None

    def xy(self, network: Network) -> tuple[float, float]:
        """The activity's coordinates, or its node's if it has none."""
        if self.x is not None:
            return self.x, self.y
        x, y = network.node_xy[self.node].tolist()
        return x, y

    def planned_end(self, start: float) -> Optional[float]:
        """End time given the time the agent starts this activity."""
        if self.end_time is not None:
            return max(start, self.end_time)
        if self.duration is not None:
            return start + self.duration
        return None


_I32 = np.dtype("=i4")  # exactly 32-bit, native byte order (as the Rust router writes)


class Route:
    """A network route: link indices plus the router's expectations.

    ``Route(links, travel_time, distance)``. ``links`` may be any sequence of
    ints, an integer numpy array, or ``bytes`` of packed native int32 (what
    routers produce). The links are kept as those bytes (``data``, 4 bytes
    per link) rather than as a tuple of Python ints, about 9x smaller, and
    :meth:`PlanBatch.from_plans <mobslim.planbatch.PlanBatch.from_plans>`
    builds the mobsim's link array by joining them.

    * ``links``: the links as a tuple (decoded on each access).
    * ``link_array``: a read-only int32 numpy view, no copy.
    * ``n_links``: the number of links.

    Immutable, hashable and compared by value, like the frozen dataclasses
    around it.
    """

    __slots__ = ("data", "travel_time", "distance")

    def __init__(self, links, travel_time: float, distance: float):
        if isinstance(links, bytes):
            if len(links) % 4:
                raise ValueError("Route: packed links must be a multiple of 4 bytes (int32).")
            data = links
        else:
            a = links if isinstance(links, np.ndarray) else np.array(links, dtype=np.int64)
            if a.ndim != 1:
                raise ValueError("Route: links must be one-dimensional.")
            if a.size and not np.issubdtype(a.dtype, np.integer):
                raise TypeError("Route: links must be integers.")
            if a.size and (a.min() < 0 or a.max() > np.iinfo(np.int32).max):
                raise ValueError("Route: link indices must fit in int32 and be >= 0.")
            data = a.astype(_I32, copy=False).tobytes()
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "travel_time", travel_time)
        object.__setattr__(self, "distance", distance)

    @classmethod
    def packed(cls, data: bytes, travel_time: float, distance: float) -> "Route":
        """Fast constructor from packed int32 ``bytes`` (no checks)."""
        route = object.__new__(cls)
        object.__setattr__(route, "data", data)
        object.__setattr__(route, "travel_time", travel_time)
        object.__setattr__(route, "distance", distance)
        return route

    @property
    def links(self) -> tuple[int, ...]:
        return tuple(np.frombuffer(self.data, dtype=_I32).tolist())

    @property
    def link_array(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype=_I32)

    @property
    def n_links(self) -> int:
        return len(self.data) >> 2

    def __setattr__(self, name, value):
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other):
        if other.__class__ is not Route:
            return NotImplemented
        return (self.data == other.data and self.travel_time == other.travel_time
                and self.distance == other.distance)

    def __hash__(self):
        return hash((self.data, self.travel_time, self.distance))

    def __reduce__(self):
        return (Route.packed, (self.data, self.travel_time, self.distance))

    def __repr__(self):
        return f"Route(links={self.links!r}, travel_time={self.travel_time!r}, distance={self.distance!r})"


class TransitPassengerRoute:
    """A transit route: the rides of one pt trip, plus the router's expectations.

    ``TransitPassengerRoute(rides, travel_time, distance)``. Each ride is a pair
    ``(board_route_stop, alight_route_stop)`` of global route-stop indices
    into the scenario's :class:`~mobslim.transit.TransitSchedule`. A route
    stop names a transit route, so a ride says which line and route to take,
    not which departure: the passenger boards the first vehicle of that
    route with room. The walks (to the first stop, between rides, from the
    last stop) are implied by the stops. Zero rides means the trip is walked.

    Stored like :class:`Route`: ``data`` holds the pairs as packed native
    int32 (8 bytes per ride), and :class:`~mobslim.planbatch.PlanBatch`
    joins those buffers.

    * ``rides``: the rides as a tuple of pairs (decoded on each access).
    * ``ride_array``: a read-only ``(n_rides, 2)`` int32 view, no copy.
    * ``n_rides``: the number of rides.

    ``travel_time`` is the router's expected door-to-door time, used to
    estimate later departure times while routing a plan.
    """

    __slots__ = ("data", "travel_time", "distance")

    def __init__(self, rides, travel_time: float, distance: float):
        if isinstance(rides, bytes):
            if len(rides) % 8:
                raise ValueError("TransitPassengerRoute: packed rides must be a multiple of 8 bytes (int32 pairs).")
            data = rides
        else:
            a = rides if isinstance(rides, np.ndarray) else np.array(list(rides), dtype=np.int64)
            if a.size == 0:
                a = a.reshape(0, 2)
            if a.ndim != 2 or a.shape[1] != 2:
                raise ValueError("TransitPassengerRoute: rides must be (board, alight) pairs.")
            if a.size and not np.issubdtype(a.dtype, np.integer):
                raise TypeError("TransitPassengerRoute: route stops must be integers.")
            if a.size and (a.min() < 0 or a.max() > np.iinfo(np.int32).max):
                raise ValueError("TransitPassengerRoute: route-stop indices must fit in int32 and be >= 0.")
            data = a.astype(_I32, copy=False).tobytes()
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "travel_time", travel_time)
        object.__setattr__(self, "distance", distance)

    @classmethod
    def packed(cls, data: bytes, travel_time: float, distance: float) -> "TransitPassengerRoute":
        """Fast constructor from packed int32 ``bytes`` (no checks)."""
        route = object.__new__(cls)
        object.__setattr__(route, "data", data)
        object.__setattr__(route, "travel_time", travel_time)
        object.__setattr__(route, "distance", distance)
        return route

    @property
    def ride_array(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype=_I32).reshape(-1, 2)

    @property
    def rides(self) -> tuple[tuple[int, int], ...]:
        return tuple(tuple(r) for r in self.ride_array.tolist())

    @property
    def n_rides(self) -> int:
        return len(self.data) >> 3

    def __setattr__(self, name, value):
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other):
        if other.__class__ is not TransitPassengerRoute:
            return NotImplemented
        return (self.data == other.data and self.travel_time == other.travel_time
                and self.distance == other.distance)

    def __hash__(self):
        return hash((TransitPassengerRoute, self.data, self.travel_time, self.distance))

    def __reduce__(self):
        return (TransitPassengerRoute.packed, (self.data, self.travel_time, self.distance))

    def __repr__(self):
        return (f"TransitPassengerRoute(rides={self.rides!r}, travel_time={self.travel_time!r}, "
                f"distance={self.distance!r})")


@dataclass(frozen=True, slots=True)
class Leg:
    mode: str
    route: Optional[Union[Route, TransitPassengerRoute]] = None
    travel_time: Optional[float] = None  # fixed time for teleported legs


PlanElement = Union[Activity, Leg]


class Plan:
    """Alternating ``Activity, Leg, Activity, ..., Activity``.

    A plan is pure content: what a person intends to do. Its score is *not*
    part of it. Scores change every iteration while plans do not, so scores
    live on the :class:`Person` (``person.scores``) and, inside a
    :class:`Population`, in a score array.

    Immutable. A plan read from a population is a *view* of a row of its
    :class:`~mobslim.planstore.PlanStore`: its elements are built from the
    store's columns on first access. A view remembers its row, so a
    population updated with persons whose plans did not change stores no
    new rows for them. Plans compare by content.
    """

    __slots__ = ("_elements", "_store", "_token", "_row")

    def __init__(self, elements: Iterable[PlanElement]):
        elements = tuple(elements)
        _validate_elements(elements)
        _set(self, "_elements", elements)
        _set(self, "_store", None)
        _set(self, "_token", None)
        _set(self, "_row", -1)

    @classmethod
    def _view(cls, store, row: int) -> "Plan":
        plan = _new(cls)
        _set(plan, "_elements", None)
        _set(plan, "_store", store)
        _set(plan, "_token", store.token)
        _set(plan, "_row", row)
        return plan

    def _stored_as(self, store, row: int) -> None:
        """Record that this plan is ``store``'s row ``row`` (it is unchanged)."""
        _set(self, "_store", None)  # elements are kept, no need to hold the store
        _set(self, "_token", store.token)
        _set(self, "_row", row)

    @property
    def elements(self) -> tuple[PlanElement, ...]:
        elements = self._elements
        if elements is None:
            elements = self._store.plan_elements(self._row)
            _set(self, "_elements", elements)
            _set(self, "_store", None)
        return elements

    def __eq__(self, other):
        if other.__class__ is not Plan:
            return NotImplemented
        if self is other or (self._token is not None and self._token == other._token
                             and self._row == other._row):
            return True
        return self.elements == other.elements

    def __hash__(self):
        return hash(self.elements)

    def __setattr__(self, name, value):
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __reduce__(self):
        return (_restore_plan, (self.elements, self._token, self._row))

    def __repr__(self) -> str:
        return f"Plan(elements={self.elements!r})"

    @property
    def activities(self) -> tuple[Activity, ...]:
        return self.elements[::2]

    @property
    def legs(self) -> tuple[Leg, ...]:
        return self.elements[1::2]

    def trips(self) -> Iterator[tuple[int, Leg, int, int]]:
        """Yield ``(element_index, leg, origin_node, destination_node)``."""
        elements = self.elements
        for i in range(1, len(elements), 2):
            yield i, elements[i], elements[i - 1].node, elements[i + 1].node

    def with_elements(self, elements: Iterable[PlanElement]) -> "Plan":
        """A new, validated plan with these elements."""
        return Plan(tuple(elements))

    @staticmethod
    def builder() -> "PlanBuilder":
        return PlanBuilder()


def _restore_plan(elements, token, row) -> Plan:
    plan = _new(Plan)
    _set(plan, "_elements", elements)
    _set(plan, "_store", None)
    _set(plan, "_token", token)
    _set(plan, "_row", row)
    return plan


def _validate_elements(elements: tuple) -> None:
    if not elements:
        raise ValueError("A plan needs at least one activity.")
    for i, e in enumerate(elements):
        expected = Activity if i % 2 == 0 else Leg
        if not isinstance(e, expected):
            raise ValueError(
                f"Plan element {i} must be {expected.__name__}, got {type(e).__name__}."
            )
    if not isinstance(elements[-1], Activity):
        raise ValueError("A plan must end with an activity.")
    for i, act in enumerate(elements[:-1:2]):
        if act.end_time is None and act.duration is None:
            raise ValueError(
                f"Activity {2 * i} ({act.type}) needs an end_time or duration."
            )


def validate_batch(batch) -> None:
    """Raise unless every plan of a :class:`~mobslim.planbatch.PlanBatch` is
    valid: at least one activity, and every activity but the last of its plan
    has an end time or a duration (as :class:`Plan` checks)."""
    counts = np.diff(batch.act_offsets)
    if (counts < 1).any():
        raise ValueError(f"Plan {int(np.argmax(counts < 1))} has no activity.")
    last = np.zeros(batch.n_acts, dtype=bool)
    last[batch.act_offsets[1:] - 1] = True
    bad = ~last & np.isnan(batch.act_end_time) & np.isnan(batch.act_duration)
    if bad.any():
        a = int(np.argmax(bad))
        p = int(batch.plan_of_act()[a])
        raise ValueError(f"Activity {2 * (a - int(batch.act_offsets[p]))} "
                         f"({batch.act_types[batch.act_type[a]]}) of plan {p} needs an "
                         f"end_time or duration.")


class PlanBuilder:
    """Fluent helper: ``Plan.builder().act(...).leg("car").act(...).build()``."""

    def __init__(self):
        self._elements: list[PlanElement] = []

    def act(self, type: str, node: int, *, end_time=None, duration=None, xy=None,
            link=None) -> "PlanBuilder":
        x, y = (None, None) if xy is None else (float(xy[0]), float(xy[1]))
        self._elements.append(Activity(type, node, end_time, duration, x, y, link))
        return self

    def leg(self, mode: str, route: Optional[Union[Route, TransitPassengerRoute]] = None,
            travel_time=None) -> "PlanBuilder":
        self._elements.append(Leg(mode, route, travel_time))
        return self

    def build(self) -> Plan:
        return Plan(tuple(self._elements))


Score = Optional[float]  # None: the plan has not been executed yet


@dataclass(frozen=True, slots=True)
class Person:
    """A person's plan memory: plans, their scores, and which one is selected.

    ``scores[i]`` is the score of ``plans[i]`` (``None`` = not tried yet) and
    defaults to all ``None``. A ``Person`` is a value: strategies return
    updated copies. Inside a :class:`Population` persons are stored as arrays
    and a ``Person`` is built on access.
    """

    id: Hashable
    plans: tuple[Plan, ...]
    selected: int = 0
    scores: tuple[Score, ...] = None  # type: ignore[assignment]  # filled in __post_init__

    def __post_init__(self):
        plans = tuple(self.plans)
        object.__setattr__(self, "plans", plans)
        if not plans:
            raise ValueError(f"Person {self.id!r} has no plans.")
        if self.scores is None:
            object.__setattr__(self, "scores", (None,) * len(plans))
        else:
            object.__setattr__(self, "scores", tuple(self.scores))
            if len(self.scores) != len(plans):
                raise ValueError(f"Person {self.id!r}: need one score per plan.")
        if not 0 <= self.selected < len(plans):
            raise ValueError(f"Person {self.id!r}: selected index out of range.")

    @property
    def selected_plan(self) -> Plan:
        return self.plans[self.selected]

    @property
    def selected_score(self) -> Score:
        return self.scores[self.selected]

    def select(self, index: int) -> "Person":
        if index == self.selected:
            return self
        if not 0 <= index < len(self.plans):
            raise ValueError(f"Person {self.id!r}: selected index out of range.")
        return self._unchecked(self.plans, index, self.scores)

    def with_selected_score(self, score: Score) -> "Person":
        i = self.selected
        return self._unchecked(self.plans, i, self.scores[:i] + (score,) + self.scores[i + 1 :])

    def with_plans(self, plans: Sequence[Plan]) -> "Person":
        """The same person with each plan replaced (same number; scores and
        selection are kept)."""
        plans = tuple(plans)
        if len(plans) != len(self.plans):
            raise ValueError(f"Person {self.id!r}: with_plans needs {len(self.plans)} plans.")
        return self._unchecked(plans, self.selected, self.scores)

    def add_plan(self, plan: Plan, select: bool = True, score: Score = None) -> "Person":
        plans = self.plans + (plan,)
        return self._unchecked(plans, len(plans) - 1 if select else self.selected,
                               self.scores + (score,))

    def remove_plan(self, index: int) -> "Person":
        if len(self.plans) == 1:
            raise ValueError("Cannot remove a person's only plan.")
        if not 0 <= index < len(self.plans):
            raise ValueError(f"Person {self.id!r}: plan index out of range.")
        if index == self.selected:
            selected = 0
        else:
            selected = self.selected - (index < self.selected)
        return self._unchecked(
            self.plans[:index] + self.plans[index + 1 :],
            selected,
            self.scores[:index] + self.scores[index + 1 :],
        )

    def _unchecked(self, plans: tuple, selected: int, scores: tuple) -> "Person":
        new = object.__new__(Person)
        object.__setattr__(new, "id", self.id)
        object.__setattr__(new, "plans", plans)
        object.__setattr__(new, "selected", selected)
        object.__setattr__(new, "scores", scores)
        return new


class Population:
    """All persons, stored as tables. Plans are referenced, not nested.

    ========  ===============================  ==================================
    plans     :class:`PlanStore`               every stored plan, as columns
    plan_ids  ``int32[n_persons, n_slots]``    row in ``plans``; ``-1`` = empty
    scores    ``float64[n_persons, n_slots]``  ``NaN`` = unscored (or empty slot)
    selected  ``int32[n_persons]``             selected slot
    n_plans   ``int32[n_persons]``             used slots per person
    ========  ===============================  ==================================

    Row ``i`` is the person with *person index* ``i`` (``ids.id(i)``), used by
    the mobsim and in events. Scoring and selection-wide statistics work on
    the arrays directly; :meth:`selected_batch` gathers the selected plans
    for a mobsim without building plan objects; ``population[i]`` and
    iteration build :class:`Person` values (whose plans are lazy views of the
    store) for replanning.

    Person attributes (``subpopulation``, ``age``, ...) are a
    :class:`~mobslim.attributes.PersonAttributes` column table in
    ``attributes``, row ``i`` again person index ``i``. They are not part of
    :class:`Person` values: replanning never changes them, and every derived
    population keeps them.

    Immutable: the arrays are read-only and every update returns a new
    population that shares whatever did not change (``ids``, the plan store,
    unchanged arrays, the attributes). :meth:`replace` appends only plans
    that are not already rows of the store, and compacts the store (drops the
    rows no person holds) once it has more than twice as many rows as the
    persons hold plans, so its size stays bounded by about
    ``2 * n_persons * n_slots`` however long a run is.
    """

    __slots__ = ("ids", "plans", "plan_ids", "scores", "selected", "n_plans", "attributes")

    def __init__(self, persons: Iterable[Person], attributes=None):
        """``attributes``: ``None``, a :class:`~mobslim.attributes.PersonAttributes`,
        ``{person_id: {name: value}}`` records, or a pandas ``DataFrame``
        indexed by person id."""
        persons = tuple(persons)
        ids = IdMap(p.id for p in persons)
        self._set(ids, *_tables(persons, PlanStore.empty()), PersonAttributes.coerce(attributes, ids))

    def _set(self, ids, plans, plan_ids, scores, selected, n_plans, attributes) -> None:
        for a in (plan_ids, scores, selected, n_plans):
            a.flags.writeable = False
        self.ids = ids
        self.plans = plans
        self.plan_ids = plan_ids
        self.scores = scores
        self.selected = selected
        self.n_plans = n_plans
        self.attributes = attributes

    def _derive(self, **changes) -> "Population":
        new = object.__new__(Population)
        state = {k: getattr(self, k) for k in Population.__slots__}
        state.update(changes)
        new._set(**state)
        return new

    @classmethod
    def from_plans(cls, plans: Mapping[Hashable, Plan], attributes=None) -> "Population":
        """One single-plan person per ``{person_id: plan}`` entry."""
        return cls((Person(pid, (plan,)) for pid, plan in plans.items()), attributes)

    @classmethod
    def from_tables(cls, ids, plans: PlanStore, plan_ids, scores=None, selected=None,
                    attributes=None) -> "Population":
        """A population straight from tables (what readers build).

        ``ids``: person ids (an :class:`IdMap` or an iterable); ``plan_ids``:
        ``(n_persons, n_slots)`` rows of ``plans`` (``-1`` = empty, used slots
        first); ``scores`` (``NaN`` = unscored) and ``selected`` default to
        unscored and slot 0."""
        ids = ids if isinstance(ids, IdMap) else IdMap(ids)
        plan_ids = np.array(plan_ids, dtype=np.int32, ndmin=2).reshape(len(ids), -1)
        n_plans = (plan_ids >= 0).sum(axis=1).astype(np.int32)
        width = plan_ids.shape[1]
        used = np.arange(width)[None, :] < n_plans[:, None]
        if (used != (plan_ids >= 0)).any():
            raise ValueError("plan_ids: each person's plans must fill the first slots.")
        if (n_plans < 1).any():
            raise ValueError(f"Person {ids.id(int(np.argmax(n_plans < 1)))!r} has no plans.")
        if (plan_ids >= len(plans)).any():
            raise ValueError("plan_ids: row out of range of the plan store.")
        scores = (np.full(plan_ids.shape, np.nan) if scores is None
                  else np.array(scores, dtype=float).reshape(plan_ids.shape))
        scores[~used] = np.nan
        selected = (np.zeros(len(ids), dtype=np.int32) if selected is None
                    else np.array(selected, dtype=np.int32).reshape(len(ids)))
        if ((selected < 0) | (selected >= n_plans)).any():
            raise ValueError("selected: slot out of range.")
        new = object.__new__(cls)
        new._set(ids, plans, plan_ids, scores, selected, n_plans,
                 PersonAttributes.coerce(attributes, ids))
        return new

    def with_attributes(self, attributes) -> "Population":
        """The same population with these person attributes (anything
        ``Population(..., attributes=)`` accepts)."""
        return self._derive(attributes=PersonAttributes.coerce(attributes, self.ids))

    # -- updates --------------------------------------------------------------
    def with_tables(self, plans: PlanStore, plan_ids, scores, selected, n_plans) -> "Population":
        """The same persons with new plan tables (what vectorised replanning
        builds): ``plans`` must extend this population's store (or replace
        it), ``plan_ids`` index it. Trailing empty slot columns are dropped
        and the store is compacted if more than half its rows are unused."""
        plan_ids = np.asarray(plan_ids, dtype=np.int32)
        n_plans = np.asarray(n_plans, dtype=np.int32)
        width = max(int(n_plans.max(initial=1)), 1)
        plan_ids = np.ascontiguousarray(plan_ids[:, :width])
        scores = np.ascontiguousarray(np.asarray(scores, dtype=float)[:, :width])
        if plan_ids.shape != (len(self), width) or len(n_plans) != len(self):
            raise ValueError("with_tables: one row per person.")
        plans, plan_ids = _maybe_compact(plans, plan_ids, n_plans)
        return self._derive(plans=plans, plan_ids=plan_ids, scores=scores,
                            selected=np.asarray(selected, dtype=np.int32).copy(), n_plans=n_plans.copy())

    def with_plan_store(self, plans: PlanStore) -> "Population":
        """The same tables over ``plans``, a store that extends this one
        (``self.plans.append(...)``: the same rows, then new ones). Never
        compacts, so rows appended but not yet held by anyone stay valid."""
        if len(plans) < len(self.plans):
            raise ValueError("with_plan_store: the store must extend this population's store.")
        return self._derive(plans=plans)

    def replace(self, persons: Iterable[Person]) -> "Population":
        """Rebuild from updated persons (same persons, same order).

        Plans that are rows of this population's store (views read from it,
        unchanged) keep their rows; only new plans are appended. The store
        is then compacted if more than half its rows are unused."""
        persons = tuple(persons)
        if len(persons) != len(self) or any(
            a.id != b for a, b in zip(persons, self.ids)
        ):
            raise ValueError("Population.replace must keep the same persons in the same order.")
        plans, plan_ids, scores, selected, n_plans = _tables(persons, self.plans)
        plans, plan_ids = _maybe_compact(plans, plan_ids, n_plans)
        return self._derive(plans=plans, plan_ids=plan_ids, scores=scores, selected=selected,
                            n_plans=n_plans)

    def with_selected_scores(self, scores) -> "Population":
        """Set the score of every person's selected plan (indexed by person)."""
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(self),):
            raise ValueError(f"Expected {len(self)} scores, got shape {scores.shape}.")
        table = self.scores.copy()
        table[np.arange(len(self)), self.selected] = scores
        return self._derive(scores=table)

    def with_scores(self, persons, slots, values) -> "Population":
        """Set the score of slot ``slots[i]`` of person ``persons[i]`` to
        ``values[i]`` (a slot must hold a plan)."""
        persons = np.asarray(persons, dtype=np.int64)
        slots = np.asarray(slots, dtype=np.int64)
        values = np.asarray(values, dtype=float)
        if not persons.shape == slots.shape == values.shape:
            raise ValueError("with_scores: persons, slots and values must have the same length.")
        if len(slots) and ((slots < 0).any() or (slots >= self.n_plans[persons]).any()):
            raise ValueError("with_scores: slot out of range.")
        table = self.scores.copy()
        table[persons, slots] = values
        return self._derive(scores=table)

    def add_plans(self, persons, plan_rows, scores=None, select: bool = False) -> "Population":
        """Give person ``persons[i]`` the stored plan ``plan_rows[i]`` (a row
        of this population's store) as a new last slot, with score
        ``scores[i]`` (``NaN``/omitted: unscored). A person may appear more
        than once. With ``select``, each person's last added plan becomes
        selected. ``max_plans`` is not enforced here."""
        persons = np.asarray(persons, dtype=np.int64)
        plan_rows = np.asarray(plan_rows, dtype=np.int64)
        scores = (np.full(len(persons), np.nan) if scores is None
                  else np.asarray(scores, dtype=float))
        if not persons.shape == plan_rows.shape == scores.shape:
            raise ValueError("add_plans: persons, plan_rows and scores must have the same length.")
        if len(plan_rows) and (plan_rows.min() < 0 or plan_rows.max() >= len(self.plans)):
            raise ValueError("add_plans: row out of range of the plan store.")
        n = self.n_plans.astype(np.int64)
        # each added plan's slot: after the person's plans and earlier additions
        order = np.argsort(persons, kind="stable")
        sp = persons[order]
        first = np.r_[True, sp[1:] != sp[:-1]] if len(sp) else np.zeros(0, bool)
        start = np.flatnonzero(first)
        rank = np.arange(len(sp)) - np.repeat(start, np.diff(np.r_[start, len(sp)]))
        slot = np.empty(len(persons), dtype=np.int64)
        slot[order] = n[sp] + rank
        new_n = n + np.bincount(persons, minlength=len(self))
        width = max(int(new_n.max(initial=1)), self.plan_ids.shape[1])
        plan_ids = np.full((len(self), width), -1, dtype=np.int32)
        plan_ids[:, :self.plan_ids.shape[1]] = self.plan_ids
        table = np.full((len(self), width), np.nan)
        table[:, :self.scores.shape[1]] = self.scores
        plan_ids[persons, slot] = plan_rows
        table[persons, slot] = scores
        selected = self.selected.copy()
        if select and len(persons):
            last = np.full(len(self), -1, dtype=np.int64)
            np.maximum.at(last, persons, slot)
            selected = np.where(last >= 0, last, selected).astype(np.int32)
        return self._derive(plan_ids=plan_ids, scores=table, selected=selected,
                            n_plans=new_n.astype(np.int32))

    def map_plans(self, fn: Callable[[Plan], Plan]) -> "Population":
        """Apply ``fn`` to every stored plan once. Scores and selection are kept."""
        return self.map_plan_store(lambda plans: [fn(plan) for plan in plans])

    def map_plan_store(self, fn: Callable[[Sequence[Plan]], Sequence[Plan]]) -> "Population":
        """Like :meth:`map_plans`, but ``fn`` maps the whole plan store at once
        (for batch work such as routing) and returns one plan per input.

        Only the plans the persons hold are passed (the store is compacted
        first), as views."""
        live = self.live_plan_rows()
        plan_ids = self.plan_ids
        if len(live) != len(self.plans):
            plan_ids = _remap(plan_ids, live, len(self.plans))
        views = [self.plans[j] for j in live.tolist()]
        plans = list(fn(views))
        if len(plans) != len(views):
            raise ValueError("map_plan_store: fn must return one plan per stored plan.")
        if all(a is b for a, b in zip(plans, views)):
            return self if len(live) == len(self.plans) else self._derive(
                plans=self.plans.compact(live), plan_ids=plan_ids)
        store = PlanStore.from_plans(plans)
        for j, plan in enumerate(plans):
            plan._stored_as(store, j)
        return self._derive(plans=store, plan_ids=plan_ids)

    def with_live_batch(self, live: np.ndarray, batch: PlanBatch) -> "Population":
        """This population with its held plans replaced by ``batch``, row
        ``i`` of which replaces store row ``live[i]`` (``live`` as
        :meth:`live_plan_rows` returns it). For vectorised edits of every
        held plan at once: ``pop.with_live_batch(live, edit(pop.plans.take(live)))``."""
        live = np.asarray(live, dtype=np.int64)
        if batch.n_plans != len(live):
            raise ValueError("with_live_batch: batch must have one plan per live row.")
        plan_ids = self.plan_ids
        if len(live) != len(self.plans) or (len(live) and live[-1] != len(live) - 1):
            plan_ids = _remap(plan_ids, live, len(self.plans))
        return self._derive(plans=PlanStore.from_batch(batch), plan_ids=plan_ids)

    # -- reading --------------------------------------------------------------
    def live_plan_rows(self) -> np.ndarray:
        """Sorted rows of ``plans`` that some person holds."""
        ids = self.plan_ids
        return np.unique(ids[ids >= 0]).astype(np.int64)

    def live_batch(self) -> PlanBatch:
        """Every plan a person holds (each once), as a :class:`PlanBatch`."""
        return self.plans.take(self.live_plan_rows())

    def selected_plan_ids(self) -> np.ndarray:
        return self.plan_ids[np.arange(len(self)), self.selected]

    def selected_plans(self) -> list[Plan]:
        store = self.plans
        return [store[j] for j in self.selected_plan_ids().tolist()]

    def selected_batch(self) -> PlanBatch:
        """Each person's selected plan, in person order, as a
        :class:`PlanBatch` (a vectorised gather; equal to
        ``PlanBatch.from_plans(self.selected_plans())``)."""
        return self.plans.take(self.selected_plan_ids())

    def selected_scores(self) -> np.ndarray:
        """Score of each person's selected plan (``NaN`` = unscored)."""
        return self.scores[np.arange(len(self)), self.selected]

    def by_id(self, person_id: Hashable) -> Person:
        return self[self.ids.index(person_id)]

    def __getitem__(self, index: int) -> Person:
        if not -len(self) <= index < len(self):
            raise IndexError("person index out of range")
        index %= len(self)
        return _person(
            self.plans, self.ids.id(index), self.plan_ids[index].tolist(),
            self.scores[index].tolist(), int(self.selected[index]), int(self.n_plans[index]),
        )

    def __iter__(self) -> Iterator[Person]:
        store = self.plans
        rows = zip(self.ids, self.plan_ids.tolist(), self.scores.tolist(),
                   self.selected.tolist(), self.n_plans.tolist())
        for pid, ids, scores, selected, n in rows:
            yield _person(store, pid, ids, scores, selected, n)

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        attrs = f", attributes={list(self.attributes.names)}" if len(self.attributes) else ""
        return f"Population(persons={len(self)}, plans={len(self.plans)}{attrs})"


_new = object.__new__
_set = object.__setattr__


def _person(store, person_id, ids, scores, selected, n) -> Person:
    """A :class:`Person` from one table row (already valid: skip checks)."""
    person = _new(Person)
    view = Plan._view
    _set(person, "id", person_id)
    _set(person, "plans", tuple([view(store, j) for j in ids[:n]]))
    _set(person, "selected", selected)
    _set(person, "scores", tuple([None if s != s else s for s in scores[:n]]))
    return person


def _tables(persons: tuple[Person, ...], store: PlanStore):
    """``(plans, plan_ids, scores, selected, n_plans)`` for these persons.

    Plans that are rows of ``store`` (or of an older snapshot of its lineage)
    keep their rows. Other plans are appended once per distinct object, so a
    plan shared by several persons is not duplicated.
    """
    width = max((len(p.plans) for p in persons), default=1)
    token, n0 = store.token, store.n_plans
    new: list[Plan] = []
    index: dict = {}  # id(plan) or (token, row) of another store -> row, for new plans
    # flat row-major lists: one numpy conversion each, no nested lists
    plan_ids, scores, selected, n_plans = [], [], [], []
    add_id, add_scores = plan_ids.append, scores.extend
    nan = math.nan
    get = index.get
    for p in persons:
        for plan in p.plans:
            j = plan._row
            if plan._token != token or j >= n0:
                # a row of another store is one plan, however many views of it
                key = id(plan) if plan._token is None else (plan._token, j)
                j = get(key)
                if j is None:
                    j = index[key] = n0 + len(new)
                    new.append(plan)
            add_id(j)
        add_scores([nan if s is None else s for s in p.scores])
        n = len(p.plans)
        pad = width - n
        if pad:
            plan_ids += [-1] * pad
            scores += [nan] * pad
        selected.append(p.selected)
        n_plans.append(n)
    if new:
        store = store.append_plans(new)
        for j, plan in enumerate(new, n0):
            plan._stored_as(store, j)
    shape = (len(persons), width)
    return (
        store,
        np.array(plan_ids, dtype=np.int32).reshape(shape),
        np.array(scores, dtype=float).reshape(shape),
        np.array(selected, dtype=np.int32),
        np.array(n_plans, dtype=np.int32),
    )


def _remap(plan_ids: np.ndarray, live: np.ndarray, n_rows: int) -> np.ndarray:
    """``plan_ids`` after compacting the store to the sorted rows ``live``."""
    new_row = np.full(n_rows + 1, -1, dtype=np.int32)  # last entry: -1 stays -1
    new_row[live] = np.arange(len(live), dtype=np.int32)
    return new_row[plan_ids]


def _maybe_compact(store: PlanStore, plan_ids: np.ndarray, n_plans: np.ndarray):
    """Compact ``store`` if it has more than twice as many rows as the
    persons hold plans (order-preserving, so results do not depend on it)."""
    held = max(int(n_plans.sum()), len(n_plans), 1)
    if len(store) <= 2 * held:
        return store, plan_ids
    live = np.unique(plan_ids[plan_ids >= 0]).astype(np.int64)
    return store.compact(live), _remap(plan_ids, live, len(store))


def locate_activities(plan: Plan, network: Network) -> Plan:
    """``plan`` with every activity's coordinates set (its node's where it has
    none). Returns ``plan`` itself if nothing changes."""
    elements = plan.elements
    if all(e.x is not None for e in elements[::2]):
        return plan
    xy = network.node_xy
    out = list(elements)
    for i in range(0, len(out), 2):
        act = out[i]
        if act.x is None:
            x, y = xy[act.node].tolist()
            out[i] = Activity(act.type, act.node, act.end_time, act.duration, x, y, act.link)
    return Plan(tuple(out))
