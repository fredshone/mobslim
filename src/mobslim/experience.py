"""Experienced plans: what each person actually did in a mobsim run.

An :class:`ExperiencedPlan` holds, as objects, the actual activity start/end
times and leg departure/arrival times, distances and links of one person.
Experienced plans are kept separate from the (intended) plans. They are the
readable form of a run, used by per-plan scorers, analysis, and as an
*observation* for learning-based replanning.

In the controller they are built from the mobsim's
:class:`~mobslim.trajectories.Trajectories` on request
(``Trajectories.experienced_plans()``, ``IterationState.experienced``).
:class:`ExperiencedPlanCollector` builds them from events instead, for event
streams, and serves as an independent reference in the tests.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from mobslim.events import (
    ActivityEnd,
    ActivityStart,
    EventHandler,
    LinkEnter,
    PersonArrival,
    PersonDeparture,
    PersonStuck,
    Teleportation,
)
from mobslim.network import Network
from mobslim.population import Population


@dataclass(frozen=True, slots=True)
class ExperiencedActivity:
    type: str
    node: int
    start: Optional[float]  # None for the first activity of the day
    end: Optional[float]  # None for the last activity (or if stuck in it)


@dataclass(frozen=True, slots=True)
class ExperiencedRide:
    """One ride of a pt leg. Times are ``None`` where it did not happen."""

    board_route_stop: int
    alight_route_stop: int
    trip: int  # departure boarded, -1 = none
    wait_start: Optional[float]
    board_time: Optional[float]
    alight_time: Optional[float]
    distance: float = 0.0  # in-vehicle [m], 0 if not completed

    @property
    def wait_time(self) -> Optional[float]:
        if self.board_time is None or self.wait_start is None:
            return None
        return self.board_time - self.wait_start

    @property
    def in_vehicle_time(self) -> Optional[float]:
        if self.alight_time is None or self.board_time is None:
            return None
        return self.alight_time - self.board_time


@dataclass(frozen=True, slots=True)
class ExperiencedLeg:
    mode: str
    departure_time: float
    arrival_time: Optional[float]  # None if the person got stuck
    distance: float
    links: tuple[int, ...]
    # pt legs: the rides started (possibly none: walked); None for other legs
    rides: Optional[tuple[ExperiencedRide, ...]] = None

    @property
    def travel_time(self) -> Optional[float]:
        if self.arrival_time is None:
            return None
        return self.arrival_time - self.departure_time


ExperiencedElement = Union[ExperiencedActivity, ExperiencedLeg]


@dataclass(frozen=True, slots=True)
class ExperiencedPlan:
    """``activities`` and ``legs`` are split out once at construction."""

    elements: tuple[ExperiencedElement, ...]
    stuck: bool = False
    activities: tuple[ExperiencedActivity, ...] = field(init=False, repr=False, compare=False)
    legs: tuple[ExperiencedLeg, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        elements = tuple(self.elements)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(
            self, "activities", tuple(e for e in elements if type(e) is ExperiencedActivity)
        )
        object.__setattr__(self, "legs", tuple(e for e in elements if type(e) is ExperiencedLeg))


class _Builder:
    __slots__ = ("elements", "act", "leg", "stuck")

    def __init__(self, first_type: str, first_node: int):
        self.elements: list = []
        self.act = (first_type, first_node, None)  # open activity
        self.leg = None  # open leg: [mode, dep, distance, links]
        self.stuck = False


class ExperiencedPlanCollector(EventHandler):
    """Builds one :class:`ExperiencedPlan` per person from events."""

    handles = (ActivityStart, ActivityEnd, PersonDeparture, PersonArrival,
               LinkEnter, Teleportation, PersonStuck)

    def __init__(self, network: Network, population: Population):
        self._length = network.length.tolist()
        self._first = [
            (plan.elements[0].type, plan.elements[0].node)
            for plan in population.selected_plans()
        ]
        self.reset(0)

    def reset(self, iteration: int) -> None:
        self._b = [_Builder(t, n) for t, n in self._first]
        self._plans: Optional[list[ExperiencedPlan]] = None

    def handle(self, e) -> None:
        b = self._b[e.person]
        t = type(e)
        if t is LinkEnter:
            b.leg[2] += self._length[e.link]
            b.leg[3].append(e.link)
        elif t is ActivityEnd:
            act_type, node, start = b.act
            b.elements.append(ExperiencedActivity(act_type, node, start, e.time))
            b.act = None
        elif t is PersonDeparture:
            b.leg = [e.mode, e.time, 0.0, []]
        elif t is Teleportation:
            b.leg[2] = e.distance
        elif t is PersonArrival:
            mode, dep, dist, links = b.leg
            b.elements.append(ExperiencedLeg(mode, dep, e.time, dist, tuple(links)))
            b.leg = None
        elif t is ActivityStart:
            b.act = (e.act_type, e.node, e.time)
        elif t is PersonStuck:
            b.stuck = True
            if b.leg is not None:
                mode, dep, dist, links = b.leg
                b.elements.append(ExperiencedLeg(mode, dep, None, dist, tuple(links)))
                b.leg = None

    def finish(self) -> None:
        plans = []
        for b in self._b:
            elements = list(b.elements)
            if b.act is not None:
                act_type, node, start = b.act
                elements.append(ExperiencedActivity(act_type, node, start, None))
            plans.append(ExperiencedPlan(tuple(elements), b.stuck))
        self._plans = plans

    def plans(self) -> list[ExperiencedPlan]:
        """Experienced plans indexed by person index (after ``finish``)."""
        if self._plans is None:
            raise RuntimeError("ExperiencedPlanCollector.finish() has not been called.")
        return self._plans
