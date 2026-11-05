"""Typed simulation events, handlers, and the events manager.

Events mirror MATSim's core event types (names in ``matsim_type``). Persons,
links and nodes are referenced by integer index; map back to external ids via
``population.ids`` / ``network.link_ids`` / ``network.node_ids``.

Events are an optional view of a mobsim run. The mobsim's primary output is
:class:`~mobslim.trajectories.Trajectories` (arrays), from which the
controller derives scores, travel times and metrics. Events are for handlers
that want them: writers, analysis, animation. The mobsim pushes them into an
:class:`EventsManager`, which fans them out to registered
:class:`EventHandler` objects, and builds an event only if some handler is
registered for its type. Handlers aggregate on the fly, so nothing needs to
hold the full event stream unless you ask for it (:class:`EventLog`).

Events are slotted but deliberately *not* frozen: a mobsim may create one per
vehicle move, and frozen dataclasses are about 3x slower to construct. The
same instance is passed to every handler, so treat events as read-only.

Events about a person subclass :class:`PersonEvent` (``time, person``);
transit vehicle events subclass :class:`Event` (``time``) directly.

Custom event types are supported: subclass :class:`Event` (or
:class:`PersonEvent`) and push instances
into :meth:`EventsManager.process`; handlers receive them if ``handles`` is
``None`` or lists the type (or a base class of it).
"""

import gzip
from dataclasses import asdict, dataclass, fields
from typing import ClassVar, Iterable, Optional

from mobslim.timeutils import format_time


@dataclass(slots=True)
class Event:
    time: float
    matsim_type: ClassVar[str] = "event"


@dataclass(slots=True)
class PersonEvent(Event):
    """An event about one person (all event types except transit vehicles')."""

    person: int


@dataclass(slots=True)
class ActivityStart(PersonEvent):
    act_type: str
    node: int
    matsim_type: ClassVar[str] = "actstart"


@dataclass(slots=True)
class ActivityEnd(PersonEvent):
    act_type: str
    node: int
    matsim_type: ClassVar[str] = "actend"


@dataclass(slots=True)
class PersonDeparture(PersonEvent):
    mode: str
    node: int
    matsim_type: ClassVar[str] = "departure"


@dataclass(slots=True)
class PersonArrival(PersonEvent):
    mode: str
    node: int
    matsim_type: ClassVar[str] = "arrival"


@dataclass(slots=True)
class LinkEnter(PersonEvent):
    link: int
    matsim_type: ClassVar[str] = "entered link"


@dataclass(slots=True)
class LinkLeave(PersonEvent):
    link: int
    matsim_type: ClassVar[str] = "left link"


@dataclass(slots=True)
class Teleportation(PersonEvent):
    mode: str
    distance: float
    matsim_type: ClassVar[str] = "travelled"


@dataclass(slots=True)
class PersonStuck(PersonEvent):
    mode: Optional[str]
    link: Optional[int]
    matsim_type: ClassVar[str] = "stuckAndAbort"


# -- transit. ``trip`` is a departure index,
# ``stop`` a stop index; vehicle events carry no person.


@dataclass(slots=True)
class TransitDriverStarts(Event):
    trip: int
    matsim_type: ClassVar[str] = "TransitDriverStarts"


@dataclass(slots=True)
class VehicleArrivesAtFacility(Event):
    trip: int
    stop: int
    delay: float  # against the scheduled arrival [s]
    matsim_type: ClassVar[str] = "VehicleArrivesAtFacility"


@dataclass(slots=True)
class VehicleDepartsAtFacility(Event):
    trip: int
    stop: int
    delay: float  # against the scheduled departure [s]
    matsim_type: ClassVar[str] = "VehicleDepartsAtFacility"


@dataclass(slots=True)
class AgentWaitingForPt(PersonEvent):
    stop: int
    matsim_type: ClassVar[str] = "waitingForPt"


@dataclass(slots=True)
class PersonEntersVehicle(PersonEvent):
    trip: int
    matsim_type: ClassVar[str] = "PersonEntersVehicle"


@dataclass(slots=True)
class PersonLeavesVehicle(PersonEvent):
    trip: int
    matsim_type: ClassVar[str] = "PersonLeavesVehicle"


@dataclass(slots=True)
class TransitVehicleLinkEnter(Event):
    trip: int
    link: int
    matsim_type: ClassVar[str] = "entered link"


@dataclass(slots=True)
class TransitVehicleLinkLeave(Event):
    trip: int
    link: int
    matsim_type: ClassVar[str] = "left link"


TRANSIT_EVENT_TYPES: tuple[type[Event], ...] = (
    TransitDriverStarts,
    VehicleArrivesAtFacility,
    VehicleDepartsAtFacility,
    AgentWaitingForPt,
    PersonEntersVehicle,
    PersonLeavesVehicle,
    TransitVehicleLinkEnter,
    TransitVehicleLinkLeave,
)

EVENT_TYPES: tuple[type[Event], ...] = (
    ActivityStart,
    ActivityEnd,
    PersonDeparture,
    PersonArrival,
    LinkEnter,
    LinkLeave,
    Teleportation,
    PersonStuck,
) + TRANSIT_EVENT_TYPES


class EventHandler:
    """Base class for event handlers.

    Set ``handles`` to the event types you want (``None`` = all) and implement
    :meth:`handle`. ``reset`` is called before each iteration's mobsim run and
    ``finish`` after it.
    """

    handles: ClassVar[Optional[tuple[type[Event], ...]]] = None

    def handle(self, event: Event) -> None:
        raise NotImplementedError

    def reset(self, iteration: int) -> None:
        pass

    def finish(self) -> None:
        pass


class EventsManager:
    """Dispatches events to handlers by event type."""

    def __init__(self, handlers: Iterable[EventHandler] = ()):
        self.handlers: list[EventHandler] = []
        self._dispatch: dict[type, list] = {t: [] for t in EVENT_TYPES}
        for h in handlers:
            self.add_handler(h)

    @staticmethod
    def _wants(handler: EventHandler, event_type: type) -> bool:
        wanted = getattr(handler, "handles", None)
        return wanted is None or issubclass(event_type, tuple(wanted))

    def add_handler(self, handler: EventHandler) -> EventHandler:
        self.handlers.append(handler)
        # extend the existing lists in place, so lists returned by
        # handlers_for() stay valid when handlers are added later
        for t, fns in self._dispatch.items():
            if self._wants(handler, t):
                fns.append(handler.handle)
        return handler

    def handlers_for(self, event_type: type) -> list:
        """The (live) list of ``handle`` callables that receive ``event_type``.

        A mobsim can bind this once and skip building events nobody listens to.
        """
        fns = self._dispatch.get(event_type)
        if fns is None:
            fns = self._dispatch[event_type] = [
                h.handle for h in self.handlers if self._wants(h, event_type)
            ]
        return fns

    def process(self, event: Event) -> None:
        fns = self._dispatch.get(type(event))
        if fns is None:
            fns = self.handlers_for(type(event))
        for fn in fns:
            fn(event)

    def process_many(self, events: Iterable[Event]) -> None:
        process = self.process
        for e in events:
            process(e)

    def reset(self, iteration: int) -> None:
        for h in self.handlers:
            h.reset(iteration)

    def finish(self) -> None:
        for h in self.handlers:
            h.finish()


class EventLog(EventHandler):
    """Keeps every event in memory. Useful for tests, notebooks and animation."""

    def __init__(self):
        self.events: list[Event] = []

    def handle(self, event: Event) -> None:
        self.events.append(event)

    def reset(self, iteration: int) -> None:
        self.events = []

    def of_type(self, *types: type[Event]) -> list[Event]:
        return [e for e in self.events if isinstance(e, types)]

    def to_dataframe(self):
        import pandas as pd

        rows = []
        for e in self.events:
            row = {"type": e.matsim_type}
            for f in fields(e):
                row[f.name] = getattr(e, f.name)
            rows.append(row)
        return pd.DataFrame(rows)


class EventsXMLWriter(EventHandler):
    """Writes events in MATSim's ``events_v1`` XML format (``.gz`` supported).

    Useful for comparing runs against Java MATSim tooling (e.g. Via, simwrapper).
    """

    def __init__(self, path, network, population, transit=None, vehicle_per_trip: bool = False):
        """``vehicle_per_trip``: name transit vehicles ``<vehicle>#<trip>``.
        MobSlim simulates trips, not vehicles, so a late trip can still run
        when the next trip of its vehicle starts; MATSim tools that track
        vehicles need a distinct id per trip then."""
        self.path = str(path)
        self._link_ids = network.link_ids
        self._node_ids = network.node_ids
        self._person_ids = population.ids
        self._transit = transit
        self._vehicle_per_trip = vehicle_per_trip
        self._file = None

    def _vehicle(self, trip: int) -> str:
        if self._transit is None:
            return f"trip_{trip}"
        vehicle = str(self._transit.vehicle_id(trip))
        return f"{vehicle}#{trip}" if self._vehicle_per_trip else vehicle

    def _open(self):
        opener = gzip.open if self.path.endswith(".gz") else open
        self._file = opener(self.path, "wt", encoding="utf-8")
        self._file.write('<?xml version="1.0" encoding="utf-8"?>\n<events version="1.0">\n')

    def handle(self, event: Event) -> None:
        if self._file is None:
            self._open()
        attrs = {"time": repr(float(event.time)), "type": event.matsim_type}
        if isinstance(event, PersonEvent):
            attrs["person"] = self._person_ids.id(event.person)
        if isinstance(event, (TransitVehicleLinkEnter, TransitVehicleLinkLeave)):
            attrs["link"] = self._link_ids.id(event.link)
            attrs["vehicle"] = self._vehicle(event.trip)
        elif isinstance(event, TransitDriverStarts):
            vehicle = self._vehicle(event.trip)
            attrs["driverId"] = f"pt_{vehicle}"
            attrs["vehicleId"] = vehicle
            if self._transit is not None:
                s = self._transit.schedule
                line, route, dep = s.dep_ids.id(event.trip)
                attrs["transitLineId"], attrs["transitRouteId"], attrs["departureId"] = line, route, dep
        elif isinstance(event, (VehicleArrivesAtFacility, VehicleDepartsAtFacility)):
            attrs["vehicle"] = self._vehicle(event.trip)
            attrs["facility"] = self._stop(event.stop)
            attrs["delay"] = f"{event.delay:.1f}"
        elif isinstance(event, AgentWaitingForPt):
            attrs["agent"] = attrs["person"]
            attrs["atStop"] = self._stop(event.stop)
        elif isinstance(event, (PersonEntersVehicle, PersonLeavesVehicle)):
            attrs["vehicle"] = self._vehicle(event.trip)
        elif isinstance(event, (ActivityStart, ActivityEnd)):
            attrs["actType"] = event.act_type
            attrs["node"] = self._node_ids.id(event.node)
        elif isinstance(event, (PersonDeparture, PersonArrival)):
            attrs["legMode"] = event.mode
            attrs["node"] = self._node_ids.id(event.node)
        elif isinstance(event, (LinkEnter, LinkLeave)):
            attrs["link"] = self._link_ids.id(event.link)
            attrs["vehicle"] = attrs["person"]
        elif isinstance(event, Teleportation):
            attrs["mode"] = event.mode
            attrs["distance"] = f"{event.distance:.1f}"
        elif isinstance(event, PersonStuck):
            if event.mode is not None:
                attrs["legMode"] = event.mode
            if event.link is not None:
                attrs["link"] = self._link_ids.id(event.link)
        body = " ".join(f'{k}="{_xml_escape(v)}"' for k, v in attrs.items())
        self._file.write(f"\t<event {body} />\n")

    def _stop(self, stop: int):
        return self._transit.schedule.stop_ids.id(stop) if self._transit is not None else stop

    def finish(self) -> None:
        if self._file is None:
            self._open()
        self._file.write("</events>\n")
        self._file.close()
        self._file = None


def _xml_escape(value) -> str:
    return (
        str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


def describe(event: Event) -> str:
    """Human-readable one-liner, e.g. for debugging."""
    attrs = {k: v for k, v in asdict(event).items() if k != "time"}
    return f"{format_time(event.time)} {event.matsim_type} {attrs}"
