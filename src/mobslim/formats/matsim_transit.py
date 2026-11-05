"""Readers for MATSim ``transitSchedule`` (v1/v2) and ``vehicleDefinitions``
(v1/v2) files. ``.gz`` is allowed. Schedules are streamed with ``iterparse``.

Ignored for now: ``attributes``, ``chainedDepartures`` and stop areas.
"""

import gzip
import xml.etree.ElementTree as ET

from mobslim.network import Network
from mobslim.timeutils import parse_time
from mobslim.transit.core import Transit, TransitParams
from mobslim.transit.schedule import TransitSchedule, build_schedule
from mobslim.transit.vehicles import TransitVehicles, build_vehicles


def _open(path):
    path = str(path)
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _tag(el) -> str:
    t = el.tag
    return t.rsplit("}", 1)[-1] if "}" in t else t


def _time(value):
    return None if value is None or value == "" else parse_time(value)


def _children(el, name):
    return [c for c in el if _tag(c) == name]


def _child(el, name):
    for c in el:
        if _tag(c) == name:
            return c
    return None


def read_transit_schedule_xml(path, network: Network) -> TransitSchedule:
    stops, routes, transfers = [], [], []
    line = None
    with _open(path) as f:
        for event, el in ET.iterparse(f, events=("start", "end")):
            tag = _tag(el)
            if event == "start":
                if tag == "transitLine":
                    line = el.get("id")
                continue
            if tag == "stopFacility":
                stops.append((el.get("id"), float(el.get("x")), float(el.get("y")),
                              el.get("linkRefId"), el.get("isBlocking") == "true"))
                el.clear()
            elif tag == "relation":
                transfers.append((el.get("fromStop"), el.get("toStop"),
                                  float(el.get("transferTime"))))
                el.clear()
            elif tag == "transitRoute":
                routes.append(_read_route(line, el))
                el.clear()
            elif tag == "transitLine":
                el.clear()
    return build_schedule(network, stops, routes, transfers)


def _read_route(line, el) -> dict:
    mode_el = _child(el, "transportMode")
    profile = _child(el, "routeProfile")
    route = _child(el, "route")
    deps = _child(el, "departures")
    stops = [(s.get("refId"), _time(s.get("arrivalOffset")), _time(s.get("departureOffset")),
              s.get("awaitDeparture") == "true")
             for s in (_children(profile, "stop") if profile is not None else [])]
    links = [lk.get("refId") for lk in _children(route, "link")] if route is not None else []
    departures = [(d.get("id"), parse_time(d.get("departureTime")), d.get("vehicleRefId"))
                  for d in (_children(deps, "departure") if deps is not None else [])]
    return dict(line=line, route=el.get("id"), mode=(mode_el.text or "").strip() if mode_el is not None else "pt",
                links=links, stops=stops, departures=departures)


def read_transit_vehicles_xml(path) -> TransitVehicles:
    with _open(path) as f:
        root = ET.parse(f).getroot()
    types, vehicles = [], []
    for el in root:
        tag = _tag(el)
        if tag == "vehicleType":
            types.append(_read_type(el))
        elif tag == "vehicle":
            vehicles.append((el.get("id"), el.get("type")))
    return build_vehicles(types, vehicles)


def _read_type(el) -> dict:
    t = {"id": el.get("id")}
    cap = _child(el, "capacity")
    if cap is not None:
        if cap.get("seats") is not None:  # v2
            t["seats"] = int(cap.get("seats"))
            t["standing"] = int(cap.get("standingRoomInPersons", 0))
        else:  # v1
            seats, standing = _child(cap, "seats"), _child(cap, "standingRoom")
            if seats is not None:
                t["seats"] = int(seats.get("persons"))
            if standing is not None:
                t["standing"] = int(standing.get("persons"))
    for name, key, attr in (("length", "length", "meter"),
                            ("maximumVelocity", "max_speed", "meterPerSecond"),
                            ("accessTime", "access_time", "secondsPerPerson"),
                            ("egressTime", "egress_time", "secondsPerPerson"),
                            ("passengerCarEquivalents", "pce", "pce")):
        c = _child(el, name)
        if c is not None and c.get(attr) is not None:
            t[key] = float(c.get(attr))
    door = _child(el, "doorOperation")
    if door is not None and door.get("mode"):
        t["doors"] = door.get("mode")
    attrs = _child(el, "attributes")  # v2 keeps the pt parameters as attributes
    if attrs is not None:
        for a in _children(attrs, "attribute"):
            name, text = a.get("name"), (a.text or "").strip()
            if name == "accessTimeInSecondsPerPerson":
                t["access_time"] = float(text)
            elif name == "egressTimeInSecondsPerPerson":
                t["egress_time"] = float(text)
            elif name == "doorOperationMode":
                t["doors"] = text
    return t


def read_transit(schedule_path, vehicles_path, network: Network,
                 params: TransitParams = TransitParams()) -> Transit:
    return Transit(read_transit_schedule_xml(schedule_path, network),
                   read_transit_vehicles_xml(vehicles_path), params)
