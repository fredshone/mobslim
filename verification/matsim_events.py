"""Rewrite mobslim's ``EventsXMLWriter`` output into events MATSim's scoring reads.

mobslim's events follow its own model, so MATSim's ``EventsToScore`` needs
more than it gets:

* Activity, departure and arrival events get the ``link`` of the activity
  (from the plans file MATSim also reads); mobslim writes nodes.
* Car legs get ``PersonEntersVehicle`` and ``vehicle enters traffic`` after
  the departure and ``vehicle leaves traffic`` and ``PersonLeavesVehicle``
  before the arrival (which replace mobslim's ``left link`` on the arrival
  link).
* A pt trip, one ``pt`` leg in mobslim, is expanded into MATSim's legs: a
  walk to the stop, then per ride a ``pt interaction``, the ``pt`` leg
  (from the start of the wait to alighting) and a walk to the next stop or
  the destination. Walk distances are unknown (``NaN``), as in MATSim's own
  pt walks; they only count if priced.
* ``waitingForPt`` gets ``destinationStop``: the stop where the passenger
  alights (mobslim's event has only the boarding stop).
* Transit vehicles' link events are left out (MATSim's leg builder only
  knows vehicles that entered traffic with a driver; scoring doesn't use them).
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path


def activity_links(plans_path) -> dict[str, list[str]]:
    """Per person: the ``link`` of each real activity of the selected plan."""
    out = {}
    opener = gzip.open if str(plans_path).endswith(".gz") else open
    with opener(plans_path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "person":
                continue
            plans = el.findall("plan")
            plan = next((p for p in plans if p.get("selected") == "yes"), plans[0] if plans else None)
            if plan is not None:
                out[el.get("id")] = [a.get("link") for a in plan
                                     if a.tag in ("act", "activity")
                                     and not a.get("type", "").endswith(" interaction")]
            el.clear()
    return out


def _read(path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return [dict(el.attrib) for _, el in ET.iterparse(f, events=("end",)) if el.tag == "event"]


def _write(path, events: list[dict]) -> None:
    from mobslim.events import _xml_escape

    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n<events version="1.0">\n')
        for e in events:
            body = " ".join(f'{k}="{_xml_escape(v)}"' for k, v in e.items())
            f.write(f"\t<event {body} />\n")
        f.write("</events>\n")


def to_matsim(events_path, plans_path, out_path, network_modes=("car",),
              transit_modes=("pt",)) -> Path:
    events = _read(events_path)
    links = activity_links(plans_path)
    network_modes, transit_modes = set(network_modes), set(transit_modes)

    # mobslim's left link on the arrival link: dropped (MATSim ends with
    # "vehicle leaves traffic"). It is the person's last event before arrival.
    last_event: dict[str, int] = {}
    drop = set()
    for i, e in enumerate(events):
        p = e.get("person")
        if p is None:
            continue
        if e["type"] == "arrival" and p in last_event and events[last_event[p]]["type"] == "left link":
            drop.add(last_event[p])
        last_event[p] = i

    out: list[dict] = []
    at_stop: dict[str, str] = {}  # transit vehicle -> stop it is at
    waiting_event: dict[str, dict] = {}  # person -> its last waitingForPt (gets destinationStop)
    act = {}  # person -> index of the current (or last ended) real activity
    leg_mode = {}  # person -> mode of the open leg
    walking = {}  # person -> a pt walk leg is open

    def ev(t, typ, **attrs):
        out.append({"time": t, "type": typ, **attrs})

    def stage(t, p, link):
        ev(t, "actstart", person=p, link=link, actType="pt interaction")
        ev(t, "actend", person=p, link=link, actType="pt interaction")

    for i, e in enumerate(events):
        if i in drop:
            continue
        typ, t, p = e["type"], e["time"], e.get("person")
        if typ in ("entered link", "left link"):
            if e.get("vehicle") in links:  # a person's car; transit vehicles are left out
                out.append({k2: v for k2, v in e.items() if k2 != "person"})
            continue
        if typ == "VehicleArrivesAtFacility":
            at_stop[e["vehicle"]] = e["facility"]
        if p is None or p not in links:
            out.append(e)
            continue
        k = act.setdefault(p, 0)
        here = links[p][min(k, len(links[p]) - 1)]
        if typ == "actend":
            ev(t, "actend", person=p, link=here, actType=e["actType"])
        elif typ == "actstart":
            k = act[p] = k + 1
            ev(t, "actstart", person=p, link=links[p][k], actType=e["actType"])
        elif typ == "departure":
            mode = leg_mode[p] = e["legMode"]
            if mode in transit_modes:
                walking[p] = True
                ev(t, "departure", person=p, link=here, legMode="walk", computationalRoutingMode=mode)
            else:
                ev(t, "departure", person=p, link=here, legMode=mode, computationalRoutingMode=mode)
                if mode in network_modes:
                    ev(t, "PersonEntersVehicle", person=p, vehicle=p)
                    ev(t, "vehicle enters traffic", person=p, link=here, vehicle=p,
                       networkMode=mode, relativePosition="1.0")
        elif typ == "waitingForPt":
            if walking.pop(p, False):
                ev(t, "travelled", person=p, distance="NaN", mode="walk")
                ev(t, "arrival", person=p, link=here, legMode="walk")
                stage(t, p, here)
                ev(t, "departure", person=p, link=here, legMode=leg_mode[p],
                   computationalRoutingMode=leg_mode[p])
            e = {**e, "agent": p, "destinationStop": e["atStop"]}  # set on alighting
            waiting_event[p] = e
            out.append(e)
        elif typ == "PersonLeavesVehicle" and leg_mode.get(p) in transit_modes:
            if p in waiting_event and e["vehicle"] in at_stop:
                waiting_event.pop(p)["destinationStop"] = at_stop[e["vehicle"]]
            out.append(e)
            ev(t, "arrival", person=p, link=here, legMode=leg_mode[p])
            stage(t, p, here)
            ev(t, "departure", person=p, link=here, legMode="walk",
               computationalRoutingMode=leg_mode[p])
            walking[p] = True
        elif typ == "arrival":
            mode = leg_mode.pop(p, e["legMode"])
            there = links[p][min(k + 1, len(links[p]) - 1)]
            if mode in transit_modes:
                walking.pop(p, None)
                ev(t, "travelled", person=p, distance="NaN", mode="walk")
                ev(t, "arrival", person=p, link=there, legMode="walk")
            else:
                if mode in network_modes:
                    ev(t, "vehicle leaves traffic", person=p, link=there, vehicle=p,
                       networkMode=mode, relativePosition="1.0")
                    ev(t, "PersonLeavesVehicle", person=p, vehicle=p)
                ev(t, "arrival", person=p, link=there, legMode=mode)
        else:
            out.append(e)
    out_path = Path(out_path)
    _write(out_path, out)
    return out_path
