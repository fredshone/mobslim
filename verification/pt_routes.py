"""MATSim's pt routes (``default_pt``) as mobslim :class:`~mobslim.population.TransitPassengerRoute`s.

Layer 2 runs mobslim on MATSim's routes, so pt riders must take the same
lines between the same stops as in MATSim (mobslim would otherwise route
them afresh with its RAPTOR). Each pt leg of a MATSim trip
(``{"transitLineId", "transitRouteId", "accessFacilityId", "egressFacilityId"}``)
becomes one ride ``(board_route_stop, alight_route_stop)``; the walks are
implied, as in mobslim. A pt trip MATSim's router made walk-only gets an
empty ``TransitPassengerRoute`` (walked).
"""

from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET

import numpy as np

from mobslim.population import (
    Leg,
    Person,
    Plan,
    Population,
    TransitPassengerRoute,
)
from mobslim.timeutils import parse_time


def _trips(plan_el, transit_modes):
    """Per trip of a MATSim plan: (routing mode, [rides as dicts], travel time)."""
    trips, legs = [], []
    for el in plan_el:
        if el.tag in ("act", "activity"):
            if el.get("type", "").endswith(" interaction"):
                continue
            if legs:
                trips.append(legs)
            legs = []
        elif el.tag == "leg":
            legs.append(el)
    out = []
    for legs in trips:
        mode = None
        for el in legs:
            for a in el.iter("attribute"):
                if a.get("name") == "routingMode" and a.text:
                    mode = a.text.strip()
        rides, tt = [], 0.0
        for el in legs:
            t = el.get("trav_time")
            tt += float(parse_time(t)) if t and t != "undefined" else 0.0
            r = el.find("route")
            if el.get("mode") in transit_modes and r is not None and r.get("type") == "default_pt":
                rides.append(json.loads(r.text))
        out.append((mode, rides, tt))
    return out


def _route_stop_after(schedule, line, route, stop, after: int = -1) -> int:
    """Global route-stop index of ``stop`` on ``route``, the first after ``after``."""
    r = schedule.route_index(line, route)
    s = schedule.stop_ids.index(stop)
    k0, k1 = schedule.route_stop_offsets[r], schedule.route_stop_offsets[r + 1]
    hits = k0 + np.flatnonzero(schedule.rs_stop[k0:k1] == s)
    hits = hits[hits > after]
    if not len(hits):
        raise KeyError(f"Route {line}/{route} does not serve stop {stop!r} after route stop {after}.")
    return int(hits[0])


def apply_matsim_pt_routes(population: Population, plans_path, schedule,
                           transit_modes=("pt",)) -> Population:
    """``population`` with the pt legs of each selected plan routed as in ``plans_path``."""
    transit_modes = set(transit_modes)
    by_person = {}
    with gzip.open(plans_path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "person":
                continue
            plans = el.findall("plan")
            plan = next((p for p in plans if p.get("selected") == "yes"), plans[0] if plans else None)
            if plan is not None:
                by_person[el.get("id")] = _trips(plan, transit_modes)
            el.clear()

    persons = []
    for i in range(len(population)):
        person = population[i]
        trips = by_person.get(str(person.id))
        plan = person.selected_plan
        if trips is None or not any(leg.mode in transit_modes for leg in plan.legs):
            persons.append(person)
            continue
        elements, k = [], 0
        for e in plan.elements:
            if isinstance(e, Leg):
                mode, rides, tt = trips[k]
                k += 1
                if e.mode in transit_modes:
                    pairs = []
                    for rd in rides:
                        line, route = rd["transitLineId"], rd["transitRouteId"]
                        board = _route_stop_after(schedule, line, route, rd["accessFacilityId"])
                        alight = _route_stop_after(schedule, line, route, rd["egressFacilityId"], board)
                        pairs.append((board, alight))
                    e = Leg(e.mode, TransitPassengerRoute(pairs, tt, 0.0))
            elements.append(e)
        plans = list(person.plans)
        plans[person.selected] = Plan(tuple(elements))
        persons.append(Person(person.id, tuple(plans), person.selected, person.scores))
    return Population(persons)
