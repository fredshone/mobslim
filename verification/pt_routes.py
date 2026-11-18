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

import numpy as np

from mobslim.formats._xmlstream import iter_root_children
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


BOARDING_TOLERANCE = 300.0  # s: MATSim's boardingTime against the schedule's times


def _route_stop_after(schedule, line, route, stop, after: int = -1, departure=None) -> int:
    """Global route-stop index of ``stop`` on ``route``, the first after
    ``after``; with ``departure`` (MATSim's ``boardingTime``, within seconds
    of a trip's arrival or departure there), the visit with a trip's
    arrival or departure nearest to it (within :data:`BOARDING_TOLERANCE`).
    A route may serve a stop many times (Berlin's circle line laps the ring
    all day): a passenger waits for a trip at one of the visits, the one
    MATSim's router chose."""
    r = schedule.route_index(line, route)
    s = schedule.stop_ids.index(stop)
    k0, k1 = schedule.route_stop_offsets[r], schedule.route_stop_offsets[r + 1]
    hits = k0 + np.flatnonzero(schedule.rs_stop[k0:k1] == s)
    hits = hits[hits > after]
    if not len(hits):
        raise KeyError(f"Route {line}/{route} does not serve stop {stop!r} after route stop {after}.")
    if departure is not None and len(hits) > 1:
        d0, d1 = schedule.route_dep_offsets[r], schedule.route_dep_offsets[r + 1]
        trips = schedule.dep_time[d0:d1]
        if len(trips):
            gap = [min(np.abs(trips + schedule.rs_arrival[k] - departure).min(),
                       np.abs(trips + schedule.rs_departure[k] - departure).min())
                   for k in hits.tolist()]
            if min(gap) <= BOARDING_TOLERANCE:
                return int(hits[int(np.argmin(gap))])
    return int(hits[0])


def apply_matsim_pt_routes(population: Population, plans_path, schedule,
                           transit_modes=("pt",)) -> Population:
    """``population`` with the pt legs of each selected plan routed as in ``plans_path``."""
    transit_modes = set(transit_modes)
    by_person = {}
    with gzip.open(plans_path, "rb") as f:
        for el in iter_root_children(f, "person"):  # streamed, one person at a time
            plans = el.findall("plan")
            plan = next((p for p in plans if p.get("selected") == "yes"), plans[0] if plans else None)
            if plan is not None:
                by_person[el.get("id")] = _trips(plan, transit_modes)

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
                        t = rd.get("boardingTime")  # ("undefined": none)
                        board = _route_stop_after(
                            schedule, line, route, rd["accessFacilityId"],
                            departure=parse_time(t) if t and t != "undefined" else None)
                        alight = _route_stop_after(schedule, line, route, rd["egressFacilityId"], board)
                        pairs.append((board, alight))
                    e = Leg(e.mode, TransitPassengerRoute(pairs, tt, 0.0))
            elements.append(e)
        plans = list(person.plans)
        plans[person.selected] = Plan(tuple(elements))
        persons.append(Person(person.id, tuple(plans), person.selected, person.scores))
    return Population(persons)
