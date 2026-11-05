"""MATSim events -> mobslim :class:`~mobslim.experience.ExperiencedPlan` per person.

Lets mobslim's scorer score exactly what MATSim's agents did (layer 1).
MATSim's trips are mapped onto mobslim's model:

* Stage activities (``... interaction``) are dropped; the legs between two
  real activities form one trip.
* A trip containing a transit-mode leg becomes one ``pt`` leg with one
  :class:`ExperiencedRide` per transit leg. The wait runs from that leg's
  departure (the end of the stage activity before it, MATSim's
  ``lastActivityEndTime``) to boarding; the rest of the trip is walking.
* Any other trip is one leg of its main (non-walk) mode, or ``walk``.
* Network leg distance is the length of the links entered (MATSim's route
  distance: without the departure link, with the arrival link); teleported
  legs use the ``travelled`` distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mobslim.experience import (
    ExperiencedActivity,
    ExperiencedLeg,
    ExperiencedPlan,
    ExperiencedRide,
)

_WALKS = {"walk", "transit_walk", "non_network_walk", "access_walk", "egress_walk"}


def _is_stage(act_type: Optional[str]) -> bool:
    return act_type is not None and act_type.endswith(" interaction")


@dataclass
class _Leg:
    mode: str
    departure: float
    arrival: Optional[float] = None
    distance: float = 0.0
    board: Optional[float] = None
    alight: Optional[float] = None


@dataclass
class _Person:
    elements: list = field(default_factory=list)
    act: Optional[tuple] = None  # (type, start) of the open real activity
    trip: list = field(default_factory=list)  # legs of the current trip
    in_trip: bool = False
    stuck: bool = False


def experienced_plans(ev, link_length: dict, transit_modes=("pt",)) -> dict[str, ExperiencedPlan]:
    """One plan per person id, from an events table (:func:`events_io.read_events`)."""
    persons: dict[str, _Person] = {}
    driving: dict[str, str] = {}  # network vehicle -> person driving it
    transit_modes = set(transit_modes)

    def person(pid) -> _Person:
        p = persons.get(pid)
        if p is None:
            p = persons[pid] = _Person()
        return p

    cols = [ev[c].to_numpy() for c in ("time", "type", "person", "vehicle", "link", "legMode",
                                        "actType", "distance")]
    for t, typ, pid, veh, link, mode, act, dist in zip(*cols):
        if typ == "entered link":
            d = driving.get(veh)
            if d is not None:
                legs = persons[d].trip
                if legs:
                    legs[-1].distance += link_length.get(link, 0.0)
            continue
        if pid is None or (isinstance(pid, str) and pid.startswith("pt_")):
            continue  # transit drivers
        if typ == "actend":
            p = person(pid)
            if _is_stage(act):
                continue
            start = p.act[1] if p.act is not None else None
            p.elements.append(ExperiencedActivity(act, -1, start, t))
            p.act = None
            p.in_trip = True
            p.trip = []
        elif typ == "actstart":
            if _is_stage(act):
                continue
            p = person(pid)
            if p.in_trip:
                p.elements.append(_trip_leg(p.trip, transit_modes))
                p.in_trip = False
            p.act = (act, t)
        elif typ == "departure":
            person(pid).trip.append(_Leg(mode, t))
        elif typ == "arrival":
            person(pid).trip[-1].arrival = t
        elif typ == "travelled":
            person(pid).trip[-1].distance = float(dist)
        elif typ == "vehicle enters traffic":
            driving[veh] = pid
        elif typ == "vehicle leaves traffic":
            driving.pop(veh, None)
        elif typ == "PersonEntersVehicle":
            p = persons.get(pid)
            if p is not None and p.trip and p.trip[-1].mode in transit_modes:
                p.trip[-1].board = t
        elif typ == "PersonLeavesVehicle":
            p = persons.get(pid)
            if p is not None and p.trip and p.trip[-1].mode in transit_modes:
                p.trip[-1].alight = t
        elif typ == "stuckAndAbort":
            person(pid).stuck = True

    plans = {}
    for pid, p in persons.items():
        elements = list(p.elements)
        if p.in_trip and p.trip:  # stuck mid-trip: an unfinished leg
            elements.append(_trip_leg(p.trip, transit_modes, finished=False))
        elif p.act is not None:
            elements.append(ExperiencedActivity(p.act[0], -1, p.act[1], None))
        plans[pid] = ExperiencedPlan(tuple(elements), stuck=p.stuck)
    return plans


def _trip_leg(legs: list[_Leg], transit_modes: set, finished: bool = True) -> ExperiencedLeg:
    dep = legs[0].departure
    arr = legs[-1].arrival if finished else None
    dist = float(sum(leg.distance for leg in legs))
    rides = [leg for leg in legs if leg.mode in transit_modes]
    if rides:
        return ExperiencedLeg("pt", dep, arr, dist, (), rides=tuple(
            ExperiencedRide(-1, -1, -1, leg.departure, leg.board, leg.alight) for leg in rides))
    main = [leg.mode for leg in legs if leg.mode not in _WALKS]
    return ExperiencedLeg(main[0] if main else "walk", dep, arr, dist, ())


def score_plans(scorer, plans: dict[str, ExperiencedPlan]) -> dict[str, float]:
    return {pid: float(scorer.score(plan)) for pid, plan in plans.items()}


def as_array(scores: dict, ids) -> np.ndarray:
    return np.array([scores.get(i, np.nan) for i in ids], dtype=float)
