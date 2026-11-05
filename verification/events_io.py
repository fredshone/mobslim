"""Read ``events_v1`` XML (MATSim's or mobslim's ``EventsXMLWriter``) into tables.

Both sides go through the same functions here, so every metric is computed
the same way for MATSim and mobslim.

Differences in what each side writes, and how the tables absorb them:

* MATSim emits ``vehicle enters traffic`` on the departure link (never
  ``entered link``) and ends a leg with ``vehicle leaves traffic``; mobslim
  starts at the departure link's to-node and ends with ``left link``. A
  traversal is an ``entered link`` closed by the vehicle's next ``left link``
  or ``vehicle leaves traffic`` on the same link, so both give the same
  links: every route link after the departure link.
* MATSim's link events carry only ``vehicle``; for car traffic in these
  scenarios the vehicle id is the person id on both sides.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

_COLUMNS = ("time", "type", "person", "vehicle", "link", "legMode", "actType", "facility",
            "distance", "vehicleId", "transitLineId", "transitRouteId", "departureId", "driverId")


def read_events(path) -> pd.DataFrame:
    """All events, in file order, with the attributes the metrics use."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = {c: [] for c in _COLUMNS}
    with opener(path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            a = el.attrib
            for c in _COLUMNS:
                rows[c].append(a.get(c))
            el.clear()
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype(float)
    return df


def transit_vehicles(ev: pd.DataFrame) -> set:
    return set(ev.loc[ev["type"] == "TransitDriverStarts", "vehicleId"])


def link_traversals(ev: pd.DataFrame, exclude_transit: bool = True) -> pd.DataFrame:
    """One row per link traversal: ``vehicle, link, enter, leave`` (``leave``
    is NaN if the vehicle never left, e.g. stuck or cut off). Transit
    vehicles are left out unless ``exclude_transit=False``."""
    sub = ev[ev["type"].isin(("entered link", "left link", "vehicle leaves traffic"))]
    if exclude_transit:
        sub = sub[~sub["vehicle"].isin(transit_vehicles(ev))]
    open_: dict = {}  # vehicle -> row index in out
    vehicle, link, enter, leave = [], [], [], []
    for t, typ, veh, lnk in zip(sub["time"].to_numpy(), sub["type"].to_numpy(),
                                sub["vehicle"].to_numpy(), sub["link"].to_numpy()):
        if typ == "entered link":
            open_[veh] = len(vehicle)
            vehicle.append(veh)
            link.append(lnk)
            enter.append(t)
            leave.append(np.nan)
        else:
            i = open_.get(veh)
            if i is not None and link[i] == lnk:
                leave[i] = t
                del open_[veh]
    return pd.DataFrame({"vehicle": vehicle, "link": link,
                         "enter": np.array(enter, dtype=float),
                         "leave": np.array(leave, dtype=float)})


def trips(ev: pd.DataFrame, transit_modes=("pt",)) -> pd.DataFrame:
    """One row per trip (door to door, between two real activities): ``person,
    k`` (trip index within the day), ``mode`` (the first non-walk leg mode, or
    ``walk``; a transit trip nobody boarded on is ``walk``, as in MATSim's
    mode statistics), ``departure, arrival`` (NaN if stuck or cut off).

    MATSim splits a pt trip into walk and pt legs joined by ``pt interaction``
    activities; mobslim writes one leg. Trips compare the same thing on both.
    Transit drivers are left out.
    """
    sub = ev[ev["type"].isin(("actend", "actstart", "departure", "arrival", "PersonEntersVehicle"))]
    drivers = set(ev.loc[ev["type"] == "TransitDriverStarts", "driverId"])
    sub = sub[~sub["person"].isin(drivers)]  # MATSim's transit drivers depart too
    transit_modes = set(transit_modes)
    rows: list = []
    boarded: list = []
    open_: dict = {}  # person -> row index of the open trip
    count: dict = {}
    for t, typ, p, act, mode in zip(sub["time"].to_numpy(), sub["type"].to_numpy(),
                                    sub["person"].to_numpy(), sub["actType"].to_numpy(),
                                    sub["legMode"].to_numpy()):
        stage = isinstance(act, str) and act.endswith(" interaction")
        if typ == "departure":
            i = open_.get(p)
            if i is None:
                k = count[p] = count.get(p, -1) + 1
                open_[p] = len(rows)
                rows.append([p, k, mode, t, np.nan])
                boarded.append(False)
            elif rows[i][2] in _WALK_MODES and mode not in _WALK_MODES:
                rows[i][2] = mode
        elif typ == "PersonEntersVehicle":
            i = open_.get(p)
            if i is not None:
                boarded[i] = True
        elif typ == "arrival":
            i = open_.get(p)
            if i is not None:
                rows[i][4] = t
        elif typ == "actstart" and not stage:
            open_.pop(p, None)
    for row, b in zip(rows, boarded):
        if row[2] in transit_modes and not b:
            row[2] = "walk"
    return pd.DataFrame(rows, columns=["person", "k", "mode", "departure", "arrival"])


_WALK_MODES = {"walk", "transit_walk", "non_network_walk"}


legs = trips  # (car-only scenarios: one leg per trip)


def activities(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per activity: ``person, k, type, start, end`` (``start`` NaN for
    the first activity, ``end`` NaN for the last)."""
    end = ev[ev["type"] == "actend"][["person", "time", "actType"]].copy()
    start = ev[ev["type"] == "actstart"][["person", "time"]].copy()
    end["k"] = end.groupby("person").cumcount()
    start["k"] = start.groupby("person").cumcount() + 1
    out = end.rename(columns={"time": "end", "actType": "type"}).merge(
        start.rename(columns={"time": "start"}), on=["person", "k"], how="outer")
    return out[["person", "k", "type", "start", "end"]].sort_values(["person", "k"])


def stuck_persons(ev: pd.DataFrame) -> set:
    return set(ev.loc[ev["type"] == "stuckAndAbort", "person"])


def transit_stops(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per stop served by a transit departure: ``line, route,
    departure, stop, visit`` (n-th call at that stop on the trip),
    ``arrival, departure_time, boardings, alightings``.

    Departures are told apart by ``TransitDriverStarts``; mobslim must write
    one vehicle id per trip (``EventsXMLWriter(vehicle_per_trip=True)``),
    as MATSim's vehicles serve one departure at a time.
    """
    trip_of = {}  # vehicle -> (line, route, departure), the current trip
    rows: list = []
    current: dict = {}  # vehicle -> row index of its current stop
    visits: dict = {}
    relevant = ev[ev["type"].isin(("TransitDriverStarts", "VehicleArrivesAtFacility",
                                   "VehicleDepartsAtFacility", "PersonEntersVehicle",
                                   "PersonLeavesVehicle"))]
    for t, typ, veh, fac, vid, line, route, dep, person in zip(
            relevant["time"].to_numpy(), relevant["type"].to_numpy(), relevant["vehicle"].to_numpy(),
            relevant["facility"].to_numpy(), relevant["vehicleId"].to_numpy(),
            relevant["transitLineId"].to_numpy(), relevant["transitRouteId"].to_numpy(),
            relevant["departureId"].to_numpy(), relevant["person"].to_numpy()):
        if typ == "TransitDriverStarts":
            trip_of[vid] = (line, route, dep)
            current.pop(vid, None)
        elif typ == "VehicleArrivesAtFacility":
            key = trip_of.get(veh)
            if key is None:
                continue
            n = visits[(key, fac)] = visits.get((key, fac), -1) + 1
            current[veh] = len(rows)
            rows.append([*key, fac, n, t, np.nan, 0, 0])
        elif typ == "VehicleDepartsAtFacility":
            i = current.get(veh)
            if i is not None and rows[i][3] == fac:
                rows[i][6] = t
        elif veh in current and not (isinstance(person, str) and person.startswith("pt_")):
            rows[current[veh]][7 if typ == "PersonEntersVehicle" else 8] += 1
    return pd.DataFrame(rows, columns=["line", "route", "departure", "stop", "visit", "arrival",
                                       "departure_time", "boardings", "alightings"])
