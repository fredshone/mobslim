"""Transit vehicle types and vehicles, as tables (MATSim defaults:
1 s access/egress per person, serial doors, 1 PCE, no speed limit)."""

from dataclasses import dataclass

import numpy as np

from mobslim.ids import IdMap

_ARRAYS = ("seats", "standing", "capacity", "length", "pce", "access_time", "egress_time",
           "door_parallel", "max_speed", "vehicle_type")


@dataclass(frozen=True, eq=False)
class TransitVehicles:
    type_ids: IdMap
    seats: np.ndarray
    standing: np.ndarray
    length: np.ndarray
    pce: np.ndarray
    access_time: np.ndarray
    egress_time: np.ndarray
    door_parallel: np.ndarray
    max_speed: np.ndarray
    vehicle_ids: IdMap
    vehicle_type: np.ndarray
    capacity: np.ndarray = None  # seats + standing (derived)

    def __post_init__(self):
        object.__setattr__(self, "capacity", (self.seats + self.standing).astype(np.int32))
        for name in _ARRAYS:
            getattr(self, name).flags.writeable = False


def build_vehicles(types: list[dict], vehicles: list[tuple]) -> TransitVehicles:
    type_ids, cols = IdMap(), {k: [] for k in ("seats", "standing", "length", "pce",
                                               "access_time", "egress_time", "door_parallel",
                                               "max_speed")}
    for t in types:
        if t["id"] in type_ids:
            raise ValueError(f"Duplicate vehicle type {t['id']!r}.")
        doors = t.get("doors", "serial")
        if doors not in ("serial", "parallel"):
            raise ValueError(f"Vehicle type {t['id']!r}: doors must be 'serial' or 'parallel'.")
        type_ids.add(t["id"])
        cols["seats"].append(int(t.get("seats", 1)))
        cols["standing"].append(int(t.get("standing", 0)))
        cols["length"].append(float(t.get("length", 7.5)))
        cols["pce"].append(float(t.get("pce", 1.0)))
        cols["access_time"].append(float(t.get("access_time", 1.0)))
        cols["egress_time"].append(float(t.get("egress_time", 1.0)))
        cols["door_parallel"].append(doors == "parallel")
        cols["max_speed"].append(float(t.get("max_speed", np.inf)))
    vehicle_ids, vehicle_type = IdMap(), []
    for vid, tid in vehicles:
        if vid in vehicle_ids:
            raise ValueError(f"Duplicate vehicle id {vid!r}.")
        if tid not in type_ids:
            raise ValueError(f"Vehicle {vid!r}: unknown vehicle type {tid!r}.")
        vehicle_ids.add(vid)
        vehicle_type.append(type_ids.index(tid))
    f = lambda k, dt: np.array(cols[k], dtype=dt)  # noqa: E731
    return TransitVehicles(type_ids, f("seats", np.int32), f("standing", np.int32),
                           f("length", float), f("pce", float), f("access_time", float),
                           f("egress_time", float), f("door_parallel", bool),
                           f("max_speed", float), vehicle_ids,
                           np.array(vehicle_type, dtype=np.int32))
