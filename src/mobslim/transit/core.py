"""``Transit``: a schedule, its vehicles and how to simulate them."""

from dataclasses import dataclass, field

import numpy as np

from mobslim.routing.teleport import TeleportedModeParams
from mobslim.transit.schedule import TransitSchedule
from mobslim.transit.vehicles import TransitVehicles

VEHICLE_MODES = ("schedule", "network")


@dataclass(frozen=True)
class TransitParams:
    """How transit is simulated: passenger modes, vehicles on the timetable
    (``"schedule"``) or in the queue network (``"network"``), access/egress walks,
    door times, and whether transit vehicles feed the learned link travel times."""

    passenger_modes: tuple[str, ...] = ("pt",)
    vehicles: str = "schedule"  # or "network"
    walk: TeleportedModeParams = TeleportedModeParams(3 / 3.6, 1.3)
    door_open_time: float = 1.0
    door_close_time: float = 1.0
    learn_from_transit: bool = False

    def __post_init__(self):
        if self.vehicles not in VEHICLE_MODES:
            raise ValueError(f"TransitParams.vehicles must be one of {VEHICLE_MODES}.")
        object.__setattr__(self, "passenger_modes", tuple(self.passenger_modes))


@dataclass(frozen=True, eq=False)
class Transit:
    """Schedule + vehicles + params. ``dep_type`` is each trip's vehicle type."""

    schedule: TransitSchedule
    vehicles: TransitVehicles
    params: TransitParams = TransitParams()
    dep_type: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        s, v = self.schedule, self.vehicles
        index = np.empty(len(s.vehicle_ids), dtype=np.int32)
        for k, vid in enumerate(s.vehicle_ids):
            j = v.vehicle_ids.get(vid)
            if j is None:
                raise ValueError(f"Transit schedule refers to unknown vehicle {vid!r}.")
            index[k] = j
        dep_type = np.asarray(v.vehicle_type[index[s.dep_vehicle]] if s.n_departures
                              else np.zeros(0), dtype=np.int32)
        dep_type.flags.writeable = False
        object.__setattr__(self, "dep_type", dep_type)

    @property
    def trip_capacity(self) -> np.ndarray:
        return self.vehicles.capacity[self.dep_type]

    def vehicle_id(self, trip: int):
        return self.schedule.vehicle_ids.id(int(self.schedule.dep_vehicle[trip]))

    def with_params(self, params: TransitParams) -> "Transit":
        return Transit(self.schedule, self.vehicles, params)
