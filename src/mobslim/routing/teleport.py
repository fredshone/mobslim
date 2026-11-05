"""Teleported (non-network) modes, e.g. walk or bike.

A teleported leg is not simulated on the network. Its travel time is
``beeline_distance * beeline_distance_factor / speed`` (or the leg's fixed
``travel_time`` if one is given), as in MATSim's teleportation. The beeline
runs between the two activities' coordinates (``Activity.x, y``, or their
nodes' where they have none), so it is as long as in MATSim.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TeleportedModeParams:
    speed: float  # m/s
    beeline_distance_factor: float = 1.3


DEFAULT_TELEPORTED_MODES = {
    "walk": TeleportedModeParams(speed=3 / 3.6),
    "bike": TeleportedModeParams(speed=15 / 3.6),
}


def teleport(
    origin_xy,
    destination_xy,
    params: TeleportedModeParams,
    fixed_travel_time: Optional[float] = None,
) -> tuple[float, float]:
    """Return ``(travel_time, distance)`` for a teleported leg between two points."""
    (x0, y0), (x1, y1) = origin_xy, destination_xy
    distance = math.hypot(x1 - x0, y1 - y0) * params.beeline_distance_factor
    if fixed_travel_time is not None:
        return fixed_travel_time, distance
    return distance / params.speed, distance


def teleport_many(origin_xy: np.ndarray, destination_xy: np.ndarray,
                  params: TeleportedModeParams) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`teleport` (no fixed times): ``(travel_times, distances)``."""
    d = origin_xy - destination_xy
    distance = np.hypot(d[:, 0], d[:, 1]) * params.beeline_distance_factor
    return distance / params.speed, distance
