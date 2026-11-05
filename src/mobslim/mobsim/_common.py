"""Helpers shared by the Python mobsims (``QSim``, ``AgentSim``).

Keeping the link parameters, route validation and event dispatch in one place
means the two simulations cannot drift apart on the queue model itself, only
on how they schedule it.
"""

import numpy as np

from mobslim.mobsim.base import SimulationError
from mobslim.network import Network
from mobslim.planbatch import PlanBatch
from mobslim.scenario import Scenario

EPS = 1e-9


def link_parameters(
    net: Network,
    flow_capacity_factor: float,
    storage_capacity_factor: float,
    effective_cell_size: float,
) -> tuple[list[float], list[float], list[float]]:
    """``(free_time, headway, storage)`` per link, as Python lists.

    * free-speed time: ``length / freespeed`` [s]
    * headway: ``1 / (capacity * flow_capacity_factor)`` [s/veh], capacity per
      link (not per lane) in veh/h, fractional (no rounding)
    * storage: ``length * lanes / effective_cell_size * storage_capacity_factor``
      [veh], at least enough not to restrict the link's own flow, and at least 1
    """
    cap_per_s = net.capacity / 3600.0 * flow_capacity_factor
    free_time = net.free_speed_travel_times()
    storage = net.length * net.lanes / effective_cell_size * storage_capacity_factor
    storage = np.maximum(np.maximum(storage, free_time * cap_per_s), 1.0)
    return free_time.tolist(), (1.0 / cap_per_s).tolist(), storage.tolist()


def validate(scenario: Scenario, plans: PlanBatch) -> None:
    """Raise :class:`SimulationError` for unknown modes or unexecutable routes."""
    ids = scenario.population.ids
    is_net = plans.network_legs(scenario.modes.network_modes)
    for code, mode in enumerate(plans.modes):
        if not scenario.modes.is_known(mode):
            leg = int(np.flatnonzero(plans.leg_mode == code)[0])
            person = ids.id(int(plans.plan_of_leg()[leg]))
            raise SimulationError(f"Person {person!r}: unknown mode '{mode}'.")
    bad = plans.invalid_legs(scenario.network, scenario.modes.network_modes, scenario.transit)
    if len(bad):
        leg = int(bad[0])
        person = ids.id(int(plans.plan_of_leg()[leg]))
        mode = plans.modes[plans.leg_mode[leg]]
        if is_net[leg] and not plans.leg_routed[leg]:
            raise SimulationError(
                f"Person {person!r}: '{mode}' leg has no route. "
                "Route plans before simulating (Controller.prepare does this)."
            )
        links = tuple(plans.route_links[plans.route_offsets[leg]:plans.route_offsets[leg + 1]].tolist())
        raise SimulationError(f"Person {person!r}: invalid '{mode}' route {links}.")


def none_if_nan(values: np.ndarray) -> list:
    return [None if v != v else v for v in values.tolist()]


def planned_end(end_time, duration, start: float):
    """MATSim's ``tryEndTimeThenDuration`` (see ``Activity.planned_end``)."""
    if end_time is not None:
        return start if start > end_time else end_time
    if duration is not None:
        return start + duration
    return None


def fire(handlers: list, event) -> None:
    for fn in handlers:
        fn(event)
