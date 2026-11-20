"""Least-cost routing as MATSim does it: travel time plus distance.

MATSim's routers do not minimise travel time but a *travel disutility*,
``RandomizingTimeDistanceTravelDisutility``, from the scoring parameters of
the leg's mode:

* time costs ``performing - marginal_utility_of_traveling`` per hour (the
  time not spent at an activity, plus travelling's own disutility);
* distance costs ``-(marginal_utility_of_distance + monetary_distance_rate *
  marginal_utility_of_money)`` per metre;
* the distance cost is multiplied by a random factor per person and
  routing (``routing.routingRandomness``, sigma, 3 by default): lognormal
  with mean 1, so persons spread over routes of different lengths.

:class:`TravelDisutility` holds those costs per mode. A router factory takes
it (``functools.partial(default_router, disutility=...)``); the routers then
minimise ``time_cost * travel time + distance_cost * factor * length``, and
callers that know the persons pass their factors (:meth:`TravelDisutility.factors`,
drawn per person and iteration, the same for every leg of a person in an
iteration).

MATSim's A* (``SpeedyALT``) bounds the remaining cost with the distance cost
at factor 1 (``getLinkMinimumTravelDisutility``), not the person's. Most
factors are far below 1 (their median is ``exp(-sigma^2 / 2)``, 0.011 at
sigma 3), so the bound overestimates and MATSim's routes are not least-cost:
on Open Berlin they are slower and longer than MobSlim's for a third of car
legs. ``bound_factor=1`` reproduces that (ALT only; Dijkstra stays exact),
for comparisons that need MATSim's routes rather than least-cost ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

HOUR = 3600.0


@dataclass(frozen=True)
class TravelDisutility:
    """Link cost per mode: ``time_cost[mode]`` per second of travel time plus
    ``distance_cost[mode] * factor`` per metre (both >= 0). A mode not listed
    costs its travel time. ``randomness``: sigma of the per-person lognormal
    distance factor (0: always 1). ``bound_factor``: the factor ALT's lower
    bound assumes, as MATSim's A* does with 1 (``None``, the default: each
    person's own, so routes are least-cost)."""

    time_cost: Mapping[str, float] = field(default_factory=dict)
    distance_cost: Mapping[str, float] = field(default_factory=dict)
    randomness: float = 0.0
    seed: int = 0
    bound_factor: Optional[float] = None

    @classmethod
    def from_scoring(cls, params, modes, randomness: float = 3.0, seed: int = 0,
                     bound_factor: Optional[float] = None) -> "TravelDisutility":
        """MATSim's costs for ``modes`` from ``params`` (a
        :class:`~mobslim.scoring.ScoringParams`): MATSim takes them from the
        scoring parameters without a subpopulation."""
        time, distance = {}, {}
        for mode in modes:
            m = params.mode(mode)
            time[mode] = max(0.0, (params.performing - m.marginal_utility_of_traveling) / HOUR)
            distance[mode] = max(0.0, -(m.marginal_utility_of_distance
                                        + m.monetary_distance_rate * params.marginal_utility_of_money))
        return cls(time, distance, randomness, seed, bound_factor)

    def costs(self, mode: str) -> tuple[float, float]:
        """``(time cost per s, distance cost per m)`` of ``mode``; travel time
        alone (``(1, 0)``) if neither costs anything."""
        a, b = self.time_cost.get(mode, 1.0), self.distance_cost.get(mode, 0.0)
        return (a, b) if a > 0 or b > 0 else (1.0, 0.0)

    def factors(self, persons, iteration: int) -> Optional[np.ndarray]:
        """Each person's distance factor in ``iteration``: ``exp(sigma * z -
        sigma^2 / 2)`` (mean 1, as MATSim normalises it), ``z`` a standard
        normal drawn from ``(seed, iteration, person)``; ``None`` without
        randomness or persons."""
        if self.randomness <= 0 or persons is None:
            return None
        p = np.asarray(persons, dtype=np.uint64)
        with np.errstate(over="ignore"):  # wrapping uint64 arithmetic
            run = np.uint64(self.seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(iteration & 0xFFFFFFFF)
        key = _mix(run) ^ p
        u1 = (_mix(key) >> np.uint64(11)).astype(np.float64) * 2.0**-53
        u2 = (_mix(key ^ np.uint64(0xD1B54A32D192ED03)) >> np.uint64(11)).astype(np.float64) * 2.0**-53
        z = np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)  # Box-Muller
        s = self.randomness
        return np.exp(s * z - s * s / 2.0)


def _mix(x):
    """SplitMix64's finaliser (wrapping uint64 arithmetic)."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def router_factors(router, persons, iteration: int) -> Optional[np.ndarray]:
    """The distance factors ``router`` wants for ``persons`` (``None``: it
    has no random distance cost, or routes without persons)."""
    d = getattr(router, "disutility", None)
    if d is None or persons is None or router_costs(router)[1] <= 0:
        return None
    return d.factors(persons, iteration)


def router_costs(router) -> tuple[float, float]:
    """``(time cost, distance cost)`` a router minimises (travel time: ``(1, 0)``)."""
    return getattr(router, "costs", (1.0, 0.0))
