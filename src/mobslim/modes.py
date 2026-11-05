"""Mode definitions: how a leg of each mode travels.

A :class:`Modes` object is the single place that answers "how does a leg of
this mode travel?". The :class:`~mobslim.scenario.Scenario` builds one
(``scenario.modes``), and the mobsim, replanning context and preparation all
use that same object, so the rules cannot drift apart.

Every mode is of one *kind* (:meth:`Modes.kind`), in this order of precedence:

* ``"transit"``: a passenger mode of the scenario's
  :class:`~mobslim.transit.Transit` (``TransitParams.passenger_modes``,
  ``("pt",)`` by default). Its legs ride scheduled vehicles and carry a
  :class:`~mobslim.population.TransitPassengerRoute`. It stays a transit mode even if some
  links list it (MATSim's pt-simple does).
* ``"network"``: at least one link allows it. Its legs drive the network.
* ``"teleported"``: listed in ``teleported``. Its legs travel at
  ``beeline * beeline_distance_factor / speed`` between the activities'
  coordinates.

Per-person speeds: ``person_max_speeds`` maps a mode to one maximum speed
[m/s] per person index (``NaN``: the mode's own). For a teleported mode
the person's maximum speed *is* their teleport speed. (Network modes are
not capped yet.)
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import numpy as np

from mobslim.network import Network
from mobslim.routing.teleport import TeleportedModeParams, teleport

if TYPE_CHECKING:
    from mobslim.transit import Transit

TRANSIT, NETWORK, TELEPORTED = "transit", "network", "teleported"


@dataclass(frozen=True, eq=False)
class Modes:
    network: Network
    teleported: Mapping[str, TeleportedModeParams] = field(default_factory=dict)
    transit: Optional["Transit"] = None
    person_max_speeds: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "teleported", dict(self.teleported))
        speeds = {}
        for mode, col in dict(self.person_max_speeds).items():
            col = np.asarray(col, dtype=float)
            col.flags.writeable = False
            speeds[mode] = col
        object.__setattr__(self, "person_max_speeds", speeds)

    def teleport_params(self, mode: str, person: Optional[int] = None) -> TeleportedModeParams:
        """The teleport parameters of ``mode`` for person index ``person``
        (their own maximum speed, if they have one)."""
        params = self.teleported[mode]
        if person is not None:
            col = self.person_max_speeds.get(mode)
            if col is not None:
                v = float(col[person])
                if v == v and v != params.speed:
                    return replace(params, speed=v)
        return params

    def speed_groups(self):
        """``(codes, representatives)``: persons with the same speeds in every
        mode share a code; ``representatives[c]`` is one person index of group
        ``c``. ``(None, None)`` if nobody has speeds of their own."""
        cached = self.__dict__.get("_speed_groups")
        if cached is None:
            if not self.person_max_speeds:
                cached = (None, None)
            else:
                cols = [np.where(np.isnan(c), self.teleported[m].speed if m in self.teleported else np.inf, c)
                        for m, c in self.person_max_speeds.items()]
                _, first, codes = np.unique(np.stack(cols, axis=1), axis=0, return_index=True,
                                            return_inverse=True)
                cached = (codes.reshape(-1).astype(np.int64), first.tolist())
            object.__setattr__(self, "_speed_groups", cached)
        return cached

    def teleport_speeds(self, mode: str, persons: np.ndarray) -> np.ndarray:
        """Teleport speed of ``mode`` for each of ``persons`` (person indices)."""
        default = self.teleported[mode].speed
        col = self.person_max_speeds.get(mode)
        if col is None:
            return np.full(len(persons), default)
        v = col[persons]
        return np.where(np.isnan(v), default, v)

    @property
    def transit_modes(self) -> frozenset[str]:
        if self.transit is None:
            return frozenset()
        return frozenset(self.transit.params.passenger_modes)

    @property
    def network_modes(self) -> frozenset[str]:
        """Modes that drive the network (link modes that are not transit modes)."""
        return self.network.modes - self.transit_modes

    def kind(self, mode: str) -> str:
        """``"transit"``, ``"network"`` or ``"teleported"`` (``KeyError`` if unknown)."""
        if mode in self.transit_modes:
            return TRANSIT
        if mode in self.network.modes:
            return NETWORK
        if mode in self.teleported:
            return TELEPORTED
        raise KeyError(f"Unknown mode {mode!r}.")

    def is_transit_mode(self, mode: str) -> bool:
        return mode in self.transit_modes

    def is_network_mode(self, mode: str) -> bool:
        return mode in self.network.modes and mode not in self.transit_modes

    def is_routed_mode(self, mode: str) -> bool:
        """Whether legs of ``mode`` need a route (network or transit)."""
        return mode in self.network.modes or mode in self.transit_modes

    def is_known(self, mode: str) -> bool:
        return mode in self.network.modes or mode in self.teleported or mode in self.transit_modes

    def teleport(
        self, mode: str, origin_xy, destination_xy, fixed_travel_time: Optional[float] = None,
        person: Optional[int] = None,
    ) -> tuple[float, float]:
        """``(travel_time, distance)`` for a teleported leg of ``mode`` between
        two points, by person index ``person`` (their own speed, if any)."""
        return teleport(origin_xy, destination_xy, self.teleport_params(mode, person),
                        fixed_travel_time)

    def validate(self, modes: Iterable[str]) -> None:
        """Raise ``ValueError`` if any of ``modes`` is of no kind."""
        unknown = {m for m in modes if not self.is_known(m)}
        if unknown:
            raise ValueError(
                f"Unknown leg modes {sorted(unknown)}: not on any network link, "
                f"not in teleported_modes {sorted(self.teleported)} and not a transit mode."
            )
