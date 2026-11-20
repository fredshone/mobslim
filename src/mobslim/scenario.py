"""The scenario: everything a mobsim needs to run one iteration."""

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Mapping, Optional

from mobslim.modes import Modes
from mobslim.network import Network
from mobslim.population import Population
from mobslim.routing.teleport import (
    DEFAULT_TELEPORTED_MODES,
    TeleportedModeParams,
)

if TYPE_CHECKING:
    from mobslim.transit import Transit


@dataclass(frozen=True)
class Scenario:
    """Network + population + mode definitions (+ scheduled transit).

    A leg mode is a *transit mode* if ``transit`` lists it as a passenger
    mode, else a *network mode* if some link allows it; otherwise it must be
    listed in ``teleported_modes``. ``scenario.modes`` answers mode questions
    (which kind, teleport times) for every component.
    """

    network: Network
    population: Population
    teleported_modes: Mapping[str, TeleportedModeParams] = field(
        default_factory=lambda: dict(DEFAULT_TELEPORTED_MODES)
    )
    transit: Optional["Transit"] = None
    # mode -> one maximum speed [m/s] per person index (NaN: the mode's); see Modes
    person_max_speeds: Mapping[str, Any] = field(default_factory=dict)
    # network modes simulated on the network (MATSim's qsim.mainMode; None:
    # all); the others are teleported along their routes (see Modes)
    simulated_modes: Optional[frozenset] = None
    # network mode -> passenger car equivalents of its vehicles (default 1)
    mode_pce: Mapping[str, float] = field(default_factory=dict)
    # network mode -> PCE of each person's own vehicle, per person index (NaN: the mode's)
    person_pce: Mapping[str, Any] = field(default_factory=dict, repr=False)
    # each person's initial activity end times (a replanning.context.InitialEndTimes),
    # set by the controllers when a strategy mutates around them; None: not kept
    initial_end_times: Optional[Any] = field(default=None, repr=False)

    @cached_property
    def modes(self) -> Modes:
        return Modes(self.network, self.teleported_modes, self.transit, self.person_max_speeds,
                     self.simulated_modes, self.mode_pce)

    def with_population(self, population: Population) -> "Scenario":
        return replace(self, population=population)

    def validate_modes(self) -> None:
        """Raise if any leg uses a mode that is neither networked nor teleported."""
        self.modes.validate(self.population.live_batch().modes)
