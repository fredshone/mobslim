"""The mobsim interface.

A mobsim executes every person's *selected* plan on the network and returns
what happened as :class:`~mobslim.trajectories.Trajectories`: arrays aligned
with the executed plans (one value per activity, leg and route link). It may
also push events into an :class:`~mobslim.events.EventsManager` for any
handlers registered there, and must not mutate the scenario.

That narrow contract (``Scenario`` in, arrays out) is the seam for swapping in
alternative implementations, e.g. a compiled (Rust) queue simulation or a
pseudo-simulation. A mobsim that only produces events can build its return
value with a :class:`~mobslim.trajectories.TrajectoryCollector`.
"""

from typing import Protocol, runtime_checkable

from mobslim.events import EventsManager
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories


class SimulationError(RuntimeError):
    """Raised when a plan cannot be executed (e.g. a network leg without a route)."""


@runtime_checkable
class Mobsim(Protocol):
    def run(self, scenario: Scenario, events: EventsManager) -> Trajectories:
        """Simulate all selected plans; return their trajectories and push
        events to any registered handlers."""
        ...
