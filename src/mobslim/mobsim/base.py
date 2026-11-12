"""The mobsim interface.

A mobsim executes every person's *selected* plan on the network and returns
what happened as :class:`~mobslim.trajectories.Trajectories`: arrays aligned
with the executed plans (one value per activity, leg and route link). It may
also push events into an :class:`~mobslim.events.EventsManager` for any
handlers registered there, and must not mutate the scenario.

The controllers need two things from a run: the persons' link travel
times per time bin, to route on in the next iteration, and the plans as
simulated (activity and leg times, how far each leg got), to score. The
per-link entry times (``Trajectories.link_enter``) are needed for neither,
and are the largest output by far (one value per route link). So mobsims may
take two more keywords (flagged by ``sums_traversals = True``; see
:func:`run_mobsim`):

* ``traversal_bins=(bin_size, horizon)``: sum the link traversals per
  entry-time bin while simulating (``Trajectories.aggregate``), cheaper than
  summing them from the trajectories afterwards;
* ``link_times=False``: do not record the per-link entry times, only how
  many links each leg entered (``Trajectories.leg_entered``), which is what
  leg distances and experienced routes need.

The controllers ask for both, and for link times only in their output
iterations (the last, and every ``ControllerParams.link_times_interval``).
The built-in mobsims all take them; the compiled ones (``HermesSim``,
``PSim``, ``DSim``) act on them, the Python ones always record link times
and leave the summing to the trajectories.

That narrow contract (``Scenario`` in, arrays out) is the seam for swapping in
alternative implementations, e.g. a compiled (Rust) queue simulation or a
pseudo-simulation. A mobsim that only produces events can build its return
value with a :class:`~mobslim.trajectories.TrajectoryCollector`.
"""

from typing import Optional, Protocol, runtime_checkable

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


def run_mobsim(mobsim: Mobsim, scenario: Scenario, events: EventsManager,
               traversal_bins: Optional[tuple[float, float]] = None,
               link_times: bool = True) -> Trajectories:
    """``mobsim.run(scenario, events)``, with the requests of mobsims that
    take them (a true ``sums_traversals`` attribute; see the module notes):

    * ``traversal_bins=(bin_size, horizon)``: link traversal sums in bins of
      ``bin_size`` seconds up to ``horizon``. Mobsims that build them while
      simulating hand them over in :attr:`Trajectories.aggregate
      <mobslim.trajectories.Trajectories.aggregate>`; either way,
      :meth:`~mobslim.trajectories.Trajectories.traversal_aggregate` gives
      the same sums.
    * ``link_times=False``: the per-link entry times may be left out
      (``Trajectories.link_enter`` is then ``None``); everything else is the
      same.

    Other mobsims are run as ``mobsim.run(scenario, events)``.
    """
    if getattr(mobsim, "sums_traversals", False):
        return mobsim.run(scenario, events, traversal_bins=traversal_bins, link_times=link_times)
    return mobsim.run(scenario, events)
