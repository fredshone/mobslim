"""Controller listeners: hooks into the iteration loop.

Subclass :class:`ControllerListener` and override any hook. Hooks receive the
mutable :class:`IterationState` for the current iteration, so a listener can
add event handlers (``before_mobsim``), inspect trajectories, experienced plans
and scores (``after_mobsim``, ``scoring``), or record metrics (``iteration_end``).
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from mobslim.events import EventLog, EventsManager, EventsXMLWriter
from mobslim.experience import ExperiencedPlan
from mobslim.routing.traveltime import TravelTimeTable
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories

if TYPE_CHECKING:
    from mobslim.controller import Controller

logger = logging.getLogger("mobslim")


@dataclass
class IterationState:
    """What is known about the current iteration, filled in as it progresses."""

    iteration: int
    is_last: bool
    scenario: Scenario
    events: Optional[EventsManager] = None  # set before the mobsim
    trajectories: Optional[Trajectories] = None  # set after the mobsim
    travel_time: Optional[TravelTimeTable] = None  # set at iteration end
    metrics: dict = field(default_factory=dict)  # set at iteration end

    @property
    def experienced(self) -> Optional[list[ExperiencedPlan]]:
        """Experienced plans, one per person, built from the trajectories on
        first access (nothing builds them unless asked)."""
        if self.trajectories is None:
            return None
        return self.trajectories.experienced_plans()


class ControllerListener:
    """No-op base class. Hooks fire in the order listed."""

    def on_startup(self, controller: "Controller") -> None: ...
    def on_iteration_start(self, state: IterationState) -> None: ...
    def on_after_replanning(self, state: IterationState) -> None: ...
    def on_before_mobsim(self, state: IterationState) -> None: ...
    def on_after_mobsim(self, state: IterationState) -> None: ...
    def on_scoring(self, state: IterationState) -> None: ...
    def on_iteration_end(self, state: IterationState) -> None: ...
    def on_shutdown(self, controller: "Controller") -> None: ...


class ProgressLogger(ControllerListener):
    """Logs one line per iteration to the ``mobslim`` logger."""

    def on_iteration_end(self, state: IterationState) -> None:
        m = state.metrics
        legs = ", ".join(
            f"{k[:-12]} tt={m[k]:.0f}s" for k in m if k.endswith("_travel_time")
        )
        logger.info(
            "it %d: score executed=%.3f best=%.3f stuck=%d %s",
            m["iteration"], m["score_executed"], m["score_best"], m["stuck"], legs,
        )


class MetricsCSVWriter(ControllerListener):
    """Writes ``metrics.csv`` (one row per iteration), rewritten each iteration."""

    def __init__(self, output_dir):
        self.path = Path(output_dir) / "metrics.csv"
        self.rows: list[dict] = []

    def on_iteration_end(self, state: IterationState) -> None:
        self.rows.append(state.metrics)
        columns = list(dict.fromkeys(k for row in self.rows for k in row))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)


class EventsXMLListener(ControllerListener):
    """Writes ``ITERS/it.N/events.xml.gz`` every ``interval`` iterations and on the last."""

    def __init__(self, output_dir, interval: int):
        self.output_dir = Path(output_dir)
        self.interval = interval

    def on_before_mobsim(self, state: IterationState) -> None:
        if state.is_last or (self.interval > 0 and state.iteration % self.interval == 0):
            path = self.output_dir / "ITERS" / f"it.{state.iteration}" / "events.xml.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            state.events.add_handler(
                EventsXMLWriter(path, state.scenario.network, state.scenario.population,
                                state.scenario.transit)
            )


class EventRecorder(ControllerListener):
    """Keeps the in-memory events of selected iterations (default: the last).

    ``recorder.events[iteration]`` is a list of events.
    """

    def __init__(self, iterations: Optional[Iterable[int]] = None):
        self.iterations = None if iterations is None else set(iterations)
        self.events: dict[int, list] = {}
        self._log: Optional[EventLog] = None

    def on_before_mobsim(self, state: IterationState) -> None:
        wanted = state.is_last if self.iterations is None else state.iteration in self.iterations
        self._log = state.events.add_handler(EventLog()) if wanted else None

    def on_after_mobsim(self, state: IterationState) -> None:
        if self._log is not None:
            self.events[state.iteration] = self._log.events
            self._log = None

    @property
    def last(self) -> list:
        return self.events[max(self.events)]
