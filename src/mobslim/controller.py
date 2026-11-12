"""The controller: MATSim's co-evolutionary loop.

Each iteration::

    iteration_start
    replanning           (from iteration 1)   -> after_replanning
    mobsim               (-> trajectories, events -> handlers) -> before_mobsim / after_mobsim
    scoring              (trajectories -> scores) -> scoring
    travel-time learning, metrics (from trajectories) -> iteration_end

Every stage is a pluggable component behind a small interface:

=================  =======================================  =====================
stage              interface                                default
=================  =======================================  =====================
mobsim             :class:`~mobslim.mobsim.base.Mobsim`     :class:`QSim`
scoring            :class:`~mobslim.scoring.Scorer`         Charypar-Nagel
replanning         :class:`StrategyManager` + strategies    MATSim-like mix
replanning runner  :class:`ReplanningRunner`                :class:`VectorRunner`
routing            router factory                           :func:`default_router` (Rust if built)
transit routing    transit router factory                   :class:`~mobslim.transit.RaptorRouter`
=================  =======================================  =====================

Unless a ``transit_router_factory`` is given, the transit router uses the
scorer's utilities (``scorer.params``, the default parameter set), as MATSim's
SwissRailRaptor uses the scoring config: see
:meth:`~mobslim.replanning.context.TransitRouterFactory.from_scoring`.

Person-specific transit weights: ``transit_weights(attributes)`` returns one
:class:`~mobslim.transit.RaptorWeights` (or ``None``: the router's own) per
person, computed once per run from the person attributes. Without it, the
scorer's ``transit_weights`` is used if it has one
(``CharyparNagelScorer``: each subpopulation's scoring params, as MATSim's
``DefaultRaptorParametersForPerson``).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

from mobslim.activity_links import ActivityLinkParams
from mobslim.attributes import PersonAttributes
from mobslim.events import EventsManager
from mobslim.listeners import (
    ControllerListener,
    EventsXMLListener,
    IterationState,
    MetricsCSVWriter,
    ProgressLogger,
)
from mobslim.metrics import iteration_metrics
from mobslim.mobsim.base import Mobsim, run_mobsim
from mobslim.mobsim.qsim import QSim
from mobslim.replanning.context import (
    ReplanningContext,
    TransitRouterFactory,
    default_transit_router,
)
from mobslim.replanning.prepare import route_scenario
from mobslim.replanning.runners import ReplanningRunner
from mobslim.replanning.strategy import StrategyManager
from mobslim.replanning.vectorised import VectorRunner
from mobslim.routing.base import RouterFactory
from mobslim.routing.compiled import default_router
from mobslim.routing.traveltime import (
    DEFAULT_HORIZON,
    DEFAULT_OFF_PEAK_BIN_SIZE,
    DEFAULT_PEAK,
    Bins,
    TimeBins,
    TravelTimeTable,
)
from mobslim.scenario import Scenario
from mobslim.scoring import CharyparNagelScorer, Scorer, apply_scores, score_all

logger = logging.getLogger("mobslim")


@dataclass(frozen=True)
class ControllerParams:
    """Settings of the iteration loop itself (not of its components)."""

    iterations: int = 10  # number of mobsim runs
    seed: int = 4711  # MATSim's default random seed
    learning_rate: float = 1.0  # score averaging: new = (1-lr) * old + lr * observed
    travel_time_bin_size: float = 900.0  # seconds per learned travel-time bin (in the peak)
    travel_time_horizon: float = DEFAULT_HORIZON  # seconds covered by the travel-time table
    # bins outside travel_time_peak (a multiple of travel_time_bin_size; equal
    # to it: uniform bins, as MATSim); see travel_time_bins
    travel_time_off_peak_bin_size: float = DEFAULT_OFF_PEAK_BIN_SIZE
    travel_time_peak: tuple[float, float] = DEFAULT_PEAK
    output_dir: Optional[Union[str, Path]] = None  # None: write no files
    write_events_interval: int = 0  # with output_dir: every N iterations (0 = last only)
    # record the persons' link entry times (Trajectories.link_enter) every N
    # iterations (0 = the last only); other iterations record how far each
    # leg got, which is all scoring and the travel times need
    link_times_interval: int = 0
    # how activities without a link are assigned one before the first
    # iteration (None: they keep reaching their node by any link)
    activity_links: Optional[ActivityLinkParams] = ActivityLinkParams()

    def travel_time_bins(self) -> Bins:
        """The learned travel times' bins: ``travel_time_bin_size`` seconds in
        ``travel_time_peak``, ``travel_time_off_peak_bin_size`` before and
        after (a :class:`~mobslim.routing.traveltime.TimeBins`), or uniform
        bins of ``travel_time_bin_size`` (a number) when the two are equal."""
        if self.travel_time_off_peak_bin_size == self.travel_time_bin_size:
            return self.travel_time_bin_size
        return TimeBins.off_peak(self.travel_time_bin_size, self.travel_time_off_peak_bin_size,
                                 tuple(self.travel_time_peak), self.travel_time_horizon)

    def records_link_times(self, iteration: int) -> bool:
        """Whether ``iteration`` records the persons' link entry times: the
        last, and every ``link_times_interval`` (as events are written)."""
        n = self.link_times_interval
        return iteration == self.iterations - 1 or (n > 0 and iteration % n == 0)

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1.")
        self.travel_time_bins()  # checks the bins
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")


@dataclass
class RunResult:
    """What :meth:`Controller.run` returns."""

    scenario: Scenario  # final population with scored plans
    travel_time: TravelTimeTable  # learned in the last iteration
    metrics: list[dict]  # one row per iteration (see mobslim.metrics.iteration_metrics)

    def metrics_frame(self):
        """The metrics as a pandas DataFrame indexed by iteration."""
        import pandas as pd

        return pd.DataFrame(self.metrics).set_index("iteration")


class Controller:
    """Runs MATSim's co-evolutionary loop on a scenario.

    Every component is optional and has a MATSim-like default (see the module
    docstring). ``Controller(scenario).run()`` routes un-routed legs, then
    repeats replan, simulate, score and learn travel times for
    ``params.iterations`` iterations.

    Args:
        scenario: network, population, mode definitions and transit.
        mobsim: executes the selected plans (default :class:`QSim`).
        scorer: scores what happened (default :class:`CharyparNagelScorer`).
        strategies: replanning strategies (default
            :meth:`StrategyManager.default`).
        runner: how replanning is executed (default :class:`VectorRunner`).
        router_factory: ``(network, mode, travel_time) -> Router`` for
            network modes (default: the Rust router if built, else Dijkstra).
        transit_router_factory: ``(network, transit) -> transit router``
            (default: RAPTOR with the scorer's pt utilities).
        transit_weights: ``(attributes) -> one RaptorWeights or None per
            person``: person-specific transit routing (default: from the
            scorer, if it has ``transit_weights``).
        listeners: :class:`ControllerListener` objects, hooked into each
            stage of the loop.
        params: loop settings (iterations, seed, output directory...).
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        mobsim: Optional[Mobsim] = None,
        scorer: Optional[Scorer] = None,
        strategies: Optional[StrategyManager] = None,
        runner: Optional[ReplanningRunner] = None,
        router_factory: RouterFactory = default_router,
        transit_router_factory: Optional[Callable] = None,
        transit_weights: Optional[Callable[[PersonAttributes], Optional[Sequence]]] = None,
        listeners: Iterable[ControllerListener] = (),
        params: ControllerParams = ControllerParams(),
    ):
        self.scenario = scenario
        self.mobsim = mobsim or QSim()
        self.scorer = scorer or CharyparNagelScorer()
        self.strategies = strategies or StrategyManager.default()
        self.runner = runner or VectorRunner()
        self.router_factory = router_factory
        if transit_router_factory is None:
            transit_router_factory = self._transit_router_from_scoring(scenario, self.scorer)
        self.transit_router_factory = transit_router_factory
        self.transit_weights = transit_weights
        self.params = params
        self.listeners: list[ControllerListener] = [ProgressLogger()]
        if params.output_dir is not None:
            self.listeners.append(MetricsCSVWriter(params.output_dir))
            self.listeners.append(EventsXMLListener(params.output_dir, params.write_events_interval))
        self.listeners.extend(listeners)
        self.metrics: list[dict] = []

    @classmethod
    def _from_kwargs(cls, kwargs: dict, overrides: dict) -> "Controller":
        """``overrides`` replace any keyword argument, except ``listeners``,
        which are added to the config's own."""
        listeners = [*kwargs.pop("listeners", ()), *overrides.pop("listeners", ())]
        kwargs.update(overrides)
        return cls(kwargs.pop("scenario"), listeners=listeners, **kwargs)

    @classmethod
    def from_matsim_config(cls, path, *, strict: bool = False, **overrides) -> "Controller":
        """A controller for a MATSim ``config.xml`` (a path or a
        :class:`~mobslim.config.MatsimConfig`).

        Reads the network, plans and transit it names (paths relative to the
        config file) and translates the ``controller``, ``qsim``/``hermes``,
        ``scoring``, ``replanning`` and ``routing`` modules (see
        :mod:`mobslim.config.matsim`). Params mobslim cannot honour are
        logged as a warning, or raise :class:`UnsupportedConfigError` if
        ``strict``. ``overrides`` replace any ``Controller`` argument, e.g.
        ``params=ControllerParams(iterations=5)``; ``listeners`` are added.
        """
        from mobslim.config.matsim import (
            MatsimConfig,
            UnsupportedConfigError,
            controller_kwargs,
        )

        cfg = path if isinstance(path, MatsimConfig) else MatsimConfig.load(path)
        kwargs, unsupported = controller_kwargs(cfg)
        if unsupported:
            if strict:
                raise UnsupportedConfigError(unsupported)
            logger.warning("MATSim config params mobslim cannot honour (ignored):\n  %s",
                           "\n  ".join(unsupported))
        return cls._from_kwargs(kwargs, overrides)

    @classmethod
    def from_mobslim_config(cls, path, **overrides) -> "Controller":
        """A controller for a mobslim YAML config (a path or a
        :class:`~mobslim.config.MobslimConfig`; see :mod:`mobslim.config.mobslim`).

        Loads and checks it, reads the inputs, prechecks them against it
        (raising one ``ConfigError`` that lists every problem), and writes
        ``output_config.yaml`` at startup. ``overrides`` replace any
        ``Controller`` argument, e.g. ``params=ControllerParams(iterations=5)``;
        ``listeners`` are added to the config's own."""
        from mobslim.config.mobslim import MobslimConfig

        cfg = path if isinstance(path, MobslimConfig) else MobslimConfig.load(path)
        return cls._from_kwargs(cfg.controller_kwargs(), overrides)

    @staticmethod
    def _transit_router_from_scoring(scenario: Scenario, scorer):
        """The scorer's pt utilities (for the first transit passenger mode),
        or RaptorParams's defaults if the scorer has no ``ScoringParams``."""
        from mobslim.scoring import ScoringParams

        params = getattr(scorer, "params", None)
        if not isinstance(params, ScoringParams):
            return default_transit_router
        transit = scenario.transit
        mode = transit.params.passenger_modes[0] if transit is not None else "pt"
        return TransitRouterFactory.from_scoring(params, mode)

    def person_transit_weights(self, scenario: Scenario):
        """Each person's transit weights (a
        :class:`~mobslim.transit.PersonRaptorWeights`), or ``None`` if nobody
        has weights of their own (or there is no transit)."""
        from mobslim.transit.raptor import PersonRaptorWeights

        if scenario.transit is None:
            return None
        attributes = scenario.population.attributes
        if self.transit_weights is not None:
            weights = self.transit_weights(attributes)
        else:
            from_scorer = getattr(self.scorer, "transit_weights", None)
            if from_scorer is None:
                return None
            weights = from_scorer(attributes, scenario.transit.params.passenger_modes[0])
        if weights is None:
            return None
        if not isinstance(weights, PersonRaptorWeights):
            weights = PersonRaptorWeights.from_sequence(weights)
        if len(weights) != len(scenario.population):
            raise ValueError(f"transit_weights gave {len(weights)} weights for "
                             f"{len(scenario.population)} persons.")
        return weights if weights.any else None

    def add_listener(self, listener: ControllerListener) -> ControllerListener:
        """Add a listener after construction; returns it."""
        self.listeners.append(listener)
        return listener

    def _notify(self, hook: str, arg) -> None:
        for listener in self.listeners:
            getattr(listener, hook)(arg)

    def free_speed_travel_time(self) -> TravelTimeTable:
        c = self.params
        return TravelTimeTable.free_speed(self.scenario.network, c.travel_time_bins(),
                                          c.travel_time_horizon)

    def prepare(self, scenario: Scenario) -> Scenario:
        """Validate modes, assign activities to links (``params.activity_links``)
        and route any un-routed network (free speed) and pt legs."""
        return route_scenario(scenario, self.free_speed_travel_time(), self.router_factory,
                              self.transit_router_factory, self.person_transit_weights(scenario),
                              self.params.activity_links)

    def run(self) -> RunResult:
        """Run every iteration and return the final scenario, the last
        learned travel times and the per-iteration metrics."""
        cfg = self.params
        scenario = self.prepare(self.scenario)
        transit_weights = self.person_transit_weights(scenario)  # attributes never change
        travel_time = self.free_speed_travel_time()
        self.metrics = []
        self._notify("on_startup", self)
        try:
            for it in range(cfg.iterations):
                state = IterationState(it, it == cfg.iterations - 1, scenario)
                self._notify("on_iteration_start", state)

                if it > 0:
                    ctx = ReplanningContext.from_scenario(scenario, travel_time, it, self.router_factory,
                                                          self.transit_router_factory, transit_weights)
                    population = self.runner.run(scenario.population, self.strategies, ctx, cfg.seed)
                    scenario = state.scenario = scenario.with_population(population)
                    self._notify("on_after_replanning", state)

                # listeners may add event handlers; without any, no events are built
                events = state.events = EventsManager()
                self._notify("on_before_mobsim", state)
                events.reset(it)
                trajectories = state.trajectories = run_mobsim(
                    self.mobsim, scenario, events,
                    (cfg.travel_time_bins(), cfg.travel_time_horizon),
                    link_times=cfg.records_link_times(it))
                events.finish()
                self._notify("on_after_mobsim", state)

                scores = score_all(self.scorer, trajectories, scenario.population.attributes)
                scenario = state.scenario = scenario.with_population(
                    apply_scores(scenario.population, scores, cfg.learning_rate)
                )
                self._notify("on_scoring", state)

                travel_time = state.travel_time = trajectories.travel_time_table(
                    scenario.network, cfg.travel_time_bins(), cfg.travel_time_horizon
                )
                state.metrics = iteration_metrics(it, scenario.population, trajectories)
                self.metrics.append(state.metrics)
                self._notify("on_iteration_end", state)
                del trajectories  # free before the next iteration (listeners keep ``state``)
        finally:
            self.runner.close()
            self._notify("on_shutdown", self)
        return RunResult(scenario, travel_time, self.metrics)
