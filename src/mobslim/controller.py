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
from typing import Iterable, Optional, Union

from mobslim.events import EventsManager
from mobslim.listeners import (
    ControllerListener,
    EventsXMLListener,
    IterationState,
    MetricsCSVWriter,
    ProgressLogger,
)
from mobslim.metrics import iteration_metrics
from mobslim.mobsim.base import Mobsim
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
from mobslim.routing.traveltime import DEFAULT_HORIZON, TravelTimeTable
from mobslim.scenario import Scenario
from mobslim.scoring import CharyparNagelScorer, Scorer, apply_scores, score_all

logger = logging.getLogger("mobslim")


@dataclass(frozen=True)
class ControllerConfig:
    iterations: int = 10  # number of mobsim runs
    seed: int = 4711
    learning_rate: float = 1.0  # score averaging: new = (1-lr) * old + lr * observed
    travel_time_bin_size: float = 900.0
    travel_time_horizon: float = DEFAULT_HORIZON
    output_dir: Optional[Union[str, Path]] = None
    write_events_interval: int = 0  # with output_dir: every N iterations (0 = last only)

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1.")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")


@dataclass
class RunResult:
    scenario: Scenario  # final population with scored plans
    travel_time: TravelTimeTable  # learned in the last iteration
    metrics: list[dict]

    def metrics_frame(self):
        import pandas as pd

        return pd.DataFrame(self.metrics).set_index("iteration")


class Controller:
    def __init__(
        self,
        scenario: Scenario,
        *,
        mobsim: Optional[Mobsim] = None,
        scorer: Optional[Scorer] = None,
        strategies: Optional[StrategyManager] = None,
        runner: Optional[ReplanningRunner] = None,
        router_factory: RouterFactory = default_router,
        transit_router_factory=None,
        transit_weights=None,
        listeners: Iterable[ControllerListener] = (),
        config: ControllerConfig = ControllerConfig(),
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
        self.config = config
        self.listeners: list[ControllerListener] = [ProgressLogger()]
        if config.output_dir is not None:
            self.listeners.append(MetricsCSVWriter(config.output_dir))
            self.listeners.append(EventsXMLListener(config.output_dir, config.write_events_interval))
        self.listeners.extend(listeners)
        self.metrics: list[dict] = []

    @classmethod
    def from_matsim_config(cls, path, *, strict: bool = False, **overrides) -> "Controller":
        """A controller for a MATSim ``config.xml``.

        Reads the network, plans and transit it names (paths relative to the
        config file) and translates the ``controller``, ``qsim``/``hermes``,
        ``scoring``, ``replanning`` and ``routing`` modules (see
        :mod:`mobslim.config.matsim`). Params mobslim cannot honour are
        logged as a warning, or raise :class:`UnsupportedConfigError` if
        ``strict``. ``overrides`` replace any ``Controller`` argument, e.g.
        ``listeners=[...]`` or ``config=ControllerConfig(iterations=5)``.
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
        kwargs.update(overrides)
        return cls(kwargs.pop("scenario"), **kwargs)

    @classmethod
    def from_mobslim_config(cls, path, **overrides) -> "Controller":
        """A controller for a mobslim YAML config (see
        :mod:`mobslim.config.mobslim`): loads and checks it, reads the
        inputs, prechecks them against it (raising one ``ConfigError`` that
        lists every problem), and writes ``output_config.yaml`` at startup.
        ``overrides`` replace any ``Controller`` argument; ``listeners`` are
        added to the config's own."""
        from mobslim.config.mobslim import MobslimConfig

        cfg = path if isinstance(path, MobslimConfig) else MobslimConfig.load(path)
        kwargs = cfg.controller_kwargs()
        listeners = [*kwargs.pop("listeners"), *overrides.pop("listeners", ())]
        kwargs.update(overrides)
        return cls(kwargs.pop("scenario"), listeners=listeners, **kwargs)

    @staticmethod
    def _transit_router_from_scoring(scenario: Scenario, scorer):
        """The scorer's pt utilities (for the first transit passenger mode),
        or RaptorConfig's defaults if the scorer has no ``ScoringParams``."""
        from mobslim.scoring import ScoringParams

        params = getattr(scorer, "params", None)
        if not isinstance(params, ScoringParams):
            return default_transit_router
        transit = scenario.transit
        mode = transit.config.passenger_modes[0] if transit is not None else "pt"
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
            weights = from_scorer(attributes, scenario.transit.config.passenger_modes[0])
        if weights is None:
            return None
        if not isinstance(weights, PersonRaptorWeights):
            weights = PersonRaptorWeights.from_sequence(weights)
        if len(weights) != len(scenario.population):
            raise ValueError(f"transit_weights gave {len(weights)} weights for "
                             f"{len(scenario.population)} persons.")
        return weights if weights.any else None

    def add_listener(self, listener: ControllerListener) -> ControllerListener:
        self.listeners.append(listener)
        return listener

    def _notify(self, hook: str, arg) -> None:
        for listener in self.listeners:
            getattr(listener, hook)(arg)

    def free_speed_travel_time(self) -> TravelTimeTable:
        c = self.config
        return TravelTimeTable.free_speed(self.scenario.network, c.travel_time_bin_size, c.travel_time_horizon)

    def prepare(self, scenario: Scenario) -> Scenario:
        """Validate modes and route any un-routed network (free speed) and pt legs."""
        return route_scenario(scenario, self.free_speed_travel_time(), self.router_factory,
                              self.transit_router_factory, self.person_transit_weights(scenario))

    def run(self) -> RunResult:
        cfg = self.config
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
                trajectories = state.trajectories = self.mobsim.run(scenario, events)
                events.finish()
                self._notify("on_after_mobsim", state)

                scores = score_all(self.scorer, trajectories, scenario.population.attributes)
                scenario = state.scenario = scenario.with_population(
                    apply_scores(scenario.population, scores, cfg.learning_rate)
                )
                self._notify("on_scoring", state)

                travel_time = state.travel_time = trajectories.travel_time_table(
                    scenario.network, cfg.travel_time_bin_size, cfg.travel_time_horizon
                )
                state.metrics = iteration_metrics(it, scenario.population, trajectories)
                self.metrics.append(state.metrics)
                self._notify("on_iteration_end", state)
        finally:
            self.runner.close()
            self._notify("on_shutdown", self)
        return RunResult(scenario, travel_time, self.metrics)
