"""MobSlim: a modular, MATSim-inspired agent-based transport simulation.

The most common entry points are re-exported here; see the sub-packages for
the full interfaces (``mobsim``, ``routing``, ``replanning``).
"""

from mobslim.attributes import PersonAttributes
from mobslim.controller import Controller, ControllerConfig, RunResult
from mobslim.events import EventLog, EventsManager
from mobslim.experience import ExperiencedPlan
from mobslim.formats import read_population, write_population
from mobslim.listeners import ControllerListener, EventRecorder, IterationState
from mobslim.mobsim import (
    AgentSim,
    AgentSimConfig,
    HermesSim,
    HermesSimConfig,
    QSim,
    QSimConfig,
)
from mobslim.modes import Modes
from mobslim.network import (
    Network,
    NetworkBuilder,
    grid_network,
    linear_network,
    read_network_xml,
)
from mobslim.planbatch import PlanBatch
from mobslim.planstore import PlanStore
from mobslim.population import (
    Activity,
    Leg,
    Person,
    Plan,
    Population,
    Route,
    read_population_xml,
)
from mobslim.replanning import PlanStrategy, StrategyManager, route_scenario
from mobslim.routing import TeleportedModeParams, TravelTimeTable
from mobslim.scenario import Scenario
from mobslim.scoring import (
    ActivityParams,
    CharyparNagelScorer,
    ModeParams,
    ScoringParams,
    apply_scores,
)
from mobslim.trajectories import Trajectories, TrajectoryCollector

__all__ = [
    "Activity",
    "ActivityParams",
    "AgentSim",
    "AgentSimConfig",
    "CharyparNagelScorer",
    "Controller",
    "ControllerConfig",
    "ControllerListener",
    "EventLog",
    "EventRecorder",
    "EventsManager",
    "ExperiencedPlan",
    "HermesSim",
    "HermesSimConfig",
    "IterationState",
    "Modes",
    "Leg",
    "ModeParams",
    "Network",
    "NetworkBuilder",
    "Person",
    "PersonAttributes",
    "Plan",
    "PlanBatch",
    "PlanStore",
    "PlanStrategy",
    "Population",
    "QSim",
    "QSimConfig",
    "Route",
    "RunResult",
    "Scenario",
    "ScoringParams",
    "StrategyManager",
    "TeleportedModeParams",
    "Trajectories",
    "TrajectoryCollector",
    "TravelTimeTable",
    "apply_scores",
    "grid_network",
    "linear_network",
    "read_network_xml",
    "read_population",
    "read_population_xml",
    "write_population",
    "route_scenario",
]
