from functools import partial
from pathlib import Path

import pytest

from mobslim import (
    NetworkBuilder,
    Plan,
    Population,
    Scenario,
    read_network_xml,
    read_population_xml,
    route_scenario,
)
from mobslim.events import EventLog, EventsManager
from mobslim.mobsim import (
    AgentSim,
    AgentSimConfig,
    HermesSim,
    HermesSimConfig,
    QSim,
    QSimConfig,
)
from mobslim.mobsim.hermes import rust_available

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def equil_paths():
    return FIXTURES / "equil" / "network.xml", FIXTURES / "equil" / "plans100.xml"


@pytest.fixture
def equil_scenario(equil_paths):
    net_path, plans_path = equil_paths
    network = read_network_xml(net_path)
    return Scenario(network, read_population_xml(plans_path, network))


def chain_network(lengths, freespeed=10.0, capacity=3600.0, lanes=1.0):
    """Nodes 0..n in a line with one-directional links i -> i+1."""
    b = NetworkBuilder()
    for i in range(len(lengths) + 1):
        b.add_node(i, sum(lengths[:i]), 0.0)
    for i, length in enumerate(lengths):
        b.add_link(f"{i}-{i + 1}", i, i + 1, length, freespeed, capacity, lanes)
    return b.build()


def routed(scenario: Scenario) -> Scenario:
    """Route all network legs at free speed."""
    return route_scenario(scenario)


MOBSIMS = {
    "qsim": (QSim, QSimConfig),
    "agentsim": (AgentSim, AgentSimConfig),
    "agentsim_chain": (AgentSim, partial(AgentSimConfig, blocked="chain")),
    "hermes_py": (HermesSim, partial(HermesSimConfig, backend="python")),
}
if rust_available():  # the compiled kernel (rust/mobslim-hermes) is optional
    MOBSIMS["hermes"] = (HermesSim, partial(HermesSimConfig, backend="rust"))
DEFAULT_MOBSIM = "qsim"  # tests/test_qsim.py runs its spec against every entry of MOBSIMS
# clock resolution [s] of mobsims that run in whole time steps (others: continuous)
TIME_STEP = {"hermes_py": 1.0, "hermes": 1.0}


def time_step(kind=None) -> float:
    """Clock resolution of ``kind`` (default: ``DEFAULT_MOBSIM``); 0 = continuous."""
    return TIME_STEP.get(kind or DEFAULT_MOBSIM, 0.0)


def make_mobsim(kind=None, **config):
    cls, cfg = MOBSIMS[kind or DEFAULT_MOBSIM]
    return cls(cfg(**config))


def simulate(scenario: Scenario, mobsim=None, **config) -> list:
    """Run ``scenario`` on ``mobsim`` (a key of ``MOBSIMS``; default
    ``DEFAULT_MOBSIM``) and return the events."""
    log = EventLog()
    make_mobsim(mobsim, **config).run(scenario, EventsManager([log]))
    return log.events


def commuters(n, origin, destination, dep=0.0, mode="car"):
    plans = {
        i: Plan.builder()
        .act("home", origin, end_time=dep)
        .leg(mode)
        .act("work", destination)
        .build()
        for i in range(n)
    }
    return Population.from_plans(plans)
