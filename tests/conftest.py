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
    AgentSimParams,
    HermesSim,
    HermesSimParams,
    ParallelSim,
    ParallelSimParams,
    QSim,
    QSimParams,
    ShardSim,
    ShardSimParams,
    parallel,
    shard,
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
    "qsim": (QSim, QSimParams),
    "agentsim": (AgentSim, AgentSimParams),
    "agentsim_chain": (AgentSim, partial(AgentSimParams, blocked="chain")),
    "hermes_py": (HermesSim, partial(HermesSimParams, backend="python")),
    "parallel_py": (ParallelSim, partial(ParallelSimParams, backend="python")),
}
if rust_available():  # the compiled kernel (rust/mobslim-hermes) is optional
    MOBSIMS["hermes"] = (HermesSim, partial(HermesSimParams, backend="rust"))
if parallel.rust_available():  # rust/mobslim-parallel, on 3 threads
    MOBSIMS["parallel"] = (ParallelSim, partial(ParallelSimParams, backend="rust", threads=3))
if shard.rust_available():  # rust/mobslim-shard (no Python kernel), on 3 workers
    MOBSIMS["shard"] = (ShardSim, partial(ShardSimParams, threads=3))
DEFAULT_MOBSIM = "qsim"  # tests/test_qsim.py runs its spec against every entry of MOBSIMS
# clock resolution [s] of mobsims that run in whole time steps (others: continuous)
TIME_STEP = {"hermes_py": 1.0, "hermes": 1.0, "parallel_py": 1.0, "parallel": 1.0, "shard": 1.0}


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


def transit_grid(seed=1, vehicles="network", blocking=False, size=6, persons=400, pt_share=0.7,
                 capacity=600.0, rail=False):
    """Bus lines along every other row and column of a grid (both ways),
    stops at every node, buses every 4 minutes with small fractional door
    and passenger times; commuters by car and by pt (routed by RAPTOR),
    home -> work -> home. Small enough for the Python kernels, busy enough
    that buses fill and queue with cars, and lines cross partitions. With
    ``rail``, the column lines are rail and keep to the timetable
    (``schedule_modes``) while the row lines drive in traffic."""
    from random import Random

    from mobslim import grid_network
    from mobslim.routing import TeleportedModeParams
    from mobslim.transit import TransitBuilder, TransitParams

    walk = TeleportedModeParams(speed=1.2, beeline_distance_factor=1.3)
    net = grid_network(size, spacing=200.0, capacity=capacity, modes=("car", "bus"))
    rng = Random(seed)
    b = TransitBuilder(net)
    b.vehicle_type("bus", seats=2, standing=1, access_time=1.3, egress_time=0.8,
                   doors=rng.choice(["serial", "parallel"]), pce=2.0)
    stops = {}

    def stop_on(link):
        if link not in stops:
            (u, v) = link
            stops[link] = f"S{u}-{v}"
            b.stop(stops[link], v[1] * 200.0, v[0] * 200.0, link=link, blocking=blocking)
        return stops[link]

    lines = []
    for i in range(0, size + 1, 2):
        lines.append(([(i, j) for j in range(size + 1)], "bus"))  # a row
        lines.append(([(j, i) for j in range(size + 1)], "rail" if rail else "bus"))  # a column
    for n, (nodes, mode) in enumerate(lines):
        for name, seq in (("a", nodes), ("b", nodes[::-1])):
            links = list(zip(seq, seq[1:]))
            offsets = [30.0 * k for k in range(len(links))]
            b.route(f"L{n}", name, mode=mode, links=links,
                    stops=[(stop_on(lk), off, off + 10.0, k % 2 == 0)
                           for k, (lk, off) in enumerate(zip(links, offsets))])
            first = 6 * 3600 + rng.uniform(0, 240)
            b.departures(f"L{n}", name, times=[first + 240.0 * k for k in range(25)],
                         vehicle_type="bus")
    transit = b.build(TransitParams(vehicles=vehicles, walk=walk, door_open_time=0.6,
                                    door_close_time=1.5,
                                    schedule_modes=("rail",) if rail else ()))
    plans = {}
    for p in range(persons):
        home, work = rng.sample(range(net.n_nodes), 2)
        mode = "pt" if rng.random() < pt_share else "car"
        plans[p] = (Plan.builder().act("h", home, end_time=6 * 3600 + rng.uniform(0, 3600))
                    .leg(mode).act("w", work, duration=rng.uniform(600, 1800)).leg(mode)
                    .act("h", home).build())
    return route_scenario(Scenario(net, Population.from_plans(plans),
                                   teleported_modes={"walk": walk}, transit=transit))
