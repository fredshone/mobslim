"""Benchmark scenarios.

A :class:`ScenarioSpec` says how to build a network and a population. Every
benchmark runs against one spec, so adding a scenario (bigger network, more
agents, more congestion, synthetic grids, ...) means registering one more
spec here; the benchmarks themselves do not change.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Optional

from mobslim import (
    Network,
    Plan,
    Population,
    grid_network,
    read_network_xml,
    read_population_xml,
)
from mobslim.transit import Transit, read_transit

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    load_network: Callable[[], Network]
    load_population: Callable[[Network], Population]
    iterations: int = 10  # for the end-to-end "run" benchmarks
    load_transit: Optional[Callable[[Network], Transit]] = None  # scheduled transit, if any


SCENARIOS: dict[str, ScenarioSpec] = {}


def register(spec: ScenarioSpec) -> ScenarioSpec:
    if spec.name in SCENARIOS:
        raise ValueError(f"Scenario {spec.name!r} is already registered.")
    SCENARIOS[spec.name] = spec
    return spec


def xml_scenario(name: str, description: str, network: Path, population: Path,
                 iterations: int = 10, schedule: Optional[Path] = None,
                 vehicles: Optional[Path] = None) -> ScenarioSpec:
    """A scenario loaded from MATSim network and population XML files (and
    a transit schedule and vehicles, if given)."""
    load_transit = None
    if schedule is not None:
        def load_transit(net: Network) -> Transit:
            return read_transit(schedule, vehicles, net)
    return ScenarioSpec(
        name,
        description,
        lambda: read_network_xml(network),
        lambda net: read_population_xml(population, net),
        iterations,
        load_transit,
    )


register(
    xml_scenario(
        "equil",
        "MATSim equil: 15 nodes, 23 links, 100 car commuters",
        SCENARIO_DIR / "equil" / "network.xml",
        SCENARIO_DIR / "equil" / "plans100.xml",
    )
)


def grid_commuters(name: str, description: str, size: int, persons: int,
                   iterations: int = 5, seed: int = 1) -> ScenarioSpec:
    """Synthetic: a ``size x size`` grid and random home-work-home car commuters.

    Big enough that routing and the mobsim dominate, unlike equil. Fully
    deterministic for a given ``seed``.
    """

    def load_network() -> Network:
        return grid_network(size, spacing=200.0, capacity=600.0)

    def load_population(net: Network) -> Population:
        rng = Random(seed)
        n = net.n_nodes
        plans = {}
        for i in range(persons):
            home = rng.randrange(n)
            work = rng.randrange(n - 1)
            work += work >= home  # never the home node
            plans[i] = (
                Plan.builder()
                .act("home", home, end_time=6 * 3600 + rng.uniform(0, 3 * 3600))
                .leg("car")
                .act("work", work, duration=8 * 3600)
                .leg("car")
                .act("home", home)
                .build()
            )
        return Population.from_plans(plans)

    return ScenarioSpec(name, description, load_network, load_population, iterations)


register(
    grid_commuters(
        "grid",
        "Synthetic 20x20 grid (441 nodes, 1680 links), 2000 random car commuters",
        size=20,
        persons=2000,
    )
)


register(
    grid_commuters(
        "grid_large",
        "Synthetic 40x40 grid (1681 nodes, 6560 links), 100k random car commuters",
        size=40,
        persons=100_000,
        iterations=2,
    )
)


FIXTURE_DIR = ROOT / "tests" / "fixtures"

register(
    xml_scenario(
        "pt_tutorial",
        "MATSim pt-tutorial: 3x3 km grid, 2 rail lines (1 line switch), 1000 car/pt commuters",
        FIXTURE_DIR / "pt_tutorial" / "network.xml",
        FIXTURE_DIR / "pt_tutorial" / "population.xml.gz",
        iterations=5,
        schedule=FIXTURE_DIR / "pt_tutorial" / "transitSchedule.xml",
        vehicles=FIXTURE_DIR / "pt_tutorial" / "transitVehicles.xml",
    )
)

# Sioux Falls (not bundled): set MOBSLIM_SIOUXFALLS to a directory with either
# matsim-eth/sioux-falls's src/main/resources/ch/ethz/matsim/sioux_falls files
# or MATSim's examples/scenarios/siouxfalls-2014
SIOUXFALLS_LAYOUTS = (
    ("network.xml", "population.xml.gz", "schedule.xml", "vehicles.xml"),
    ("Siouxfalls_network_PT.xml", "Siouxfalls_population.xml.gz",
     "Siouxfalls_transitSchedule.xml", "Siouxfalls_vehicles.xml"),
)
SIOUXFALLS = os.environ.get("MOBSLIM_SIOUXFALLS")
if SIOUXFALLS:
    _sf = Path(SIOUXFALLS)
    _files = next((f for f in SIOUXFALLS_LAYOUTS if (_sf / f[2]).exists()), None)
    if _files is None:
        raise FileNotFoundError(f"MOBSLIM_SIOUXFALLS={SIOUXFALLS} holds no Sioux Falls schedule")
    register(
        xml_scenario(
            "siouxfalls",
            "MATSim Sioux Falls: 10 bus lines, ~2000 departures, 84k car/pt/walk persons",
            _sf / _files[0],
            _sf / _files[1],
            iterations=3,
            schedule=_sf / _files[2],
            vehicles=_sf / _files[3],
        )
    )
