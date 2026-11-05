"""Verification cases: a MATSim scenario, config edits, and the layer they test."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mobslim.mobsim import (
    AgentSim,
    AgentSimParams,
    HermesSim,
    HermesSimParams,
    QSim,
    QSimParams,
)
from mobslim.mobsim.hermes import rust_available

MATSIM_LIBS = Path(os.environ.get("MATSIM_LIBS", Path.home() / "projects" / "matsim-libs"))
SCENARIOS = MATSIM_LIBS / "examples" / "scenarios"


@dataclass(frozen=True)
class Case:
    name: str
    layer: str  # "l1" scoring, "l2" mobsim, "l3" loop
    scenario: str  # directory under SCENARIOS
    config: str = "config.xml"
    # (module, param) -> value, applied to the MATSim config (both sides read it)
    params: dict = field(default_factory=dict)
    # (scoring parameterset type, param) -> value, set in every such set
    scoring_sets: dict = field(default_factory=dict)
    description: str = ""
    seeds: int = 1  # layer 3: runs per side
    references: tuple = ("qsim",)  # layer 2: MATSim mobsims to compare against
    vehicles: tuple = ("network",)  # layer 2 with transit: mobslim vehicle modes
    # mobslim's pt walk (speed m/s, beeline factor) if not its default (MATSim
    # 2025's SwissRailRaptor: 3 km/h, x1.3)
    transit_walk: Optional[tuple] = None

    @property
    def config_path(self) -> Path:
        return SCENARIOS / self.scenario / self.config


# mobslim mobsims a layer-2 case runs: name -> factory(**qsim_kwargs)
MOBSIMS = {
    "qsim": lambda **kw: QSim(QSimParams(**kw)),
    "agentsim": lambda **kw: AgentSim(AgentSimParams(**kw)),
    "agentsim_chain": lambda **kw: AgentSim(AgentSimParams(blocked="chain", **kw)),
    "hermes": lambda **kw: HermesSim(HermesSimParams(
        backend="rust" if rust_available() else "python", **kw)),
}

_L2 = {("controller", "lastIteration"): 0}  # one iteration: the input plans, no replanning


def _equil_l2(name, plans, capacity=None, description=""):
    params = dict(_L2)
    params[("plans", "inputPlansFile")] = plans
    if capacity is not None:
        params[("qsim", "flowCapacityFactor")] = capacity
        params[("qsim", "storageCapacityFactor")] = capacity
    return Case(name, "l2", "equil", params=params, description=description,
                references=("qsim", "hermes"))


CASES = [
    _equil_l2("equil-100.l2", "plans100.xml",
              description="100 agents, light congestion: the cleanest baseline"),
    _equil_l2("equil-2000.l2", "plans2000.xml.gz",
              description="2000 agents: queues on the bottleneck links"),
    _equil_l2("equil-2000-cap0.5.l2", "plans2000.xml.gz", 0.5,
              description="2000 agents, flow and storage capacity x0.5"),
    _equil_l2("equil-2000-cap0.2.l2", "plans2000.xml.gz", 0.2,
              description="2000 agents, flow and storage capacity x0.2: spillback"),
]


CASES.append(Case(
    "pt-tutorial.l2", "l2", "pt-tutorial", "0.config.xml", params=dict(_L2),
    references=("qsim",), vehicles=("network", "schedule"),  # no MATSim Hermes reference for pt-tutorial
    # 0.plans.xml.gz comes with pt routes from an older MATSim router, whose
    # walks are beeline / 3 km/h (factor 1.0); MATSim keeps and teleports them
    transit_walk=(3 / 3.6, 1.0),
    description="900 agents, car grid and a train line: mixed traffic, stops, boardings"))

_L1 = {("controller", "lastIteration"): 0, ("scoring", "writeScoreExplanations"): "true"}
_UNIFORM = {("activityParams", "typicalDurationScoreComputation"): "uniform"}

CASES += [
    Case("equil-100.l1", "l1", "equil", params=dict(_L1),
         description="equil as shipped: relative zero-utility durations, opening times, lateness"),
    Case("equil-100-uniform.l1", "l1", "equil", params=dict(_L1), scoring_sets=dict(_UNIFORM),
         description="equil with uniform zero-utility durations"),
    Case("equil-2000-cap0.5.l1", "l1", "equil",
         params={**_L1, ("plans", "inputPlansFile"): "plans2000.xml.gz",
                 ("qsim", "flowCapacityFactor"): 0.5, ("qsim", "storageCapacityFactor"): 0.5},
         description="2000 agents at capacity x0.5: congestion, late arrivals at work"),
    Case("pt-tutorial.l1", "l1", "pt-tutorial", "0.config.xml", params=dict(_L1),
         description="900 agents, car and pt: waiting, line switches, walks, shop opening times"),
]


CASES += [
    Case("equil-100.l3", "l3", "equil", params={("controller", "lastIteration"): 50}, seeds=5,
         description="equil, 100 agents, BestScore + ReRoute, 51 iterations"),
    Case("equil-2000-cap0.5.l3", "l3", "equil",
         params={("controller", "lastIteration"): 50, ("plans", "inputPlansFile"): "plans2000.xml.gz",
                 ("qsim", "flowCapacityFactor"): 0.5, ("qsim", "storageCapacityFactor"): 0.5},
         seeds=5, description="equil, 2000 agents at capacity x0.5: route choice under congestion"),
    Case("pt-tutorial.l3", "l3", "pt-tutorial", "0.config.xml", seeds=5,
         description="pt-tutorial as shipped: 51 iterations with time and car/pt mode choice"),
]


def select(patterns=()) -> list[Case]:
    if not patterns:
        return list(CASES)
    return [c for c in CASES if any(p in c.name or p == c.layer for p in patterns)]
