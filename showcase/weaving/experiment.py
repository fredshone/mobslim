"""Weaving on a grid with medium congestion: utility against wall-clock time.

The scenario is the performance showcase's ``grid-80`` inputs (an 80x80
grid, 26,240 links, 103,680 car commuters;
``showcase/output/perf/grid-80/input``, made with ``showcase.perf`` on
first use) with flow and storage capacity
scaled by :data:`CAPACITY`, so that the first iteration's car travel time is
about 50% above free flow, with nothing stuck. It runs for :data:`ITERATIONS`
iterations with 10% plan innovation (5% reroute, 5% time mutation plus
reroute, 90% ChangeExpBeta) under a budget of :data:`BUDGET` simulation
processes (``$WEAVING_BUDGET``, default 8; set it to the machine's
physical cores). Two experiments:

* ``weaving``: HermesSim (single-threaded) with 1, 2, 4, ... strands, up to
  the budget, plus
  the plain :class:`~mobslim.Controller` for reference.
* ``budget``: PSim, splitting the budget between in-simulation threads
  ``p`` and strands ``w`` (``p * w = 8``).

Each run is a fresh process. It records, per iteration, the wall-clock time
since the loop started (after the one-off routing of the input plans, the
same for every run) and the average utility of the plans performed in that
iteration (over every strand). Results go to ``showcase/results/weaving.json``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "showcase" / "output" / "perf" / "grid-80" / "input" / "config.xml"
OUT = ROOT / "showcase" / "output" / "weaving"
RESULTS = ROOT / "showcase" / "results" / "weaving.json"
ITERATIONS = int(os.environ.get("WEAVING_ITERATIONS", 100))  # env: for a quick trial
INNOVATION = 0.1
CAPACITY = 0.3  # flow and storage capacity factor: medium congestion
BUDGET = int(os.environ.get("WEAVING_BUDGET", 8))  # processes; best: the physical cores
SEED = 4711


@dataclass(frozen=True)
class Run:
    name: str
    experiment: str
    mobsim: str  # hermes | psim
    strands: int = 1  # 0: the plain Controller
    threads: int = 1  # psim threads per simulation

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-").replace(",", "").replace("×", "x")


POWERS = [2**i for i in range(BUDGET.bit_length()) if 2**i <= BUDGET]  # 1, 2, 4, ... <= BUDGET
RUNS = [
    Run("Controller", "weaving", "hermes", strands=0),
    *(Run(f"{w} strand{'s' if w > 1 else ''}", "weaving", "hermes", strands=w) for w in POWERS),
    *(Run(f"p={p}, w={BUDGET // p}", "budget", "psim", strands=BUDGET // p, threads=p)
      for p in range(BUDGET, 0, -1) if BUDGET % p == 0),
]


def strategies():
    from mobslim import PlanStrategy, StrategyManager
    from mobslim.replanning import (
        ChangeExpBetaSelector,
        RandomSelector,
        Reroute,
        TimeAllocationMutator,
    )

    half = INNOVATION / 2
    return StrategyManager([
        (1 - INNOVATION, PlanStrategy(ChangeExpBetaSelector())),
        (half, PlanStrategy(RandomSelector(), [Reroute()])),
        (half, PlanStrategy(RandomSelector(), [TimeAllocationMutator(), Reroute()])),
    ])


def child(slug: str, out: str) -> None:
    """One run in this process; writes its per-iteration record to ``out``."""
    from mobslim import (
        Controller,
        ControllerListener,
        ControllerParams,
        HermesSim,
        HermesSimParams,
        PSim,
        PSimParams,
    )
    from mobslim.config.matsim import MatsimConfig, mobsim_params
    from mobslim.weave import Weave

    run = next(r for r in RUNS if r.slug == slug)
    cfg = MatsimConfig.load(CONFIG)
    kw, _ = mobsim_params(cfg)
    kw.update(flow_capacity_factor=CAPACITY, storage_capacity_factor=CAPACITY)
    mobsim = (HermesSim(HermesSimParams(**kw, backend="rust")) if run.mobsim == "hermes"
              else PSim(PSimParams(**kw, backend="rust", threads=run.threads)))
    params = ControllerParams(iterations=ITERATIONS, seed=SEED)

    class Record(ControllerListener):
        def __init__(self):
            self.rows = []

        def on_startup(self, controller):
            self.t0 = time.perf_counter()

        def on_iteration_end(self, state):
            m = state.metrics
            self.rows.append({
                "iteration": state.iteration,
                "wall_s": time.perf_counter() - self.t0,
                "score_performed": m.get("score_performed", m["score_executed"]),
                "score_executed": m["score_executed"],
                "score_best": m["score_best"],
                "car_travel_time": m.get("car_travel_time"),
                "coverage": m.get("coverage"),
                "replanning_s": m.get("replanning_s"),
                "simulation_s": m.get("simulation_s"),
            })

    record = Record()
    common = dict(strategies=strategies(), mobsim=mobsim, params=params, listeners=[record])
    t = time.perf_counter()
    if run.strands == 0:
        controller = Controller.from_matsim_config(cfg, **common)
    else:
        controller = Weave.from_matsim_config(cfg, strands=run.strands,
                                              innovation_rate=INNOVATION, **common)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    controller.run()
    total_s = time.perf_counter() - t
    Path(out).write_text(json.dumps({**asdict(run), "load_s": load_s, "total_s": total_s,
                                     "persons": len(controller.scenario.population),
                                     "iterations": record.rows}))


def machine() -> dict:
    model = next((line.split(":", 1)[1].strip() for line in
                  Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
                 platform.processor())
    mem = next((int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal")), 0)
    cores = len({line for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("core id")}) or os.cpu_count()
    return {"cpu": model, "logical_cpus": os.cpu_count(), "physical_cores": cores,
            "memory_gb": round(mem / 1024**2, 1), "python": platform.python_version()}


def main(only: list[str] = ()) -> None:
    if not CONFIG.exists():
        from showcase.perf import PerfCase, prepare

        prepare(PerfCase("grid-80", "80x80 grid: 26,240 links, 103,680 car commuters",
                         grid=80, persons=103_680))
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    results["machine"] = machine()
    results["settings"] = {"iterations": ITERATIONS, "innovation_rate": INNOVATION,
                           "budget": BUDGET, "seed": SEED, "capacity": CAPACITY}
    for run in RUNS:
        if only and not any(k in run.slug or k in run.experiment for k in only):
            continue
        out = OUT / f"{run.slug}.json"
        out.unlink(missing_ok=True)
        print(f"{run.experiment}: {run.name} ...", flush=True)
        t = time.perf_counter()
        with open(OUT / f"{run.slug}.log", "w") as log:
            code = subprocess.run([sys.executable, "-m", "showcase.weaving.experiment", "child",
                                   run.slug, str(out)], stdout=log, stderr=subprocess.STDOUT,
                                  cwd=ROOT).returncode
        if code != 0 or not out.exists():
            print(f"  failed (exit {code}), see {OUT / (run.slug + '.log')}")
            continue
        res = json.loads(out.read_text())
        print(f"  {time.perf_counter() - t:.0f} s; final utility "
              f"{res['iterations'][-1]['score_performed']:.3f}", flush=True)
        results["runs"][run.slug] = res
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1:])
