"""Weaving on a grid with medium congestion: utility against wall-clock time.

The scenario is the performance showcase's ``grid-80`` inputs (an 80x80
grid, 25,920 links, 104,960 car commuters;
``showcase/output/perf/grid-80/input``, made with ``showcase.perf`` on
first use) with flow and storage capacity
scaled by :data:`CAPACITY`, so that the first iteration's car travel time is
about 50% above free flow, with nothing stuck. Each run gets the same
wall-clock time, :data:`TIME_LIMIT` seconds (``$WEAVING_TIME_LIMIT``; it
stops after the iteration that reaches it, or after :data:`ITERATIONS`),
with 30% plan innovation (15% reroute, 15% time mutation plus reroute, 70%
ChangeExpBeta) under a budget of :data:`BUDGET` simulation
processes (``$WEAVING_BUDGET``, default 8; set it to the machine's
physical cores). Two experiments:

* ``weaving``: HermesSim (single-threaded) with 1, 2, 4, ... strands, up to
  the budget, plus
  the plain :class:`~mobslim.Controller` for reference.
* ``async``: HermesSim with :class:`~mobslim.weave.AsyncWeave` (strands
  merged one at a time, never waiting for the slowest) with 2, 4, ...
  strands, against the synchronous runs.

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
ITERATIONS = int(os.environ.get("WEAVING_ITERATIONS", 10_000))  # env: for a quick trial
TIME_LIMIT = float(os.environ.get("WEAVING_TIME_LIMIT", 180))  # wall seconds per run
INNOVATION = 0.3
CAPACITY = 0.3  # flow and storage capacity factor: medium congestion
BUDGET = int(os.environ.get("WEAVING_BUDGET", 8))  # processes; best: the physical cores
SEED = 4711
TIMING = ("_s", "_mb")  # metric keys recorded per strand


@dataclass(frozen=True)
class Run:
    name: str
    experiment: str
    strands: int = 1  # 0: the plain Controller
    loop: str = "sync"  # sync: Weave | async: AsyncWeave

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-").replace(",", "").replace("×", "x")


POWERS = [2**i for i in range(BUDGET.bit_length()) if 2**i <= BUDGET]  # 1, 2, 4, ... <= BUDGET
RUNS = [
    Run("Controller", "weaving", strands=0),
    *(Run(f"{w} strand{'s' if w > 1 else ''}", "weaving", strands=w) for w in POWERS),
    *(Run(f"async, {w} strands", "async", strands=w, loop="async")
      for w in POWERS if w > 1),
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
    )
    from mobslim.config.matsim import MatsimConfig, mobsim_params
    from mobslim.weave import AsyncWeave, Weave

    run = next(r for r in RUNS if r.slug == slug)
    cfg = MatsimConfig.load(CONFIG)
    kw, _ = mobsim_params(cfg)
    kw.update(flow_capacity_factor=CAPACITY, storage_capacity_factor=CAPACITY)
    mobsim = HermesSim(HermesSimParams(**kw, backend="rust"))
    params = ControllerParams(iterations=ITERATIONS, seed=SEED)

    class TimeUp(Exception):
        pass

    class Record(ControllerListener):
        def __init__(self):
            self.rows = []

        def on_startup(self, controller):
            self.t0 = time.perf_counter()

        def on_iteration_end(self, state):
            m = state.metrics
            row = {
                "iteration": state.iteration,
                "wall_s": time.perf_counter() - self.t0,
                "score_performed": m.get("score_performed", m["score_executed"]),
                "score_executed": m["score_executed"],
                "score_best": m["score_best"],
                "car_travel_time": m.get("car_travel_time"),
                "coverage": m.get("coverage"),
                **{k: v for k, v in m.items() if k.endswith("_s")},
            }
            strands = getattr(state, "strands", None)
            if strands:  # per-strand phases: strand 0's, and the max over strands
                keys = sorted({k for r in strands for k in r.metrics if k.endswith(TIMING)})
                row["strand0"] = {k: strands[0].metrics[k] for k in keys if k in strands[0].metrics}
                row["strand_max"] = {k: max(r.metrics.get(k, 0.0) for r in strands) for k in keys}
            self.rows.append(row)
            if row["wall_s"] >= TIME_LIMIT:
                raise TimeUp  # the controllers close their workers on the way out

    record = Record()
    common = dict(strategies=strategies(), mobsim=mobsim, params=params, listeners=[record])
    t = time.perf_counter()
    if run.strands == 0:
        controller = Controller.from_matsim_config(cfg, **common)
    else:
        loop = AsyncWeave if run.loop == "async" else Weave
        controller = loop.from_matsim_config(cfg, strands=run.strands,
                                             innovation_rate=INNOVATION, **common)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    try:
        controller.run()
    except TimeUp:
        pass
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

        prepare(PerfCase("grid-80", "80x80 grid: 25,920 links, 104,960 car commuters",
                         grid=80, persons=104_960))
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    results["machine"] = machine()
    results["settings"] = {"time_limit_s": TIME_LIMIT, "innovation_rate": INNOVATION,
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
