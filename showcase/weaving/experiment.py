"""Weaving on Open Berlin: utility against wall-clock time.

The scenario is MATSim Open Berlin v6.4 at 1% (about 52,000 agents; car,
truck and freight on the roads, and pt) from its *initial* plans: the
synthetic demand before calibration, unrouted, so each side routes them as
a real run starts (``showcase/output/perf/open-berlin-1pct-initial/input``,
made with ``showcase.perf.prepare`` when missing), at flow and storage
capacity :data:`CAPACITY`. Each
run gets the same wall-clock time, :data:`TIME_LIMIT` seconds
(``$WEAVING_TIME_LIMIT``; it stops after the iteration that reaches it, or
after :data:`ITERATIONS`), with 30% plan innovation (15% reroute, 15% time
mutation plus reroute, 70% ChangeExpBeta) under a budget of :data:`BUDGET`
simulation processes (``$WEAVING_BUDGET``, default 8; set it to the
machine's physical cores). Two experiments:

* ``weaving``: HermesSim (single-threaded) with 1, 2, 4, ... strands, up to
  the budget, plus the plain :class:`~mobslim.Controller` and Java MATSim
  with its Hermes mobsim (``ReplanRun``, ``showcase/java/src``; the same
  strategies, :data:`BUDGET` threads) for reference.
* ``async``: HermesSim with :class:`~mobslim.weave.AsyncWeave` (strands
  merged one at a time, never waiting for the slowest) with 2, 4, ...
  strands, against the synchronous runs.

Each run is a fresh process. It records, per iteration, the wall-clock time
since the loop started (after the one-off routing of the input plans, the
same for every run) and the average utility of the plans performed in that
iteration (over every strand); the start and end of every mobsim run, in
whichever process ran it (``sims``: ``[start_s, end_s, pid]``); and what each
process did when (``timeline``: ``[phase, start_s, end_s, pid]``, phases
``innovate`` (mutating and routing), ``mobsim``, ``score``, and in the main
process ``plan`` and ``merge``; ``main_pid`` is the main process). Results go
to ``showcase/results/weaving.json``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mobslim import HermesSim

ROOT = Path(__file__).resolve().parents[2]
CASE = "open-berlin-1pct-initial"  # Open Berlin's initial (uncalibrated, unrouted) 1% plans
CONFIG = ROOT / "showcase" / "output" / "perf" / CASE / "input" / "config.xml"
OUT = ROOT / "showcase" / "output" / "weaving"
RESULTS = ROOT / "showcase" / "results" / "weaving.json"
ITERATIONS = int(os.environ.get("WEAVING_ITERATIONS", 10_000))  # env: for a quick trial
TIME_LIMIT = float(os.environ.get("WEAVING_TIME_LIMIT", 1200))  # wall seconds per run
INNOVATION = 0.3
CAPACITY = 0.01  # flow and storage capacity factor (the case's own: the sample size)
BUDGET = int(os.environ.get("WEAVING_BUDGET", 8))  # processes; best: the physical cores
SEED = 4711
TIMING = ("_s", "_mb")  # metric keys recorded per strand


@dataclass(frozen=True)
class Run:
    name: str
    experiment: str
    strands: int = 1  # 0: the plain Controller (or MATSim)
    loop: str = "sync"  # sync: Weave | async: AsyncWeave | matsim: Java MATSim

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-").replace(",", "").replace("×", "x")


POWERS = [2**i for i in range(BUDGET.bit_length()) if 2**i <= BUDGET]  # 1, 2, 4, ... <= BUDGET
RUNS = [
    Run("Controller", "weaving", strands=0),
    Run("MATSim Hermes", "weaving", strands=0, loop="matsim"),
    *(Run(f"{w} strand{'s' if w > 1 else ''}", "weaving", strands=w) for w in POWERS),
    *(Run(f"async, {w} strands", "async", strands=w, loop="async")
      for w in POWERS if w > 1),
]


class TimedHermesSim(HermesSim):
    """HermesSim that appends each run's wall-clock start and end (epoch
    seconds) and its process id to ``log``, from whichever process runs it."""

    def __init__(self, params, log: str):
        super().__init__(params)
        self.log = log

    def run(self, *args, **kwargs):
        t0 = time.time()
        out = super().run(*args, **kwargs)
        with open(self.log, "a") as f:  # one short append per sim
            f.write(f"{t0},{time.time()},{os.getpid()}\n")
        return out


# MATSim's names for the same strategies as :func:`strategies`
MATSIM_STRATEGIES = {"ChangeExpBeta": 1 - INNOVATION, "ReRoute": INNOVATION / 2,
                     "TimeAllocationMutator_ReRoute": INNOVATION / 2}


def strategies(cfg):
    """:data:`MATSIM_STRATEGIES` in MobSlim, with the time mutation set as
    ``cfg``'s ``timeAllocationMutator`` sets MATSim's (Open Berlin's mutates
    around the initial end times)."""
    from mobslim import PlanStrategy, StrategyManager
    from mobslim.config.matsim import strategy_manager
    from mobslim.replanning import (
        ChangeExpBetaSelector,
        RandomSelector,
        Reroute,
        TimeAllocationMutator,
    )

    tam = next(m for _, s in strategy_manager(cfg)[0].strategies for m in s.modules
               if isinstance(m, TimeAllocationMutator))  # cfg has MATSIM_STRATEGIES (load_config)
    half = INNOVATION / 2
    return StrategyManager([
        (1 - INNOVATION, PlanStrategy(ChangeExpBetaSelector())),
        (half, PlanStrategy(RandomSelector(), [Reroute()])),
        (half, PlanStrategy(RandomSelector(), [tam, Reroute()])),
    ])


def load_config():
    """The case's config, with :data:`MATSIM_STRATEGIES` as its replanning
    when it has none (Open Berlin sets its strategies in code)."""
    import xml.etree.ElementTree as ET

    from mobslim.config.matsim import MatsimConfig

    cfg = MatsimConfig.load(CONFIG)
    module = cfg._module("replanning", create=True)
    if not module.findall("parameterset[@type='strategysettings']"):
        for name, weight in MATSIM_STRATEGIES.items():
            ps = ET.SubElement(module, "parameterset", type="strategysettings")
            ET.SubElement(ps, "param", name="strategyName", value=name)
            ET.SubElement(ps, "param", name="weight", value=str(weight))
    return cfg


def child(slug: str, out: str) -> None:
    """One run in this process; writes its per-iteration record to ``out``."""
    from mobslim import (
        Controller,
        ControllerListener,
        HermesSimParams,
    )
    from mobslim.config.matsim import (
        controller_params,
        mobsim_params,
        network_router_factory,
        scoring_params_by_subpopulation,
    )
    from mobslim.weave import AsyncWeave, Weave

    # by its module path, not __main__'s, so worker processes can unpickle it
    from showcase.weaving.experiment import TimedHermesSim

    run = next(r for r in RUNS if r.slug == slug)
    cfg = load_config()
    kw, _ = mobsim_params(cfg)
    kw.update(flow_capacity_factor=CAPACITY, storage_capacity_factor=CAPACITY)
    sims = Path(out).with_suffix(".sims.csv")
    sims.unlink(missing_ok=True)
    mobsim = TimedHermesSim(HermesSimParams(**kw, backend="rust"), str(sims))
    # the config's own params (as MATSim: e.g. activities to links by XY2Links' rule), writing no files
    params = replace(controller_params(cfg)[0], iterations=ITERATIONS, seed=SEED, output_dir=None)
    # A* bounded as MATSim's is (distance factor 1), so routes are MATSim's
    # not-least-cost ones and congestion compares like for like
    scoring = scoring_params_by_subpopulation(cfg)[0]
    router_factory = network_router_factory(cfg, scoring, scoring.modes, bound_factor=1.0)

    class TimeUp(Exception):
        pass

    class Record(ControllerListener):
        """Per-iteration metrics; and, for the plain Controller (a weave
        records its own timeline), its replanning and scoring as spans."""

        def __init__(self):
            self.rows = []
            self.spans = []
            self.timeline = None

        def on_startup(self, controller):
            self.t0 = time.perf_counter()
            self.epoch = time.time()
            if hasattr(controller, "timeline"):
                self.timeline = controller.timeline = []

        def _span(self, phase, start):
            if self.timeline is None:
                self.spans.append((phase, start, time.time(), os.getpid()))

        def on_iteration_start(self, state):
            self.started = time.time()

        def on_after_replanning(self, state):
            self._span("innovate", self.started)

        def on_after_mobsim(self, state):
            self.simulated = time.time()


        def on_iteration_end(self, state):
            if hasattr(self, "simulated"):  # scoring and travel times
                self._span("score", self.simulated)
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
    common = dict(strategies=strategies(cfg), mobsim=mobsim, params=params,
                  router_factory=router_factory, listeners=[record])
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
    rows = [line.split(",") for line in sims.read_text().splitlines()] if sims.exists() else []
    spans = record.spans if record.timeline is None else record.timeline
    timeline = sorted([phase, a - record.epoch, b - record.epoch, int(pid)]
                      for phase, a, b, pid in spans if a - record.epoch <= TIME_LIMIT)
    Path(out).write_text(json.dumps({**asdict(run), "load_s": load_s, "total_s": total_s,
                                     "persons": len(controller.scenario.population),
                                     "main_pid": os.getpid(), "timeline": timeline,
                                     "iterations": record.rows,
                                     "sims": sorted([float(a) - record.epoch, float(b) - record.epoch,
                                                     int(pid)] for a, b, pid in rows)}))


def run_matsim(run: Run, out: Path, log: Path) -> int:
    """Java MATSim (Hermes) for :data:`TIME_LIMIT` seconds; writes the same
    record as :func:`child` to ``out``. Returns the exit code."""
    from mobslim.config.matsim import MatsimConfig
    from showcase.perf import _java_classpath
    from verification.runner import _hermes_config

    d = OUT / run.slug
    d.mkdir(parents=True, exist_ok=True)
    cfg = _hermes_config(MatsimConfig.load(CONFIG))
    for module in ("qsim", "hermes"):
        cfg.set(module, "flowCapacityFactor", CAPACITY).set(module, "storageCapacityFactor", CAPACITY)
    cfg.set("global", "randomSeed", SEED).set("global", "numberOfThreads", BUDGET)
    cfg.set("controller", "lastIteration", ITERATIONS - 1)
    record = d / "record.csv"
    record.unlink(missing_ok=True)
    cmd = ["java", "-Xmx10g", "-cp", _java_classpath(), "ReplanRun", "--config",
           str(cfg.write(d / "config.xml")), "--output", str(d / "output"), "--record", str(record),
           "--time-limit", str(TIME_LIMIT)]
    for name, weight in MATSIM_STRATEGIES.items():
        cmd += ["--strategy", f"{name}={weight}"]
    t = time.perf_counter()
    with open(log, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    if code != 0 or not record.exists():
        return code or 1
    rows = [dict(zip(("iteration", "wall_s", "start", "end", "executed", "best"), map(float, r.split(","))))
            for r in record.read_text().splitlines()[1:]]
    out.write_text(json.dumps({
        **asdict(run), "total_s": time.perf_counter() - t,
        "iterations": [{"iteration": int(r["iteration"]), "wall_s": r["wall_s"],
                        "score_performed": r["executed"], "score_executed": r["executed"],
                        "score_best": r["best"]} for r in rows],
        "sims": [[r["start"], r["end"], 0] for r in rows]}))
    return 0


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
    from showcase.perf import PerfCase, prepare

    prepare(PerfCase(CASE, "Open Berlin v6.4, 1%, initial plans", open_berlin="1pct-initial",
                     transit=True, xmx="20g"))  # made when missing
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    results["machine"] = machine()
    settings = {"case": CASE, "time_limit_s": TIME_LIMIT, "innovation_rate": INNOVATION,
                "budget": BUDGET, "seed": SEED, "capacity": CAPACITY}
    if results.get("settings") != settings:  # runs made otherwise are not comparable
        results["runs"] = {}
    results["settings"] = settings
    for run in RUNS:
        if only and not any(k in run.slug or k in run.experiment for k in only):
            continue
        out = OUT / f"{run.slug}.json"
        out.unlink(missing_ok=True)
        print(f"{run.experiment}: {run.name} ...", flush=True)
        t = time.perf_counter()
        if run.loop == "matsim":
            code = run_matsim(run, out, OUT / f"{run.slug}.log")
        else:
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
