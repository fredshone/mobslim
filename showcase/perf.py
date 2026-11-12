"""Speed and memory: Java MATSim (QSim, Hermes) against every MobSlim mobsim.

Each case is prepared once into ``showcase/output/perf/<case>/``: a MATSim
config and **routed** plans (MATSim routes its own example scenarios in one
iteration; the synthetic grids are routed by MobSlim and written as MATSim
XML). Both sides then simulate exactly those plans and routes.

Every run is a fresh process, wrapped in ``/usr/bin/time`` for its peak
resident memory (the whole process: interpreter or JVM, scenario, mobsim).

* **MATSim**: ``PerfRun`` (``showcase/java``) runs ``REPEATS`` iterations
  with no replanning and no output files, and times each iteration's mobsim.
  The JVM gets ``-Xmx`` = :data:`JAVA_XMX`.
* **MobSlim**: :func:`child` loads the same files, then runs the mobsim
  ``REPEATS`` times with no event handlers, as the controller does in an
  iteration that is not an output iteration (:func:`~mobslim.mobsim.run_mobsim`):
  the mobsim returns trajectory arrays and, if it can, the link traversal
  sums in MATSim's 15-minute bins, without each link's entry time.

The reported time is the fastest repeat (the JVM's later iterations are
warmed up; MobSlim's first run includes nothing extra). Results go to
``showcase/results/perf.json``; machine-specific, but committed so the docs
build without rerunning.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "output" / "perf"
RESULTS = Path(__file__).parent / "results" / "perf.json"
JAVA_SRC = Path(__file__).parent / "java" / "src" / "PerfRun.java"
JAVA_CLASSES = Path(__file__).parent / "java" / "classes"
JAVA_XMX = "10g"
REPEATS = 3
SLOW = 60.0  # s: a MobSlim run slower than this is not repeated
CPUS = os.cpu_count() or 1


@dataclass(frozen=True)
class PerfCase:
    name: str
    description: str
    scenario: str = ""  # MATSim example (under verification.cases.SCENARIOS) ...
    config: str = "config.xml"
    params: dict = field(default_factory=dict)
    grid: int = 0  # ... or a synthetic grid of this size
    persons: int = 0
    capacity: float = 1.0
    transit: bool = False
    # > 0: plans learned by MATSim (this many iterations of the scenario's own
    # replanning, on ``learn_mobsim``) instead of its iteration-0 plans
    learn_iterations: int = 0
    learn_mobsim: str = "hermes"
    learn_weights: dict = field(default_factory=dict)  # strategyName -> weight while learning


def _cap(f):
    return {("qsim", "flowCapacityFactor"): f, ("qsim", "storageCapacityFactor"): f}


CASES = [
    PerfCase("grid-40", "40x40 grid: 6,560 links, 26,240 car commuters", grid=40, persons=26_240),
    PerfCase("grid-40-congested", "the same at capacity x0.1: queues and spillback everywhere",
             grid=40, persons=26_240, capacity=0.1),
    PerfCase("grid-80", "80x80 grid: 25,920 links, 104,960 car commuters", grid=80, persons=104_960),
    PerfCase("grid-80-congested", "the same at capacity x0.1: queues and spillback everywhere",
             grid=80, persons=104_960, capacity=0.1),
    # learned plans: the shipped iteration-0 plans gridlock the network on every mobsim
    PerfCase("siouxfalls", "MATSim Sioux Falls 2014: 84,110 agents, car and bus (2,040 departures)",
             "siouxfalls-2014", "config_default.xml", transit=True, learn_iterations=10,
             # 30% of agents innovate per iteration (the shipped 3% is tuned for 3,000 iterations)
             learn_weights={"ChangeExpBeta": 0.7, "ReRoute": 0.1, "SubtourModeChoice": 0.1,
                            "TimeAllocationMutator": 0.1}),
]

# label -> (mobsim, MATSim --set overrides | MobSlim params)
MATSIM_VARIANTS = {
    "MATSim QSim": ("qsim", {("qsim", "numberOfThreads"): 1, ("global", "numberOfThreads"): 1}),
    "MATSim Hermes": ("hermes", {("global", "numberOfThreads"): CPUS}),
    # DSim: MATSim's parallel mobsim, from a weekly build (see showcase/dsim.py)
    "MATSim DSim, 1 thread": ("dsim", {("dsim", "threads"): 1}),
    f"MATSim DSim, {CPUS} threads": ("dsim", {("dsim", "threads"): CPUS}),
}
MOBSLIM_VARIANTS = {
    "QSim": ("qsim", {}),
    "AgentSim": ("agentsim", {"blocked": "chain"}),
    "HermesSim": ("hermes", {"backend": "rust"}),
    "PSim, 1 thread": ("psim", {"threads": 1}),
    f"PSim, {CPUS} threads": ("psim", {"threads": CPUS}),
    "DSim, 1 thread": ("dsim", {"threads": 1}),
    f"DSim, {CPUS} threads": ("dsim", {"threads": CPUS}),
}


# ------------------------------------------------------------------ preparing


def _grid_files(case: PerfCase, d: Path) -> None:
    """A MATSim network and routed home-work-home car plans (as in benchmarks.mobsim_sweep)."""
    from mobslim import (
        Plan,
        Population,
        Scenario,
        read_network_xml,
        route_scenario,
    )
    from mobslim.formats.matsim_plans import write_population_xml

    n, spacing = case.grid, 200.0
    nodes = [f'<node id="{i}_{j}" x="{j * spacing}" y="{i * spacing}"/>'
             for i in range(n + 1) for j in range(n + 1)]
    links = []
    for i in range(n + 1):
        for j in range(n + 1):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] > n or b[1] > n:
                    continue
                for u, v in ((a, b), (b, a)):
                    links.append(f'<link id="{u[0]}_{u[1]}-{v[0]}_{v[1]}" from="{u[0]}_{u[1]}" '
                                 f'to="{v[0]}_{v[1]}" length="{spacing}" freespeed="10.0" '
                                 'capacity="600.0" permlanes="1" modes="car"/>')
    (d / "network.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE network SYSTEM '
        '"http://www.matsim.org/files/dtd/network_v1.dtd">\n<network>\n<nodes>\n'
        + "\n".join(nodes) + '\n</nodes>\n<links capperiod="01:00:00">\n' + "\n".join(links)
        + "\n</links>\n</network>\n")
    net = read_network_xml(d / "network.xml")
    rng = random.Random(1)
    # activities on a random link leaving a random node, as MATSim needs a link
    out_links = {}
    for li, f in enumerate(net.from_node.tolist()):
        out_links.setdefault(f, []).append(li)
    plans = {}
    for p in range(case.persons):
        home = rng.randrange(net.n_nodes)
        work = rng.randrange(net.n_nodes - 1)
        work += work >= home
        hl, wl = rng.choice(out_links[home]), rng.choice(out_links[work])
        plans[f"p{p}"] = (Plan.builder()
                          .act("home", int(net.to_node[hl]), link=hl,
                               end_time=6 * 3600 + rng.uniform(0, 2 * 3600))
                          .leg("car").act("work", int(net.to_node[wl]), link=wl, duration=8 * 3600)
                          .leg("car").act("home", int(net.to_node[hl]), link=hl).build())
    scenario = route_scenario(Scenario(net, Population.from_plans(plans)))
    write_population_xml(scenario.population, d / "plans.xml.gz", net)
    f = case.capacity
    (d / "config.xml").write_text(GRID_CONFIG.format(cap=f))


GRID_CONFIG = """<?xml version="1.0" ?>
<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">
<config>
  <module name="global"><param name="randomSeed" value="4711" /></module>
  <module name="network"><param name="inputNetworkFile" value="network.xml" /></module>
  <module name="plans"><param name="inputPlansFile" value="plans.xml.gz" /></module>
  <module name="controller">
    <param name="firstIteration" value="0" /><param name="lastIteration" value="0" />
  </module>
  <module name="qsim">
    <param name="flowCapacityFactor" value="{cap}" />
    <param name="storageCapacityFactor" value="{cap}" />
    <param name="stuckTime" value="10" />
  </module>
  <module name="hermes">
    <param name="flowCapacityFactor" value="{cap}" />
    <param name="storageCapacityFactor" value="{cap}" />
  </module>
  <module name="scoring">
    <parameterset type="scoringParameters">
      <parameterset type="activityParams">
        <param name="activityType" value="home" /><param name="typicalDuration" value="12:00:00" />
      </parameterset>
      <parameterset type="activityParams">
        <param name="activityType" value="work" /><param name="typicalDuration" value="08:00:00" />
      </parameterset>
      <parameterset type="modeParams"><param name="mode" value="car"/></parameterset>
      <parameterset type="modeParams"><param name="mode" value="walk"/></parameterset>
      <parameterset type="modeParams"><param name="mode" value="pt"/></parameterset>
    </parameterset>
  </module>
  <module name="replanning">
    <parameterset type="strategysettings">
      <param name="strategyName" value="KeepLastSelected"/><param name="weight" value="1.0"/>
    </parameterset>
  </module>
</config>
"""


def _set_strategy_weights(cfg, weights: dict) -> None:
    """Set the ``replanning`` strategies' weights by name (each must exist)."""
    if not weights:
        return
    found = set()
    for ps in cfg._module("replanning").findall("parameterset"):
        params = {p.get("name"): p for p in ps.findall("param")}
        name = params.get("strategyName")
        if name is not None and name.get("value") in weights:
            params["weight"].set("value", str(weights[name.get("value")]))
            found.add(name.get("value"))
    if found != set(weights):
        raise ValueError(f"strategies not in the config: {set(weights) - found}")


def prepare(case: PerfCase, force: bool = False) -> Path:
    """``<case>/input/config.xml`` with routed plans; returns its path."""
    from mobslim.config.matsim import MatsimConfig

    d = OUT / case.name / "input"
    cfg_path = d / "config.xml"
    if cfg_path.exists() and not force:
        return cfg_path
    d.mkdir(parents=True, exist_ok=True)
    if case.grid:
        _grid_files(case, d)
        return cfg_path
    from verification.cases import SCENARIOS
    from verification.matsim import run_matsim

    cfg = MatsimConfig.load(SCENARIOS / case.scenario / case.config)
    for (m, p), v in case.params.items():
        cfg.set(m, p, v)
    if case.learn_iterations:
        from verification.runner import _hermes_config

        learn = _hermes_config(cfg) if case.learn_mobsim == "hermes" else cfg.copy()
        learn.set("controller", "lastIteration", case.learn_iterations - 1)
        _set_strategy_weights(learn, case.learn_weights)
        routed = run_matsim(learn, OUT / case.name / "learning", force=force)
    else:
        cfg.set("controller", "lastIteration", 0)
        routed = run_matsim(cfg, OUT / case.name / "routing", force=force)
    cfg.set("plans", "inputPlansFile", str(routed / "output_plans.xml.gz"))
    cfg.write(cfg_path)
    return cfg_path


# ------------------------------------------------------------------ running


def _timed(cmd: list[str], log: Path) -> tuple[int, int]:
    """Run ``cmd`` under /usr/bin/time; (exit code, peak RSS in kB)."""
    rss_file = log.with_suffix(".rss")
    with open(log, "w") as f:
        proc = subprocess.run(["/usr/bin/time", "-f", "%M", "-o", str(rss_file), *cmd],
                              stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    rss = int(rss_file.read_text().split()[-1]) if rss_file.exists() else -1
    return proc.returncode, rss


def _java_classpath() -> str:
    from verification.matsim import classpath

    cp = classpath()
    cls = JAVA_CLASSES / "PerfRun.class"
    if not cls.exists() or cls.stat().st_mtime < JAVA_SRC.stat().st_mtime:
        subprocess.run(["javac", "-d", str(JAVA_CLASSES), "-cp", cp, str(JAVA_SRC)], check=True)
    return f"{JAVA_CLASSES}:{cp}"


def run_matsim(case: PerfCase, cfg_path: Path, label: str) -> dict:
    from mobslim.config.matsim import MatsimConfig
    from verification.runner import _hermes_config

    mobsim, sets = MATSIM_VARIANTS[label]
    cfg = MatsimConfig.load(cfg_path)
    if mobsim == "hermes":
        cfg = _hermes_config(cfg)
    slug = label.split(",")[0].replace(" ", "_").lower() + ("" if "," not in label else
                                                             "_" + label.split()[-2])
    d = OUT / case.name / slug
    d.mkdir(parents=True, exist_ok=True)
    run_cfg = cfg.write(d / "config.xml")
    timings = d / "timings.csv"
    timings.unlink(missing_ok=True)
    if mobsim == "dsim":
        from showcase import dsim

        threads = sets[("dsim", "threads")]
        cmd = dsim.command(run_cfg, d / "output", threads, timings, xmx=JAVA_XMX,
                           sets={"controller.lastIteration": REPEATS - 1})
    else:
        cmd = ["java", f"-Xmx{JAVA_XMX}", "-cp", _java_classpath(), "PerfRun", "--config",
               str(run_cfg), "--output", str(d / "output"), "--timings", str(timings),
               "--set", f"controller.lastIteration={REPEATS - 1}"]
        for (m, p), v in sets.items():
            cmd += ["--set", f"{m}.{p}={v}"]
    t0 = time.perf_counter()
    code, rss = _timed(cmd, d / "run.log")
    wall = time.perf_counter() - t0
    if code != 0 or not timings.exists():
        return {"error": f"exit {code}, see {d / 'run.log'}"}
    runs = [int(line.split(",")[1]) / 1e9 for line in timings.read_text().splitlines()[1:]]
    return {"time_s": min(runs), "runs_s": runs, "peak_rss_mb": rss / 1024, "process_wall_s": wall}


def run_mobslim(case: PerfCase, cfg_path: Path, label: str) -> dict:
    d = OUT / case.name / "mobslim"
    d.mkdir(parents=True, exist_ok=True)
    slug = label.replace(", ", "_").replace(" ", "_").lower()
    out = d / f"{slug}.json"
    out.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "showcase.perf", "child", case.name, label, str(out)]
    t0 = time.perf_counter()
    code, rss = _timed(cmd, d / f"{slug}.log")
    wall = time.perf_counter() - t0
    if code != 0 or not out.exists():
        return {"error": f"exit {code}, see {d / (slug + '.log')}"}
    res = json.loads(out.read_text())
    res.update(peak_rss_mb=rss / 1024, process_wall_s=wall)
    return res


def _status_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise KeyError(field)


def child(case_name: str, label: str, out: str) -> None:
    """One MobSlim run in this process: load, simulate ``REPEATS`` times, write timings."""
    import gc

    import numpy as np

    from mobslim.config.matsim import (
        MatsimConfig,
        mobsim_params,
        scoring_params,
    )
    from mobslim.events import EventsManager
    from mobslim.mobsim import (
        AgentSim,
        AgentSimParams,
        DSim,
        DSimParams,
        HermesSim,
        HermesSimParams,
        PSim,
        PSimParams,
        QSim,
        QSimParams,
        run_mobsim,
    )
    from mobslim.routing.traveltime import DEFAULT_HORIZON
    from mobslim.scoring import CharyparNagelScorer
    from verification.cases import MOBSIMS  # noqa: F401  (registers kernels)
    from verification.runner import mobslim_scenario

    case = next(c for c in CASES if c.name == case_name)
    cfg = MatsimConfig.load(OUT / case.name / "input" / "config.xml")
    kwargs, _ = mobsim_params(cfg)
    t0 = time.perf_counter()
    plans = cfg.input_path("plans", "inputPlansFile")
    scenario = mobslim_scenario(cfg, plans, vehicles="network")
    load_s = time.perf_counter() - t0
    kind, opts = MOBSLIM_VARIANTS[label]
    scorer = CharyparNagelScorer(scoring_params(cfg)[0])
    cls = {"qsim": (QSim, QSimParams), "agentsim": (AgentSim, AgentSimParams),
           "hermes": (HermesSim, HermesSimParams), "psim": (PSim, PSimParams),
           "dsim": (DSim, DSimParams)}[kind]
    sim = cls[0](cls[1](**kwargs, **opts))
    gc.collect()
    Path("/proc/self/clear_refs").write_text("5")  # reset the peak (VmHWM)
    rss0 = _status_kb("VmRSS")
    runs = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        tr = run_mobsim(sim, scenario, EventsManager([]), (900.0, DEFAULT_HORIZON),
                        link_times=False)
        runs.append(time.perf_counter() - t)
        if len(runs) == 1:
            mobsim_peak = (_status_kb("VmHWM") - rss0) / 1024
            stuck = int(np.asarray(tr.stuck).sum())
            # what MATSim's event handlers do during its mobsim, done here from
            # the trajectories afterwards: link travel times and scores
            t = time.perf_counter()
            tr.travel_time_table(scenario.network, 900.0, DEFAULT_HORIZON)
            scorer.score_trajectories(tr)
            post_s = time.perf_counter() - t
        del tr
        gc.collect()
        if runs[-1] > SLOW:
            break
    Path(out).write_text(json.dumps({
        "time_s": min(runs), "runs_s": runs, "post_s": post_s, "load_s": load_s, "rss_before_mobsim_mb": rss0 / 1024,
        "mobsim_peak_mb": mobsim_peak, "persons": len(scenario.population.ids),
        "links": int(scenario.network.n_links), "stuck": stuck}))


# ------------------------------------------------------------------ main


def machine() -> dict:
    model = next((line.split(":", 1)[1].strip() for line in
                  Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "")
    java = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "rust"], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip())
    mem = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemTotal")))
    return {"cpu": model, "cpus": CPUS, "memory_gb": round(mem / 1024 ** 2), "os": platform.platform(),
            "python": platform.python_version(), "java": java, "commit": commit, "dirty": dirty,
            "date": time.strftime("%Y-%m-%d"), "matsim": "2025.0", "xmx": JAVA_XMX, "repeats": REPEATS}


def main(patterns=(), force: bool = False, only=()) -> None:
    """Run the cases matching ``patterns``; with ``only``, just the variants
    whose label contains one of them, kept alongside the case's other results."""
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"cases": {}}
    results["machine"] = machine()
    cases = [c for c in CASES if not patterns or any(p in c.name for p in patterns)]
    for case in cases:
        print(f"== {case.name}", flush=True)
        cfg_path = prepare(case, force)
        old = results["cases"].get(case.name, {}).get("runs", {}) if only else {}
        keep = lambda label: not only or any(o in label for o in only)  # noqa: E731
        rows = dict(old)
        for label in filter(keep, MATSIM_VARIANTS):
            rows[label] = {"family": "matsim", **run_matsim(case, cfg_path, label)}
            print(f"  {label}: {rows[label]}", flush=True)
        for label in (lab for lab in MOBSLIM_VARIANTS if keep(f"MobSlim {lab}")):
            rows[f"MobSlim {label}"] = {"family": "mobslim", **run_mobslim(case, cfg_path, label)}
            print(f"  MobSlim {label}: {rows['MobSlim ' + label]}", flush=True)
        order = list(MATSIM_VARIANTS) + [f"MobSlim {lab}" for lab in MOBSLIM_VARIANTS]
        rows = {k: rows[k] for k in order if k in rows}
        results["cases"][case.name] = {"description": case.description, "runs": rows}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(*sys.argv[2:5])
