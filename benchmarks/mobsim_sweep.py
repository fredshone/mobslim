"""Mobsim scaling sweep: run time and peak memory vs network size and congestion.

The network is a ``size x size`` grid with ``density`` random home-work-home
car commuters per link, so demand grows with the network. Congestion is set
with the capacity factor (flow and storage scaled together, as for a
sampled population): a factor of 0.03 means every link passes and holds 3%
of its nominal vehicles.

Every (grid size, capacity factor, mobsim variant) cell runs in a fresh subprocess so
peak memory is not polluted by earlier cells. Inside the child:

1. build the grid scenario and route every leg at free speed (untimed);
2. reset the process's peak-RSS counter (Linux ``/proc/self/clear_refs``),
   run the mobsim once and record wall time and peak RSS above the
   pre-run RSS (everything the run allocated, Python or not);
3. if the run was quick, repeat it and keep the fastest time.

All runs use ``stuck_time=inf``: no vehicle is ever forced onto a full link,
so congestion shows as queues and spillback, and gridlocks (if any) end the
run with the blocked agents reported as stuck.

    uv run python -m benchmarks.mobsim_sweep --out benchmarks/results/sweep.json
"""

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

GRIDS = (10, 20, 40)  # grid_network(size): (size+1)^2 nodes, 4*size*(size+1) links
DENSITY = 4.0  # persons per link
CAPACITY = (1.0, 0.1, 0.05, 0.03)  # flow = storage capacity factor
WINDOW_H = 2.0  # morning departures spread uniformly over this many hours

VARIANTS = {
    "qsim": ("qsim", {}, False),
    "qsim+events": ("qsim", {}, True),
    "agentsim-poll": ("agentsim", {"blocked": "poll"}, False),
    "agentsim-chain": ("agentsim", {"blocked": "chain"}, False),
    "agentsim-poll+events": ("agentsim", {"blocked": "poll"}, True),
    "agentsim-poll-r5": ("agentsim", {"blocked": "poll", "retry_interval": 5.0}, False),
    "hermes": ("hermes", {"backend": "rust"}, False),
    "hermes+events": ("hermes", {"backend": "rust"}, True),
    "hermes-py": ("hermes", {"backend": "python"}, False),
}


def _status_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise KeyError(field)


def _build(size: int, persons: int, seed: int = 1):
    from random import Random

    from mobslim import Plan, Population, Scenario, grid_network, route_scenario

    net = grid_network(size, spacing=200.0, capacity=600.0)
    rng = Random(seed)
    n = net.n_nodes
    plans = {}
    for i in range(persons):
        home = rng.randrange(n)
        work = rng.randrange(n - 1)
        work += work >= home
        plans[i] = (
            Plan.builder()
            .act("home", home, end_time=6 * 3600 + rng.uniform(0, WINDOW_H * 3600))
            .leg("car")
            .act("work", work, duration=8 * 3600)
            .leg("car")
            .act("home", home)
            .build()
        )
    return route_scenario(Scenario(net, Population.from_plans(plans)))


CACHE = Path(__file__).resolve().parent / "results" / "sweep_cache"


def _cached(size: int, persons: int):
    """The routed scenario, pickled so each child does not reroute it."""
    import pickle

    path = CACHE / f"grid{size}_p{persons}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    scenario = _build(size, persons)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(scenario))
    return scenario


def child(size: int, persons: int, capacity: float, variant: str) -> dict:
    import numpy as np

    from mobslim import (
        AgentSim,
        AgentSimParams,
        HermesSim,
        HermesSimParams,
        QSim,
        QSimParams,
    )
    from mobslim.events import EventLog, EventsManager

    kind, opts, with_events = VARIANTS[variant]
    scenario = _cached(size, persons)
    common = dict(stuck_time=math.inf, flow_capacity_factor=capacity,
                  storage_capacity_factor=capacity)
    if kind == "qsim":
        sim = QSim(QSimParams(**common, **opts))
    elif kind == "hermes":
        sim = HermesSim(HermesSimParams(**common, **opts))
    else:
        sim = AgentSim(AgentSimParams(**common, **opts))

    def run():
        log = EventLog() if with_events else None
        tr = sim.run(scenario, EventsManager([log] if log else []))
        return tr, log

    gc.collect()
    Path("/proc/self/clear_refs").write_text("5")  # reset VmHWM
    rss0 = _status_kb("VmRSS")
    t0 = time.perf_counter()
    tr, log = run()
    first = time.perf_counter() - t0
    peak = _status_kb("VmHWM") - rss0
    n_events = len(log.events) if log else None
    times = [first]
    del log
    budget = 3.0
    while sum(times) < budget and len(times) < 5:
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)

    # congestion indicators
    free = np.array([leg.route.travel_time if leg.route else np.nan
                     for plan in scenario.population.selected_plans()
                     for leg in plan.legs])  # free-speed time per leg
    tt = tr.leg_arrival - tr.leg_departure
    ok = np.isfinite(tt) & np.isfinite(free) & (free > 0)
    return {
        "size": size,
        "nodes": scenario.network.n_nodes,
        "links": scenario.network.n_links,
        "persons": persons,
        "capacity": capacity,
        "variant": variant,
        "time_s": min(times),
        "first_s": first,
        "runs": len(times),
        "peak_rss_mb": peak / 1024,
        "link_traversals": int(np.isfinite(tr.link_enter).sum()),
        "events": n_events,
        "stuck": int(np.asarray(tr.stuck).sum()),
        "delay_ratio": float(tt[ok].sum() / free[ok].sum()) if ok.any() else None,
        "last_arrival_h": float(np.nanmax(tr.leg_arrival)) / 3600,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=4)
    ap.add_argument("--out", default="benchmarks/results/mobsim_sweep.json")
    ap.add_argument("--grids", type=int, nargs="*", default=GRIDS)
    ap.add_argument("--density", type=float, default=DENSITY)
    ap.add_argument("--capacity", type=float, nargs="*", default=CAPACITY)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--timeout", type=float, default=1200)
    args = ap.parse_args()
    if args.child:
        size, persons, capacity, variant = args.child
        print(json.dumps(child(int(size), int(persons), float(capacity), variant)))
        return

    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["size"], r["persons"], r["capacity"], r["variant"]) for r in rows}
    for size in args.grids:
        links = 4 * size * (size + 1)
        persons = int(round(args.density * links))
        for cap in args.capacity:
            for variant in args.variants:
                if (size, persons, cap, variant) in done:
                    continue
                cmd = [sys.executable, "-m", "benchmarks.mobsim_sweep",
                       "--child", str(size), str(persons), str(cap), variant]
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=args.timeout, check=True)
                    row = json.loads(p.stdout.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    row = {"size": size, "persons": persons, "capacity": cap, "variant": variant,
                           "links": links, "timeout": args.timeout}
                except subprocess.CalledProcessError as e:
                    print(e.stderr, file=sys.stderr)
                    raise
                rows.append(row)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rows, indent=1))
                print(f"grid {size:3d} persons {persons:6d} cap {cap:5.2f} {variant:22s} "
                      f"{row.get('time_s', float('nan')):8.2f}s "
                      f"{row.get('peak_rss_mb', float('nan')):8.1f}MB "
                      f"delay x{row.get('delay_ratio') or float('nan'):.2f} "
                      f"stuck {row.get('stuck')}", flush=True)


if __name__ == "__main__":
    main()
