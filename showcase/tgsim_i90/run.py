"""The I-90/I-94 pipeline: per run, measure the section and each vehicle's
crossings, then simulate (HermesSim) and compare every minute of it.

Each run is cut into one-minute windows, simulated after a minute's
warm-up, as for I-395. Nothing is fitted: the links follow
``replay.FREEWAY``, and the left lanes' exit ``network.EXIT``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from showcase import replay
from showcase.tgsim_i90 import compare, demand, load, network
from showcase.tgsim_i90.network import GROUPS

OUT = Path(__file__).resolve().parents[1] / "output" / "tgsim_i90"
RESULTS = replay.RESULTS / "tgsim_i90.json"
ATTRIBUTION = ("Observed data: Third Generation Simulation Data (TGSIM) I-90/I-94 Stationary Trajectories, "
               "Federal Highway Administration, U.S. DOT ITS JPO (data.transportation.gov).")

SITES = {"diverge": (1, 3, 5), "weave": (2, 4, 6)}
SHOWN = {"diverge": (3, 6), "weave": (4, 6)}  # (run, minute) animated in the docs
WINDOW = 60.0  # [s] measured
WARM = 60.0  # [s] simulated before it
# openDD and I-395 use 95%, but here 12-14% of tracks on runs 1, 2, 4 and 6 start or end partway
# along the road (and don't join up: not the same vehicle under a new id), so 95% falls inside it
SEEN = 0.85


def site_of(run: int) -> str:
    return next(s for s, rs in SITES.items() if run in rs)


@cache
def section(run: int) -> network.Section:
    """Where traffic comes into and leaves view: the line by which
    ``SEEN`` of each group's vehicles had been seen (those first seen after
    1 s), and the line at which ``SEEN`` were still seen (those last seen
    before the end); the later and the earlier of the two groups'."""
    tr = load.trajectories(run)
    v = tr.groupby("id").agg(t0=("t", "min"), t1=("t", "max"), s0=("s", "first"), s1=("s", "max"),
                             group=("group", "last"))
    end = tr.t.max()
    s_in = max(v[(v.group == g) & (v.t0 > 1)].s0.quantile(SEEN) for g in GROUPS)
    s_out = min(v[(v.group == g) & (v.t1 < end - 1)].s1.quantile(1 - SEEN) for g in GROUPS)
    n = max(2, round((s_out - s_in) / network.LINK_LENGTH))
    centre = {}
    s_lo, s_hi = float(tr.s.quantile(0.001)), float(tr.s.quantile(0.999))
    for g in GROUPS:
        d = tr[tr.group == g]
        bins = np.arange(np.floor(s_lo) - 5, s_hi + 10, 5.0)
        mid = d.groupby(pd.cut(d.s, bins), observed=True).w.median()
        centre[g] = ([float(i.mid) for i in mid.index], mid.round(2).tolist())
    return network.Section(run, round(float(s_in), 1), round(float(s_out), 1), n, centre,
                           round(s_lo, 1), round(s_hi, 1))


@cache
def data(run: int):
    """``(tr, vehicles)``, vehicles cached in ``output/tgsim_i90``."""
    tr, sec = load.trajectories(run), section(run)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"vehicles_{run}.parquet"
    if not p.exists():
        demand.vehicles(tr, sec).to_parquet(p)
    v = pd.read_parquet(p)
    return tr, v


def minutes(run: int) -> list[int]:
    end = load.trajectories(run).t.max()
    return [m for m in range(1, int(end // WINDOW)) if (m + 1) * WINDOW <= end]


@cache
def trips(run: int, m: int) -> pd.DataFrame:
    tr, v = data(run)
    return demand.trips(v, tr, section(run), m * WINDOW, (m + 1) * WINDOW, WARM)


def simulate(run: int, m: int):
    sec = section(run)
    net = network.build(sec)
    t = trips(run, m)
    return t, replay.simulate(net, demand.population(net, t, sec))


def evaluate(run: int) -> pd.DataFrame:
    """Time in view of every measured vehicle (``compare.trip_times``), with ``run`` and ``minute`` columns."""
    return pd.concat([compare.trip_times(sim, t).assign(run=run, minute=m) for m in minutes(run)
                      for t, sim in [simulate(run, m)]], ignore_index=True)


def meta() -> dict:
    runs = {}
    for r in load.runs():
        tr, v = data(r)
        sec = section(r)
        runs[str(r)] = dict(site=site_of(r), duration=round(float(tr.t.max()), 1), vehicles=len(v),
                            groups=v.group.value_counts().to_dict(), s_in=sec.s_in, s_out=sec.s_out, links=sec.n,
                            link_length=round(sec.length, 1), minutes=len(minutes(r)))
    return dict(runs=runs, sites={k: list(v) for k, v in SITES.items()}, shown={k: list(v) for k, v in SHOWN.items()},
                window=WINDOW, warm=WARM, exit=network.EXIT, attribution=ATTRIBUTION)


def animation() -> None:
    from showcase.tgsim_i90 import export

    for site, (r, m) in SHOWN.items():
        tr, v = data(r)
        sec = section(r)
        t0, t1 = m * WINDOW, (m + 1) * WINDOW
        t, sim = simulate(r, m)
        paths = export.write(f"i90-{site}", sec, network.build(sec), sim, t, v, tr, r, m, t0, t1, ATTRIBUTION)
        print("wrote", *paths)


def main() -> None:
    """``python -m showcase tgsim-i90``: everything, from the data to the docs."""
    T = []
    for r in load.runs():
        print(f"run {r} ({site_of(r)}): evaluating {len(minutes(r))} minutes", flush=True)
        T.append(evaluate(r).assign(site=site_of(r)))
    T = pd.concat(T, ignore_index=True)
    head = [replay.result(f"Highway I-90/I-94, {s}", d) for s, d in T.groupby("site", sort=False)]
    groups = [replay.result(f"{s.capitalize()}, {g} lanes", d) for (s, g), d in T.groupby(["site", "group"], sort=False)]
    R = dict(meta=meta(), headline=head, groups=groups)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(R, indent=1, default=str) + "\n")
    print("wrote", RESULTS, *groups, sep="\n")
    animation()
    replay.headline()
