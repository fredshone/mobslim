"""The TGSIM pipeline: per-vehicle crossings, then every minute simulated
(HermesSim) and compared.

The two hours are cut into one-minute windows, each simulated on its own
after a minute's warm-up (``demand.trips``), so the measured minute has the
simulation's own queue rather than a copy of the observed one. Nothing is
fitted: the links follow ``replay.FREEWAY``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pandas as pd

from showcase import replay
from showcase.tgsim import compare, demand, load, network

OUT = Path(__file__).resolve().parents[1] / "output" / "tgsim"
RESULTS = replay.RESULTS / "tgsim.json"
ATTRIBUTION = ("Observed data: Third Generation Simulation Data (TGSIM) I-395 Trajectories, "
               "Federal Highway Administration, U.S. DOT ITS JPO (data.transportation.gov).")

WINDOW = 60.0  # [s] measured
WARM = 60.0  # [s] simulated before it
MINUTES = list(range(1, 120))  # minute 0 has no warm-up
SHOWN = 31  # the minute animated in the docs (congested)


@cache
def data():
    """``(tr, vehicles, geometry)``, vehicles cached in ``output/tgsim``."""
    tr = load.trajectories()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "vehicles.parquet"
    if not p.exists():
        demand.vehicles(tr).to_parquet(p)
    v = pd.read_parquet(p)
    return tr, v, network.geometry(tr)


def window(m: int) -> tuple[float, float]:
    return m * WINDOW, (m + 1) * WINDOW


def view_lines() -> dict:
    """The measurements behind ``network.LINES``."""
    tr, v, _ = data()
    main, ramp = v[v.entry == "main"], v[v.entry == "ramp"]
    off, thru = v[v.exit == "off"], v[v.exit == "through"]
    o = tr[tr.id.isin(off.id)]
    q = lambda s, p: round(float(s.quantile(p)), 1)  # noqa: E731
    return {"A: 95% of freeway vehicles first seen by": q(main.y_first, 0.95),
            "M: 95% of on-ramp vehicles first seen by": q(ramp.y_first, 0.95),
            "D: off-ramp vehicles start to enter its lane (5%)": q(o[o.lane == -6].groupby("id").y.min(), 0.05),
            "Q: 99% of the added lane's track points before": q(tr[tr.lane == -2].y, 0.99),
            "E: 95% of freeway vehicles still seen at": q(thru.y_last, 0.05),
            "X: 95% of off-ramp vehicles still seen at": q(off.y_last, 0.05)}


@cache
def trips(m: int) -> pd.DataFrame:
    tr, v, _ = data()
    t0, t1 = window(m)
    return demand.trips(v, tr, t0, t1, WARM)


def _simulate(m: int, net):
    t = trips(m)
    return t, replay.simulate(net, demand.population(net, t))


def evaluate(minutes: list[int]) -> pd.DataFrame:
    """Time in view of every measured vehicle (``compare.trip_times``), with a ``minute`` column."""
    _, _, geo = data()
    net = network.build(geo)
    return pd.concat([compare.trip_times(sim, t).assign(minute=m) for m in minutes
                      for t, sim in [_simulate(m, net)]], ignore_index=True)


def meta() -> dict:
    tr, v, geo = data()
    return dict(vehicles=len(v), duration=round(float(tr.t.max() - tr.t.min()), 1), lines=network.LINES,
                view_lines=view_lines(), lengths={k: round(x, 1) for k, x in geo.length.items()},
                lanes={k: lanes for k, (_, _, lanes) in network.LINKS.items()},
                capacity_lanes={k: network.CAPACITY_LANES.get(k, n) for k, (_, _, n) in network.LINKS.items()},
                minutes=len(MINUTES), shown=SHOWN, window=WINDOW, warm=WARM, attribution=ATTRIBUTION)


def animation() -> None:
    from showcase.tgsim import export

    tr, v, geo = data()
    t0, t1 = window(SHOWN)
    net = network.build(geo)
    t, sim = _simulate(SHOWN, net)
    paths = export.write(geo, net, sim, t, v, tr, SHOWN, t0, t1, ATTRIBUTION)
    print("wrote", *paths)


def main() -> None:
    """``python -m showcase tgsim``: everything, from the data to the docs."""
    print("evaluating", len(MINUTES), "minutes")
    tt = evaluate(MINUTES)
    R = dict(meta=meta(), headline=[replay.result("Highway I-395", tt)])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(R, indent=1) + "\n")
    print("wrote", RESULTS, R["headline"])
    animation()
    replay.headline()
