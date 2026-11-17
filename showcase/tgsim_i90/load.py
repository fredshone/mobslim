"""Read the TGSIM I-90/I-94 Stationary trajectories, in road coordinates.

Each run has its own reference image, so its own coordinates. They are
turned so that ``s`` [m] runs along the road in the direction of travel (the
median vehicle's displacement) from the run's first point, and ``w`` [m]
across it, positive to the right. Columns: ``run``, ``id``, ``t`` [s],
``s``, ``w``, ``lane``, ``v`` [m/s], ``cls``, ``group`` (``left``: lanes
1-3 or 10-12, ``right``: 4-6 or 13-15).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tgsim_i90"
CSV = DATA / "I90_94_stationary_trajectories.csv"
PARQUET = DATA / "trajectories.parquet"  # converted from the CSV on first use
LEFT = {1, 2, 3, 10, 11, 12}
CLASSES = {"small-vehicle": "Small vehicle", "large-vehicle": "Large vehicle"}


def _raw() -> pd.DataFrame:
    if not PARQUET.exists():
        if not CSV.exists():
            raise FileNotFoundError(f"no {CSV.name} in {DATA}; see showcase/tgsim_i90/__init__.py")
        d = pd.read_csv(CSV, thousands=",")
        d.columns = ["id", "t", "x", "y", "lane", "v", "a", "length", "width", "cls", "av", "run"]
        d.sort_values(["run", "id", "t"]).reset_index(drop=True).to_parquet(PARQUET)
    return pd.read_parquet(PARQUET)


@cache
def frame(run: int) -> tuple[np.ndarray, np.ndarray]:
    """``(origin, u)``: the run's road frame in its image coordinates."""
    d = _raw().query("run == @run")
    v = d.groupby("id").agg(x0=("x", "first"), y0=("y", "first"), x1=("x", "last"), y1=("y", "last"))
    u = np.array([(v.x1 - v.x0).median(), (v.y1 - v.y0).median()])
    u /= np.linalg.norm(u)
    s = d.x * u[0] + d.y * u[1]
    k = s.idxmin()
    return np.array([d.x[k], d.y[k]]), u


def to_road(run: int, x, y) -> tuple[np.ndarray, np.ndarray]:
    o, u = frame(run)
    dx, dy = np.asarray(x) - o[0], np.asarray(y) - o[1]
    return dx * u[0] + dy * u[1], dx * u[1] - dy * u[0]  # right of travel: (u_y, -u_x)


@cache
def trajectories(run: int) -> pd.DataFrame:
    d = _raw().query("run == @run").copy()
    d["s"], d["w"] = to_road(run, d.x, d.y)
    d["cls"] = d.cls.map(CLASSES)
    d["group"] = np.where(d.lane.isin(LEFT), "left", "right")
    return d[["run", "id", "t", "s", "w", "lane", "v", "cls", "group"]].reset_index(drop=True)


def runs() -> list[int]:
    return sorted(_raw().run.unique().tolist())
