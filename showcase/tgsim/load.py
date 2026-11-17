"""Read the TGSIM I-395 trajectories.

Coordinates are the dataset's own, in metres on its reference image:
traffic runs towards +y, and x grows to the right of the direction of
travel. Columns: ``id``, ``t`` [s from the start of the run], ``x``, ``y``
[m], ``lane`` (-5 left, -4 middle, -3 right, -2 on-ramp and its added lane,
-1 that lane's shoulder, -6 off-ramp), ``v`` [m/s], ``a`` [m/s²],
``length``, ``width`` [m], ``cls``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tgsim"
CSV = DATA / "I395_trajectories.csv"
PARQUET = DATA / "I395_trajectories.parquet"  # converted from the CSV on first use
CLASSES = {1: "Car", 2: "Truck", 3: "Bus", 4: "Automated vehicle"}  # type_most_common
HZ = 10.0


def trajectories() -> pd.DataFrame:
    if not PARQUET.exists():
        if not CSV.exists():
            raise FileNotFoundError(f"no {CSV.name} in {DATA}; see showcase/tgsim/__init__.py")
        d = pd.read_csv(CSV, thousands=",")  # numbers are written with thousands separators
        d.columns = ["id", "t", "x", "y", "lane", "v", "a", "length", "width", "type"]
        d.sort_values(["id", "t"]).reset_index(drop=True).to_parquet(PARQUET)
    d = pd.read_parquet(PARQUET)
    d["cls"] = d.pop("type").map(CLASSES)
    return d
