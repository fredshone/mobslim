"""Compare a simulated recording with what was observed in it: per vehicle
seen coming into view and leaving it, its time in view.

The simulation itself is :func:`showcase.replay.simulate` (HermesSim).
"""

from __future__ import annotations

import pandas as pd


def _real(sim: pd.DataFrame) -> pd.DataFrame:
    return sim[~sim.lane.astype(str).str.startswith("in")]


def trip_times(sim: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Vehicles seen coming into and leaving view: ``t_dep``, observed and
    simulated ``t_exit``."""
    t = trips[trips.edge & trips.t_exit.notna()].copy()
    t["t_exit_sim"] = t.id.map(_real(sim).groupby("id").t_out.max())
    return t[["id", "t_dep", "t_exit", "t_exit_sim"]].dropna()
