"""Compare a simulated window with what was observed in it: per measured
vehicle seen crossing the entry line and the exit line, its time in view,
as for I-395 (``showcase.tgsim.compare``). The simulation itself is
:func:`showcase.replay.simulate` (HermesSim)."""

from __future__ import annotations

import re

import pandas as pd

_LINK = re.compile(r"^(?:left|right)(\d+)$")


def trip_times(sim: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Measured vehicles: ``group``, ``t_dep``, observed and simulated ``t_exit``."""
    t = trips[trips.measured].copy()
    real = sim[sim.lane.astype(str).str.match(_LINK)]
    t["t_exit_sim"] = t.id.map(real.groupby("id").t_out.max())
    return t[["id", "group", "t_dep", "t_exit", "t_exit_sim"]]
