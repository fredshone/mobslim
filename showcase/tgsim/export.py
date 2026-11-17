"""Animation data for the docs: observed and simulated vehicles of one
minute, for ``replay-anim.js`` (see :mod:`showcase.replay`).

Drawing coordinates are the data's rotated so that traffic runs left to
right: ``u`` along the road (the data's ``y``) and ``v`` across it (minus
the data's ``x``, so the right of the road is at the bottom). A link's
vehicles take its lanes in turn, in the order they entered it; the merge
link's fourth is the on-ramp's added lane. The road beyond the view is
drawn faded, and vehicles are shown on it approaching and leaving at the
free speed, or queued back from the entry line while held out of view.

Two files: ``replay-tgsim-i395-sim.js`` and ``replay-tgsim-i395-obs.js``
(the observed tracks, git-ignored until their licence is checked).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from showcase import replay
from showcase.tgsim.network import (
    ENTRY_NODE,
    EXIT_NODE,
    LANE_WIDTH,
    LINKS,
    Geometry,
)

SPEED = 2  # recorded seconds per second of playback
STRETCH = 3  # across the road, against along it, as drawn


def _draw(xy: np.ndarray) -> np.ndarray:
    """Data ``(x, y)`` -> drawing ``(u, v)``."""
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([xy[:, 1], -xy[:, 0]])


def observed(tr: pd.DataFrame, v: pd.DataFrame, trips: pd.DataFrame, t0: float, times: np.ndarray) -> list[dict]:
    """Recorded positions of the window's vehicles, over their whole track."""
    pos = {}
    for vid, d in tr[tr.id.isin(trips.id) & (tr.t >= t0 - 1)].groupby("id"):
        on = (times >= d.t.iat[0]) & (times <= d.t.iat[-1])
        f = np.flatnonzero(on)
        if len(f):
            xy = np.column_stack([np.interp(times[on], d.t, d.x), np.interp(times[on], d.t, d.y)])
            pos[vid] = dict(zip(f.tolist(), _draw(xy)))
    return replay.segments(pos)


def simulated(geo: Geometry, net, sim: pd.DataFrame, trips: pd.DataFrame, times: np.ndarray) -> list[dict]:
    speed = float(net.freespeed[net.link_index("up")])
    lines = {k: _draw(v) for k, v in geo.lines.items()}
    tails = {k: _draw(v) for k, v in geo.tails.items()}
    pos: dict = defaultdict(dict)
    real = sim[sim.lane.isin(list(LINKS))]
    for k, d in real.groupby("lane"):
        L, lanes = geo.length[k], LINKS[k][2]
        drawn = 3 if k == "merge" else lanes  # the merge link's fourth lane is drawn on its own
        for f, vid, slot, p in replay.place(d, L, L / speed, times, lanes):
            if k == "merge" and slot == 3:
                pos[vid][f] = replay.along(lines["added"], p / L * geo.length["added"])[0]
            else:
                pos[vid][f] = replay.offset(lines[k], p, (slot - (drawn - 1) / 2) * LANE_WIDTH)[0]
    for entry, n in ENTRY_NODE.items():
        d = sim[sim.lane == f"in_{entry}"]
        lanes = 3 if entry == "main" else 1
        for f, vid, slot, p in replay.waiting(d, times, lanes):
            pos[vid][f] = replay.offset(tails[entry], p, (slot - (lanes - 1) / 2) * LANE_WIDTH)[0]
        for r in d.itertuples():  # drawn approaching the entry line
            for f, p in replay.lead_in(tails[entry], speed, r.t_in, times):
                pos[r.id].setdefault(f, replay.along(tails[entry], p)[0])
    last = real.sort_values("t_in").groupby("id").last()
    kind = trips.set_index("id").exit
    for vid, r in last.iterrows():
        x = kind.get(vid)
        if x is not None and LINKS[r.lane][1] == EXIT_NODE[x]:  # left view at the exit line: drawn leaving
            for f, p in replay.coast(tails[x], speed, r.t_out, times):
                pos[vid][f] = replay.along(tails[x], p)[0]
    return replay.segments(pos)


def write(geo: Geometry, net, sim: pd.DataFrame, trips: pd.DataFrame, v: pd.DataFrame, tr: pd.DataFrame,
          minute: int, t0: float, t1: float, attribution: str) -> tuple[Path, Path]:
    frames = int((t1 - t0) * replay.FPS) + 1
    times = t0 + np.arange(frames) / replay.FPS
    lanes = {**{k: LINKS[k][2] for k in LINKS}, "merge": 3, "added": 1}  # as drawn
    roads = [replay.road(_draw(geo.lines[k]), n * LANE_WIDTH, True) for k, n in lanes.items()]
    roads += [replay.road(_draw(geo.tails[k]), n * LANE_WIDTH, False)
              for k, n in (("main", 3), ("ramp", 1), ("through", 3), ("off", 1))]
    k = v.set_index("id").loc[trips.id]
    obs_in = [getattr(r, f"t_{ENTRY_NODE[r.entry]}") for r in k.itertuples()]
    obs_out = [getattr(r, f"t_{EXIT_NODE[r.exit]}") for r in k.itertuples()]
    obs_windows = [(a if np.isfinite(a) else r.t_first, b if np.isfinite(b) else r.t_last)
                   for a, b, r in zip(obs_in, obs_out, k.itertuples())]
    real = sim[sim.lane.isin(list(LINKS))].groupby("id")
    data = {
        "title": f"Minute {minute}, from {t0:,.0f} s", "speed": SPEED, "layout": "stack", "stretch": STRETCH,
        "frames": frames, "t0": t0, "roads": roads,
        "sim": {"label": replay.MOBSIM, "tracks": simulated(geo, net, sim, trips, times),
                "inview": replay.counts(zip(real.t_in.min(), real.t_out.max()), times)},
        "obs_inview": replay.counts(obs_windows, times),
        "note": (f"Playback at {SPEED}× real time. Traffic runs left to right; across the road is drawn "
                 f"{STRETCH}× larger. Grey: the links, as wide as their lanes, which the comparison covers; "
                 "faded: the road beyond the view, where vehicles are drawn approaching and leaving at the free "
                 "speed, or queued back from the entry line while they wait to come into view."),
        "attribution": attribution,
    }
    return replay.write("tgsim-i395", data, observed(tr, v, trips, t0, times))
