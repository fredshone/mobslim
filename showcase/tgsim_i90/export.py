"""Animation data for the docs: observed and simulated vehicles of one
minute of one run, for ``replay-anim.js`` (as I-395's).

Drawing coordinates: ``u`` along the road (``s``) and ``v`` across it
(minus ``w``, so the right of the road is at the bottom). Simulated
vehicles are placed as for I-395, each in its own group's lanes; the exit
links just beyond the view are drawn as part of the road beyond it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from showcase import replay
from showcase.tgsim_i90.network import (
    GROUPS,
    LANE_WIDTH,
    LANES,
    OUT,
    TAIL,
    Section,
)

SPEED = 2  # recorded seconds per second of playback
STRETCH = 2  # across the road, against along it, as drawn
_LINK = re.compile(r"^(left|right)(\d+)$")


def _line(sec: Section, group: str, s0: float, s1: float) -> np.ndarray:
    """The group's centreline from ``s0`` to ``s1`` in drawing coordinates."""
    s = np.linspace(s0, s1, max(2, int(abs(s1 - s0) / 5) + 1))
    return np.column_stack([s, -sec.w_at(group, s)])


def observed(tr: pd.DataFrame, trips: pd.DataFrame, t0: float, times: np.ndarray) -> list[dict]:
    pos = {}
    for vid, d in tr[tr.id.isin(trips.id) & (tr.t >= t0 - 1)].groupby("id"):
        on = (times >= d.t.iat[0]) & (times <= d.t.iat[-1])
        f = np.flatnonzero(on)
        if len(f):
            pos[vid] = dict(zip(f.tolist(), np.column_stack([np.interp(times[on], d.t, d.s),
                                                             -np.interp(times[on], d.t, d.w)])))
    return replay.segments(pos)


def simulated(sec: Section, net, sim: pd.DataFrame, trips: pd.DataFrame, times: np.ndarray) -> list[dict]:
    group = trips.set_index("id").group
    speed = float(net.freespeed[net.link_index("left0")])
    lo, hi = max(sec.s_lo, sec.s_in - TAIL), min(sec.s_hi, sec.s_out + TAIL)
    tails_in = {g: _line(sec, g, sec.s_in, lo) for g in GROUPS}  # back from the view edge
    tails_out = {g: _line(sec, g, sec.s_out, hi) for g in GROUPS}  # on from it
    pos: dict = defaultdict(dict)
    for lane, d in sim.groupby("lane"):
        m = _LINK.match(str(lane))
        if m:
            g, k = m.group(1), int(m.group(2))
            line = _line(sec, g, sec.cuts[k], sec.cuts[k + 1])
            for f, vid, slot, p in replay.place(d, sec.length, sec.length / speed, times, LANES):
                pos[vid][f] = replay.offset(line, p, (slot - (LANES - 1) / 2) * LANE_WIDTH)[0]
        elif str(lane).startswith("out_"):  # the exit link, just beyond the view
            g = str(lane)[4:]
            for f, vid, slot, p in replay.place(d, OUT["length"], 1e-3, times, LANES):
                pos[vid][f] = replay.offset(tails_out[g], p, (slot - (LANES - 1) / 2) * LANE_WIDTH)[0]
        elif str(lane).startswith("in_"):  # held out of view: queued back from the edge, and approaching it
            g = str(lane)[3:]
            for f, vid, slot, p in replay.waiting(d, times, LANES):
                pos[vid][f] = replay.offset(tails_in[g], p, (slot - (LANES - 1) / 2) * LANE_WIDTH)[0]
            for r in d.itertuples():
                for f, p in replay.lead_in(tails_in[g], speed, r.t_in, times):
                    pos[r.id].setdefault(f, replay.along(tails_in[g], p)[0])
    out = sim[sim.lane.astype(str).str.startswith("out_")]
    for r in out.itertuples():  # drawn leaving beyond the exit link
        g = group.get(r.id, "right")
        for f, p in replay.coast(tails_out[g], speed, r.t_out, times):
            pos[r.id][f] = replay.along(tails_out[g], p + OUT["length"])[0]
    return replay.segments(pos)


def write(key: str, sec: Section, net, sim: pd.DataFrame, trips: pd.DataFrame, v: pd.DataFrame, tr: pd.DataFrame,
          run: int, minute: int, t0: float, t1: float, attribution: str) -> tuple[Path, Path]:
    frames = int((t1 - t0) * replay.FPS) + 1
    times = t0 + np.arange(frames) / replay.FPS
    lo, hi = max(sec.s_lo, sec.s_in - TAIL), min(sec.s_hi, sec.s_out + TAIL)
    roads = [replay.road(_line(sec, g, sec.s_in, sec.s_out), LANES * LANE_WIDTH, True) for g in GROUPS]
    roads += [replay.road(_line(sec, g, a, b), LANES * LANE_WIDTH, False)
              for g in GROUPS for a, b in ((lo, sec.s_in), (sec.s_out, hi))]
    k = v.set_index("id").loc[trips.id]
    obs_windows = [(r.t_0 if np.isfinite(r.t_0) else r.t_first,
                    getattr(r, f"t_{sec.n}") if np.isfinite(getattr(r, f"t_{sec.n}")) else r.t_last)
                   for r in k.itertuples()]
    real = sim[sim.lane.astype(str).str.match(_LINK)].groupby("id")
    data = {
        "title": f"Run {run}, minute {minute}, from {t0:,.0f} s", "speed": SPEED, "layout": "stack",
        "stretch": STRETCH, "frames": frames, "t0": t0, "roads": roads,
        "sim": {"label": replay.MOBSIM, "tracks": simulated(sec, net, sim, trips, times),
                "inview": replay.counts(zip(real.t_in.min(), real.t_out.max()), times)},
        "obs_inview": replay.counts(obs_windows, times),
        "note": (f"Playback at {SPEED}× real time. Traffic runs left to right, the left lanes at the top; across "
                 f"the road is drawn {STRETCH}× larger. Grey: the links in view, which the comparison covers; "
                 "faded: the road beyond it, where vehicles are drawn approaching and leaving at the free speed, "
                 "held on each group's exit link, or queued back from the entry line while they wait to come "
                 "into view."),
        "attribution": attribution,
    }
    return replay.write(f"tgsim-{key}", data, observed(tr, trips, t0, times))
