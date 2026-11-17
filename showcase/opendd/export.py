"""Animation data for the docs: observed and simulated vehicles of one
recording, for ``replay-anim.js`` (see :mod:`showcase.replay`).

Drawing coordinates are the map's (metres, north up). The in-view lanes are
drawn as roads; the lanes beyond each view edge are drawn faded, up to
``TAIL`` metres, and vehicles are shown on them approaching and leaving at
the free speed, or queued back from the edge while held out of view.

Two files, so the observed tracks (openDD, CC BY-ND 4.0) stay out of the
repository: ``replay-opendd-<site>-sim.js`` and ``replay-opendd-<site>-obs.js``
(git-ignored).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from showcase import replay
from showcase.opendd.load import LaneGraph
from showcase.opendd.network import entries, exits, in_view

LANE_WIDTH = 3.2  # [m], for drawing only
TAIL = 40.0  # [m] of lane drawn beyond the view at each entry and exit
SPEED = 4  # recorded seconds per second of playback


def _trim(line: np.ndarray, L: float) -> np.ndarray:
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    return np.vstack([line[cum < L], replay.along(line, min(L, cum[-1]))])


def tails(g: LaneGraph) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per entry lane, the road back out of view (from the view edge,
    backwards); per exit lane, the road onward: up to ``TAIL`` metres."""
    view = set(in_view(g))

    def follow(lane: int, nxt, reverse: bool) -> np.ndarray:
        geo = g.geometry[g.index(lane)]
        pts, cur, L = [geo[0] if reverse else geo[-1]], lane, 0.0
        while L < TAIL:
            cands = [i for i in nxt[cur] if i not in view]
            if not cands:
                break
            cur = max(cands, key=lambda i: g.length[g.index(i)])
            geo = g.geometry[g.index(cur)]
            pts += list(geo[::-1][1:] if reverse else geo[1:])
            L += g.length[g.index(cur)]
        return _trim(np.array(pts), TAIL)

    return ({e: follow(e, g.predecessors, True) for e in entries(g)},
            {x: follow(x, g.successors, False) for x in exits(g)})


def observed(tr: pd.DataFrame, ids, times: np.ndarray) -> list[dict]:
    """Recorded positions of the vehicles ``ids``, resampled to the frames, over their whole track."""
    pos = {}
    for vid, v in tr[tr.id.isin(ids)].groupby("id"):
        on = (times >= v.t.iat[0]) & (times <= v.t.iat[-1])
        f = np.flatnonzero(on)
        if len(f):
            xy = np.column_stack([np.interp(times[on], v.t, v.x), np.interp(times[on], v.t, v.y)])
            pos[vid] = dict(zip(f.tolist(), xy))
    return replay.segments(pos)


def simulated(g: LaneGraph, net, sim: pd.DataFrame, trips: pd.DataFrame, times: np.ndarray) -> list[dict]:
    view = set(in_view(g))
    ins, outs = tails(g)
    speed = {i: float(net.freespeed[net.link_index(i)]) for i in view}
    pos: dict = defaultdict(dict)
    s = sim.assign(virtual=sim.lane.astype(str).str.startswith("in"))
    for lane, d in s[~s.virtual].groupby("lane"):
        k = g.index(int(lane))
        geo, L = g.geometry[k], g.length[k]
        for f, vid, _, p in replay.place(d, L, L / speed[int(lane)], times):
            pos[vid][f] = replay.along(geo, p)[0]
    for lane, d in s[s.virtual].groupby("lane"):
        tail = ins[int(str(lane)[2:])]
        for f, vid, _, p in replay.waiting(d, times):
            pos[vid][f] = replay.along(tail, p)[0]
    first = s[~s.virtual].sort_values("t_in").groupby("id").first()
    last = s[~s.virtual].sort_values("t_in").groupby("id").last()
    for r in trips[trips.edge].itertuples():  # came into view at an edge: drawn approaching it
        lane = int(r.route[0])
        if r.id in first.index:
            t_arrive = float(s[s.virtual & (s.id == r.id)].t_in.min()) if lane in ins else first.t_in[r.id]
            for f, p in replay.lead_in(ins[lane], speed[lane], t_arrive, times):
                pos[r.id].setdefault(f, replay.along(ins[lane], p)[0])
    for vid, r in last.iterrows():  # left view at the end of an exit lane: drawn leaving
        lane = int(r.lane)
        if lane in outs:
            for f, p in replay.coast(outs[lane], speed[lane], r.t_out, times):
                pos[vid][f] = replay.along(outs[lane], p)[0]
    return replay.segments(pos)


def write(g: LaneGraph, rec: str, net, sim: pd.DataFrame, trips: pd.DataFrame, links: pd.DataFrame,
          tr: pd.DataFrame, attribution: str) -> tuple[Path, Path]:
    t_end = float(tr.t.max())
    frames = int(t_end * replay.FPS) + 1
    times = np.arange(frames) / replay.FPS
    view = in_view(g)
    ins, outs = tails(g)
    roads = ([replay.road(g.geometry[g.index(i)], LANE_WIDTH, True) for i in view]
             + [replay.road(t, LANE_WIDTH, False) for t in (*ins.values(), *outs.values())])
    on = links[links.lane.isin(view) & links.id.isin(trips.id)].groupby("id")  # observed: on the in-view lanes
    sim_view = sim[sim.lane.isin(view)].groupby("id")
    data = {
        "title": f"Recording {rec}", "speed": SPEED, "layout": "side", "stretch": 1, "frames": frames,
        "roads": roads,
        "sim": {"label": replay.MOBSIM, "tracks": simulated(g, net, sim, trips, times),
                "inview": replay.counts(zip(sim_view.t_in.min(), sim_view.t_out.max()), times)},
        "obs_inview": replay.counts(zip(on.t_in.min(), on.t_out.max()), times),
        "note": (f"Playback at {SPEED}× real time. Grey: the lanes in the camera's view, which the comparison "
                 "covers; faded: the road beyond it, where vehicles are drawn approaching and leaving at the "
                 "free speed, or queued back from the edge while they wait to come into view."),
        "attribution": attribution,
    }
    return replay.write(f"opendd-{g.site}", data, observed(tr, trips.id, times))
