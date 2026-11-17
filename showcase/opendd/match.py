"""Match each tracked vehicle to a path of lanes; lane enter and exit times.

1. **Viterbi** at 10 Hz over the lanes plus an ``off`` state (not on any
   lane). A sample's cost on a lane is its squared distance to the
   centreline (``SIGMA``) plus a heading penalty (``KAPPA``). Moves are to
   the same lane, a successor, or a successor's successor (a lane passed
   between two samples), and leaving a lane before its end or joining the
   next one past its start costs extra.
2. The lane path is the run of states with leading and trailing ``off``
   samples trimmed (the vehicle enters or leaves the map's lanes in view).
3. **Times**: every frame (29.97 Hz) is projected onto the path near its
   lane, giving distance along the path; its running maximum (vehicles do
   not reverse) is interpolated at each lane boundary. A vehicle's first and
   last lanes are usually partial: it came into view, or left it, part way
   along.

A vehicle is rejected, with the reason, when it hardly moves (parked), is
off the lanes for most of its track or for more than ``MAX_OFF`` s in the
middle (footpaths, car parks), or its path does not follow the lane graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from showcase.opendd.load import LaneGraph

SIGMA = 1.0  # [m] lateral spread around a centreline
KAPPA = 4.0  # heading penalty weight: KAPPA * (1 - cos(error))
MAX_D = 3.5  # [m] farther than this from a lane: not on it
OFF_COST = 0.5 * (MAX_D / SIGMA) ** 2  # an off-lane sample costs as much as one MAX_D away
SWITCH = 3.0  # cost of moving between off and a lane
EARLY = 1.0  # per metre of lane left behind (or skipped at the start of the next)
SLACK = 2.5  # [m] free tolerance on that
SUB = 3  # Viterbi on every SUB-th frame (10 Hz)
MAX_OFF = 2.0  # [s] longest gap off the lanes inside a track
MIN_MOVE = 5.0  # [m] a vehicle moving less than this in view is parked
MIN_ON = 0.5  # share of samples that must be on a lane


@dataclass
class Dense:
    """Every lane resampled every ``step`` metres, lanes contiguous."""

    xy: np.ndarray
    lane: np.ndarray  # lane index
    s: np.ndarray  # distance along the lane [m]
    heading: np.ndarray  # [rad]
    start: np.ndarray  # first dense point of each lane (len n_lanes + 1)


def densify(g: LaneGraph, step: float = 0.25) -> Dense:
    xy, lane, s, hd, start = [], [], [], [], [0]
    for k, geo in enumerate(g.geometry):
        seg = np.diff(geo, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        ss = np.linspace(0, cum[-1], max(2, int(np.ceil(cum[-1] / step)) + 1))
        j = np.clip(np.searchsorted(cum, ss, side="right") - 1, 0, len(seg) - 1)
        xy.append(geo[j] + seg[j] * ((ss - cum[j]) / seg_len[j])[:, None])
        hd.append(np.arctan2(seg[j, 1], seg[j, 0]))
        lane.append(np.full(len(ss), k))
        s.append(ss)
        start.append(start[-1] + len(ss))
    return Dense(np.vstack(xy), np.concatenate(lane), np.concatenate(s), np.concatenate(hd),
                 np.array(start))


def _nearest(dense: Dense, xy: np.ndarray, lanes=None):
    """Per point and lane: distance, s and lane heading at the nearest dense point."""
    lanes = range(len(dense.start) - 1) if lanes is None else lanes
    n, m = len(xy), len(lanes)
    d, s, h = np.full((n, m), np.inf), np.zeros((n, m)), np.zeros((n, m))
    for c, k in enumerate(lanes):
        a, b = dense.start[k], dense.start[k + 1]
        d2 = ((xy[:, None, :] - dense.xy[None, a:b]) ** 2).sum(2)
        j = d2.argmin(1)
        d[:, c], s[:, c], h[:, c] = np.sqrt(d2[np.arange(n), j]), dense.s[a + j], dense.heading[a + j]
    return d, s, h


def _transitions(g: LaneGraph) -> np.ndarray:
    """Base move costs between states (lanes, then ``off``); inf = not allowed."""
    k = len(g.ids)
    t = np.full((k + 1, k + 1), np.inf)
    np.fill_diagonal(t, 0.0)
    for a, i in enumerate(g.ids):
        for j in g.successors[i]:
            b = g.index(j)
            t[a, b] = 0.0
            for j2 in g.successors[j]:
                t[a, g.index(j2)] = min(t[a, g.index(j2)], 1.0)
    t[k, :k] = t[:k, k] = SWITCH
    return t


def viterbi(g: LaneGraph, dense: Dense, trans: np.ndarray, xy, heading):
    """Most likely state per sample (lane index, or ``len(g.ids)`` for off)."""
    k = len(g.ids)
    d, s, h = _nearest(dense, xy)
    emit = np.full((len(xy), k + 1), OFF_COST)
    err = 1 - np.cos(heading[:, None] - h)
    emit[:, :k] = np.where(d < MAX_D, 0.5 * (d / SIGMA) ** 2 + KAPPA * err, np.inf)
    leave = EARLY * np.maximum(0, g.length[None] - s - SLACK)  # lane left this far before its end
    join = EARLY * np.maximum(0, s - SLACK)  # lane joined this far after its start
    leave = np.hstack([leave, np.zeros((len(xy), 1))])
    join = np.hstack([join, np.zeros((len(xy), 1))])
    off = ~np.eye(k + 1, dtype=bool)
    cost = emit[0].copy()
    back = np.zeros((len(xy), k + 1), dtype=np.int16)
    for i in range(1, len(xy)):
        m = cost[:, None] + trans + off * (leave[i - 1][:, None] + join[i][None, :])
        back[i] = m.argmin(0)
        cost = m[back[i], np.arange(k + 1)] + emit[i]
    path = np.empty(len(xy), dtype=int)
    path[-1] = cost.argmin()
    for i in range(len(xy) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, np.isfinite(cost.min())


def _fill_skips(g: LaneGraph, lanes: list[int]) -> list[int] | None:
    """Insert lanes passed between two samples; None if the path breaks the graph."""
    out = [lanes[0]]
    for b in lanes[1:]:
        a = out[-1]
        if b in g.successors[a]:
            out.append(b)
            continue
        via = [m for m in g.successors[a] if b in g.successors[m]]
        if not via:
            return None
        out += [min(via, key=lambda m: g.length[g.index(m)]), b]
    return out


def match_vehicle(g: LaneGraph, dense: Dense, trans: np.ndarray, v: pd.DataFrame):
    """``(rows, reason)``: one row per lane on the vehicle's path, or a rejection reason."""
    xy = v[["x", "y"]].to_numpy()
    t = v.t.to_numpy()
    if np.linalg.norm(xy.max(0) - xy.min(0)) < MIN_MOVE:
        return None, "parked"
    sub = np.arange(0, len(v), SUB)
    if sub[-1] != len(v) - 1:
        sub = np.append(sub, len(v) - 1)
    path, ok = viterbi(g, dense, trans, xy[sub], v.angle.to_numpy()[sub])
    off = len(g.ids)
    on = path != off
    if not ok or on.mean() < MIN_ON:
        return None, "off lanes"
    first, last = np.flatnonzero(on)[[0, -1]]
    inner = path[first:last + 1]
    gaps = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], inner == off, [0]]))))[::2]
    if len(gaps) and gaps.max() * SUB / 29.97 > MAX_OFF:
        return None, "off lanes mid-track"
    inner = inner[inner != off]
    runs = [int(x) for i, x in enumerate(inner) if i == 0 or x != inner[i - 1]]
    lanes = _fill_skips(g, [g.ids[x] for x in runs])
    if lanes is None:
        return None, "breaks the lane graph"
    rows = _times(g, dense, lanes, xy, t, sub, path, first, last)
    return (rows, None) if rows else (None, "no progress along lanes")


def _times(g, dense, lanes, xy, t, sub, path, first, last):
    """Enter and exit time of each lane on ``lanes`` from every frame."""
    idx = [g.index(i) for i in lanes]
    length = g.length[idx]
    offset = np.concatenate([[0], np.cumsum(length)])
    # route position of each Viterbi sample (skipped lanes are never a sample's lane)
    pos, p = np.zeros(len(sub), dtype=int), 0
    for i in range(len(sub)):
        if first <= i <= last and path[i] != len(g.ids):
            while idx[p] != path[i]:
                p += 1
        pos[i] = p
    frame_pos = pos[np.clip(np.searchsorted(sub, np.arange(len(xy)), side="right") - 1, 0, len(sub) - 1)]
    a, b = sub[first], sub[last] + 1  # frames from the first to the last on-lane sample
    S = np.full(len(xy), np.nan)
    for p in np.unique(frame_pos[a:b]):
        near = [q for q in (p - 1, p, p + 1) if 0 <= q < len(idx)]
        fr = np.arange(a, b)[frame_pos[a:b] == p]
        d, s, _ = _nearest(dense, xy[fr], [idx[q] for q in near])
        c = d.argmin(1)
        S[fr] = offset[np.array(near)[c]] + s[np.arange(len(fr)), c]
    fr = np.arange(a, b)
    S, tt = np.maximum.accumulate(S[fr]), t[fr]

    def when(x):
        i = np.searchsorted(S, x)
        if i == 0:  # first seen at (or just past) the boundary
            return tt[0]
        if i >= len(S):
            return np.nan
        return tt[i - 1] + (tt[i] - tt[i - 1]) * (x - S[i - 1]) / (S[i] - S[i - 1])

    # Viterbi can put a sample at either end on a lane the vehicle never
    # reaches (just short of it): keep the lanes it was actually on
    keep = [m for m in range(len(lanes)) if offset[m] < S[-1] and S[0] < offset[m + 1]]
    if not keep:
        return None
    lanes, offset, length = [lanes[m] for m in keep], offset[keep[0]:keep[-1] + 2], length[keep]
    rows = []
    for m, lane in enumerate(lanes):
        s_in = max(S[0] - offset[m], 0.0)
        s_out = min(S[-1] - offset[m], length[m])
        rows.append(dict(lane=lane, seq=m, t_in=tt[0] if m == 0 else when(offset[m]),
                         t_out=tt[-1] if m == len(lanes) - 1 else when(offset[m + 1]),
                         s_in=s_in, s_out=s_out, length=length[m]))
    return rows


def match(g: LaneGraph, tr: pd.DataFrame, dense: Dense | None = None):
    """``(links, vehicles)``.

    ``links``: one row per vehicle and lane on its path: ``id, seq, lane,
    t_in, t_out, s_in, s_out, length``. ``s_in > 0`` or ``s_out < length``
    marks a lane the vehicle was seen on for only part of its length.

    ``vehicles``: one row per road user: ``id, cls, t_first, t_last,
    matched, reason``.
    """
    dense = dense or densify(g)
    trans = _transitions(g)
    rows, veh = [], []
    for vid, v in tr.groupby("id", sort=True):
        r, reason = match_vehicle(g, dense, trans, v)
        veh.append(dict(id=vid, cls=v.cls.iat[0], t_first=v.t.iat[0], t_last=v.t.iat[-1],
                        matched=r is not None, reason=reason))
        for x in r or []:
            rows.append(dict(id=vid, **x))
    return pd.DataFrame(rows), pd.DataFrame(veh)
