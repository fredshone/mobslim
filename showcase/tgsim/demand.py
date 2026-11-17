"""Observed vehicles as mobslim persons, one window of time at a time.

Per vehicle (:func:`vehicles`): where it came from (``main``: the mainline,
``ramp``: the on-ramp), where it went (``through``, ``off``: the off-ramp),
and when it crossed each node's line (``t_A`` ... ``t_X``, NaN where it was
not seen). Its route is the links of :data:`network.ROUTES` between the
first and last place it was seen.

A window simulates ``[t0 - warm, t1]`` and measures ``[t0, t1]``
(:func:`trips`). Vehicles that cross an entry line in it depart onto that
entry's virtual link when they crossed it. Vehicles already on the road at
``t0 - warm``, or first seen partway along it, depart from the link they
were on, then. The warm-up gives the measured minute the traffic of the
simulation, not a copy of the observed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mobslim import Plan, Population, Route
from mobslim.network import Network
from showcase.tgsim.network import ENTRY_NODE, EXIT_NODE, LINES, LINKS, ROUTES

AT_EDGE = 1.0  # [m]


def vehicles(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vid, d in tr.groupby("id", sort=True):
        t, y = d.t.to_numpy(), np.maximum.accumulate(d.y.to_numpy())
        entry = "ramp" if d.lane.iat[0] in (-2, -1) and y[0] > LINES["D"] else "main"
        exit_ = "off" if d.lane.iat[-1] == -6 else "through"
        row = dict(id=vid, cls=d.cls.iat[0], entry=entry, exit=exit_, t_first=t[0], t_last=t[-1],
                   y_first=y[0], y_last=y[-1])
        for n, yl in LINES.items():
            row[f"t_{n}"] = float(np.interp(yl, y, t)) if y[0] <= yl <= y[-1] else np.nan
        if exit_ == "off":
            row["t_M"] = row["t_Q"] = row["t_E"] = np.nan  # its y passes theirs on the ramp
        else:
            row["t_X"] = np.nan
        rows.append(row)
    v = pd.DataFrame(rows)
    return v[v.entry.astype(str) + v.exit != "rampoff"].reset_index(drop=True)  # none in the data


def route(entry: str, exit_: str) -> list[str]:
    return ROUTES[(entry, exit_)]


def link_at(entry: str, exit_: str, y: float) -> str | None:
    """The link of the route that holds ``y`` (None: before or after the route)."""
    for k in route(entry, exit_):
        a, b, _ = LINKS[k]
        if LINES[a] <= y < LINES[b] or (y == LINES[b] and LINKS[k][1] in EXIT_NODE.values()):
            return k
    return None


def trips(v: pd.DataFrame, tr: pd.DataFrame, t0: float, t1: float, warm: float) -> pd.DataFrame:
    """One row per vehicle simulated in the window: ``id, route, t_dep,
    s_dep, edge`` (came in at an entry line), ``measured`` (and did so in
    ``[t0, t1]``) and the observed ``t_exit`` (NaN if lost before the exit
    line)."""
    tw = t0 - warm
    at = tr[np.isclose(tr.t, tw)].groupby("id").y.first()  # positions at the start (a few rows repeat)
    rows = []
    for r in v[(v.t_last > tw) & (v.t_first <= t1)].itertuples():
        full = route(r.entry, r.exit)
        t_entry = getattr(r, f"t_{ENTRY_NODE[r.entry]}")
        if np.isfinite(t_entry) and tw <= t_entry <= t1 and r.y_first <= LINES[ENTRY_NODE[r.entry]] + AT_EDGE:
            first, t_dep, y_dep, edge = full[0], t_entry, LINES[ENTRY_NODE[r.entry]], True
        else:
            if r.t_first <= tw and r.id in at.index:
                t_dep, y_dep = tw, float(at[r.id])  # on the road when the window starts
            elif tw < r.t_first <= t1:
                t_dep, y_dep = r.t_first, r.y_first  # first seen partway along
            else:
                continue
            first, edge = link_at(r.entry, r.exit, y_dep), False
            if first is None:
                continue  # not yet in view, or already past the exit line
        last = link_at(r.entry, r.exit, min(r.y_last, LINES[EXIT_NODE[r.exit]]))
        links = full[full.index(first):full.index(last) + 1] if last in full[full.index(first):] else [first]
        t_exit = getattr(r, f"t_{EXIT_NODE[r.exit]}")
        rows.append(dict(id=r.id, entry=r.entry, exit=r.exit, route=links, t_dep=t_dep,
                         s_dep=y_dep - LINES[LINKS[first][0]], edge=edge,
                         measured=edge and t0 <= t_dep < t1, t_exit=t_exit))
    return pd.DataFrame(rows)


def population(net: Network, trips: pd.DataFrame) -> Population:
    """Vehicles departing at the same time onto the same link leave in the
    order they were along it, the one in front first."""
    t = trips.sort_values(["t_dep", "s_dep"], ascending=[True, False])
    plans = {}
    for r in t.itertuples():
        links = ([f"in_{r.entry}"] if r.edge else []) + r.route
        idx = [net.link_index(x) for x in links]
        start, end = int(net.from_node[idx[0]]), int(net.to_node[idx[-1]])
        tt = float(sum(net.length[k] / net.freespeed[k] for k in idx))
        dist = float(sum(net.length[k] for k in idx))
        plans[int(r.id)] = (Plan.builder()
                            .act("enter", start, end_time=float(r.t_dep))
                            .leg("car", Route(idx, tt, dist))
                            .act("exit", end, link=idx[-1])
                            .build())
    return Population.from_plans(plans)
