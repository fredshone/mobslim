"""Observed vehicles as mobslim persons, one window of time at a time, as
for I-395 (``showcase.tgsim.demand``).

A vehicle's group is the lane group it was last seen in: where it was
heading. Its route ends on the link where it was last seen, and goes on to
its group's exit link only if it was seen crossing the exit line. Tracks
that start or end partway along the road (12-14% on some runs) are
vehicles the tracker picked up or lost there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mobslim import Plan, Population, Route
from mobslim.network import Network
from showcase.tgsim_i90.network import Section, entry, route

AT_EDGE = 1.0  # [m]


def vehicles(tr: pd.DataFrame, sec: Section) -> pd.DataFrame:
    """Per vehicle: ``group``, first and last seen, and when it crossed each
    cut of the section (``t_0`` ... ``t_n``, NaN where not seen)."""
    rows = []
    for vid, d in tr.groupby("id", sort=True):
        t, s = d.t.to_numpy(), np.maximum.accumulate(d.s.to_numpy())
        row = dict(id=vid, cls=d.cls.iat[0], group=d.group.iat[-1], t_first=t[0], t_last=t[-1],
                   s_first=s[0], s_last=s[-1])
        for k, c in enumerate(sec.cuts):
            row[f"t_{k}"] = float(np.interp(c, s, t)) if s[0] <= c <= s[-1] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def trips(v: pd.DataFrame, tr: pd.DataFrame, sec: Section, t0: float, t1: float, warm: float) -> pd.DataFrame:
    """One row per vehicle simulated in ``[t0 - warm, t1]``: ``id, group,
    first`` (link index it departs onto), ``t_dep, s_dep, edge, measured``
    and the observed ``t_exit``."""
    tw = t0 - warm
    at = tr[np.isclose(tr.t, tw)].groupby("id").s.first()
    rows = []
    for r in v[(v.t_last > tw) & (v.t_first <= t1)].itertuples():
        t_entry = r.t_0
        if np.isfinite(t_entry) and tw <= t_entry <= t1 and r.s_first <= sec.s_in + AT_EDGE:
            first, t_dep, s_dep, edge = 0, t_entry, sec.s_in, True
        else:
            if r.t_first <= tw and r.id in at.index:
                t_dep, s_dep = tw, float(at[r.id])  # on the road when the window starts
            elif tw < r.t_first <= t1:
                t_dep, s_dep = r.t_first, r.s_first  # first seen partway along
            else:
                continue
            first, edge = sec.link_index(s_dep), False
            if first is None or s_dep >= sec.s_out:
                continue  # not yet in view, or already past the exit line
        t_exit = getattr(r, f"t_{sec.n}")
        last = sec.n - 1 if np.isfinite(t_exit) else max(first, sec.link_index(min(r.s_last, sec.s_out)) or 0)
        rows.append(dict(id=r.id, group=r.group, first=first, last=last, t_dep=t_dep,
                         s_dep=s_dep - sec.cuts[first], edge=edge, measured=edge and t0 <= t_dep < t1,
                         t_exit=t_exit))
    return pd.DataFrame(rows)


def population(net: Network, trips: pd.DataFrame, sec: Section) -> Population:
    """Vehicles departing at the same time onto the same link leave in the
    order they were along it, the one in front first."""
    t = trips.sort_values(["t_dep", "s_dep"], ascending=[True, False])
    plans = {}
    for r in t.itertuples():
        links = ([entry(r.group)] if r.edge else []) + route(r.group, r.first, r.last, np.isfinite(r.t_exit))
        idx = [net.link_index(x) for x in links]
        tt = float(sum(net.length[k] / net.freespeed[k] for k in idx))
        dist = float(sum(net.length[k] for k in idx))
        plans[int(r.id)] = (Plan.builder()
                            .act("enter", int(net.from_node[idx[0]]), end_time=float(r.t_dep))
                            .leg("car", Route(idx, tt, dist))
                            .act("exit", int(net.to_node[idx[-1]]), link=idx[-1])
                            .build())
    return Population.from_plans(plans)
