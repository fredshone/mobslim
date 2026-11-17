"""Observed vehicles as mobslim persons: one car leg each, on its observed route.

* A vehicle seen coming into view (its first in-view lane is an entry and it
  was first seen within ``AT_EDGE`` of the lane's start) departs onto that
  entry's virtual link when it crossed the view edge.
* Any other vehicle was already inside the view when first seen: in view at
  t = 0, or tracked late (occlusion). It departs onto the lane it was first
  seen on, when it was first seen. It did not come into view, so it is left
  out of the time-in-view comparison.

Vehicles departing at the same time onto the same link leave in the order
they were along it, the one in front first. Every route ends at the end of
the last in-view lane the vehicle was seen on.
"""

from __future__ import annotations

import pandas as pd

from mobslim import Plan, Population, Route
from mobslim.network import Network
from showcase.opendd.load import LaneGraph
from showcase.opendd.network import entries, in_view

AT_EDGE = 1.0  # [m]


def trips(g: LaneGraph, links: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle: ``id, route`` (in-view lanes), ``t_dep``,
    ``s_dep`` (how far along its first lane), ``edge`` (came into view), and
    the observed ``t_exit`` (when it left its last lane; NaN if it was lost
    before that lane's end)."""
    view, ent = set(in_view(g)), set(entries(g))
    rows = []
    for vid, d in links[links.lane.isin(view)].sort_values(["id", "seq"]).groupby("id"):
        if not (d.seq.diff().dropna() == 1).all():
            raise ValueError(f"vehicle {vid}: in-view lanes are not contiguous")
        first, last = d.iloc[0], d.iloc[-1]
        edge = bool(first.lane in ent and first.s_in < AT_EDGE)
        done = last.s_out > last.length - AT_EDGE
        rows.append(dict(id=vid, route=d.lane.tolist(), t_dep=first.t_in, s_dep=first.s_in,
                         edge=edge, t_exit=last.t_out if done else float("nan")))
    return pd.DataFrame(rows)


def population(net: Network, trips: pd.DataFrame) -> Population:
    t = trips.sort_values(["t_dep", "s_dep"], ascending=[True, False])
    plans = {}
    for r in t.itertuples():
        links = ([f"in{r.route[0]}"] if r.edge else []) + r.route
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
