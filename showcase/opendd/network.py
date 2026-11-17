"""A roundabout's lanes in the camera's view as a mobslim network.

One link per lane (lane-level, ``lanes=1``), cut at the edges of the view
(``Site.view_cuts``) and without the lanes outside it. Each place where
traffic comes into view also gets a virtual link (:data:`VIRTUAL`) for the
vehicle to wait on when the first link is full, so a queue that grows back
out of view forms there instead of holding back departures.

Every lane, on the ring and the arms alike, has the free speed and
capacity of :data:`showcase.replay.ROUNDABOUT`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mobslim import NetworkBuilder
from mobslim.network import Network
from showcase.opendd.load import LaneGraph
from showcase.replay import ROUNDABOUT, VIRTUAL


@dataclass(frozen=True)
class Site:
    """What the map doesn't say about a roundabout, from looking at it and its tracks."""

    # The camera sees less than the map. Measured over all recordings
    # (``view_edges``): where 95% of the vehicles first seen on a lane after
    # t = 1 s had appeared, and where 95% of those last seen on it before the
    # end were still seen. (lo, hi) [m]: the lane keeps [lo, hi].
    view_cuts: dict[int, tuple[float | None, float | None]]
    outside: frozenset[int]  # lanes left out of the network
    shown: str  # the recording animated in the docs (a held-out recording)


SITES = {
    "rdb1": Site(
        view_cuts={
            28: (5.0, None),  # east arm in: first seen by 5.0 m
            12: (None, 10.0),  # east arm out: last seen from 10.2 m
            20: (None, 10.5),  # south-west arm out: last seen from 10.9 m
        },
        # Outside the view, or with less than 5 m of them in it (17: first
        # seen by 8.9 of 10.9 m; 40: 10.6 of 14.6; 19: 1.9 of 4.4; 39: last
        # seen from 1.5 of 13.3), and the side road at the east edge of the
        # view (5, 6, 7, 8, 11, 13, 29: about one vehicle a recording, seen
        # only in part). Vehicles turning into the side road leave view at
        # the end of 9.
        outside=frozenset({17, 40, 19, 39, 5, 6, 7, 8, 11, 13, 29}
                          | {32, 33, 34, 35, 38} | set(range(41, 51))),
        shown="rdb1_1",
    ),
    "rdb2": Site(
        # Here the camera sees beyond the map, but loses vehicles in a band
        # across the north-east and south-west arms and before the south
        # exit's merge. Measured over every vehicle using a lane (those seen
        # before it count as seen at 0 m): where 95% had been seen, and where
        # 95% were still seen.
        view_cuts={
            90: (15.5, None),  # north-east arm in: 15.5 of 22.7 m
            57: (6.0, None),  # south arm in: 5.7 of 15.9 m
            68: (6.0, None),  # south-west bypass in: 5.9 of 11.6 m (after 67: 11.8 of 13.5)
            70: (4.0, None),  # south-west arm in: 3.9 of 10.3 m (after 69: 12.4 of 14.1)
            86: (None, 7.5),  # north-east arm out: 7.7 of 24.5 m
            56: (None, 7.5),  # south-west bypass out: 7.7 of 11.2 m
        },
        # With less than 5 m in view: 67, 69 (above), 71 (south-west arm
        # out: 3.5 of 20.5 m), 55 (south arm out: 2.5 of 4.7 m) and 54 after it.
        outside=frozenset({67, 69, 71, 55, 54}),
        shown="rdb2_165",
    ),
    "rdb3": Site(
        view_cuts={
            103: (4.0, None),  # east arm in: 3.7 of 19.3 m
            114: (9.5, None),  # north-west arm in: 9.4 of 17.8 m (after 116: 19.4 of 19.7)
            94: (11.5, None),  # south arm in: 11.2 of 20.9 m (after 93: 20.4 of 20.6)
            107: (6.5, None),  # west arm in: 6.2 of 28.4 m
            99: (None, 10.5),  # east arm out: 10.9 of 19.9 m
            112: (None, 14.0),  # north-west arm out: 14.2 of 23.2 m
            104: (None, 16.5),  # south arm out: 16.5 of 20.9 m
            108: (None, 23.5),  # west arm out: 23.7 of 28.3 m
        },
        # With less than 5 m in view: 116, 93 (above), 115 (north-west arm
        # out: 0.2 of 19.3 m) and 92 (south arm out: 0.2 of 21.2 m).
        outside=frozenset({116, 93, 115, 92}),
        shown="rdb3_211",
    ),
    "rdb4": Site(
        view_cuts={
            129: (18.5, None),  # west arm in: 18.5 of 39.7 m
            125: (20.5, None),  # east arm in: 20.2 of 38.1 m
            130: (None, 21.5),  # west arm out: 21.6 of 39.5 m
            123: (None, 21.0),  # east arm out: 21.4 of 39.4 m
        },
        outside=frozenset(),
        shown="rdb4_325",
    ),
    "rdb5": Site(
        view_cuts={
            135: (15.0, None),  # south-east arm in: 15.0 of 46.9 m
            150: (6.5, None),  # north-west arm in: 6.2 of 40.6 m
            145: (29.0, None),  # south-west arm in: 28.7 of 36.2 m
            144: (23.0, None),  # north-east arm in: 22.7 of 37.9 m
            137: (None, 32.0),  # south-east arm out: 32.0 of 45.2 m
            143: (None, 37.0),  # north-west arm out: 37.2 of 42.0 m
            147: (None, 10.0),  # south-west arm out: 10.2 of 37.5 m
            139: (None, 17.0),  # north-east arm out: 17.3 of 38.5 m
        },
        outside=frozenset(),
        shown="rdb5_387",
    ),
    "rdb6": Site(
        view_cuts={
            156: (4.5, None),  # east arm in: 4.2 of 27.1 m
            166: (4.0, None),  # south arm in: 4.0 of 20.5 m (after 165: 23.3 of 23.8)
            179: (7.0, None),  # north arm in: 7.0 of 19.0 m (after 180: 17.9 of 18.9)
            168: (30.5, None),  # south-east bypass in: 30.4 of 36.4 m
            153: (None, 13.5),  # east arm out: 13.6 of 15.8 m
            163: (None, 14.0),  # north arm out: 14.2 of 22.4 m
            169: (None, 19.5),  # south arm out: 19.9 of 23.9 m
            152: (None, 13.0),  # south-east bypass out: 13.2 of 14.9 m
        },
        # With less than 5 m in view: 165, 180 (above), 177 (north arm out:
        # 0.5 of 19.0 m) and 167 (south arm out: 0.5 of 24.1 m).
        outside=frozenset({165, 180, 177, 167}),
        shown="rdb6_415",
    ),
    "rdb7": Site(
        view_cuts={
            188: (13.5, None),  # east arm in: 13.5 of 23.0 m
            204: (20.5, None),  # west arm in: 20.1 of 30.3 m
            183: (5.0, None),  # south arm in: 5.0 of 18.4 m (after 182: 7.2 of 9.7)
            205: (7.5, None),  # north arm in: 7.5 of 18.7 m (after 207: 4.9 of 5.8, 206: 2.1 of 2.6)
            202: (None, 12.5),  # west arm out: 12.7 of 28.6 m
            185: (None, 15.5),  # east arm out: 15.5 of 23.9 m
            192: (None, 19.5),  # north arm out: 19.5 of 23.8 m
            197: (None, 16.5),  # south arm out: 16.9 of 21.4 m
        },
        # With less than 5 m in view: 182, 207, 206 (above), 194 (north arm
        # out: 0.9 of 3.3 m) and 196 (south arm out: 1.2 of 5.8 m).
        outside=frozenset({182, 207, 206, 194, 196}),
        shown="rdb7_485",
    ),
}


def in_view(g: LaneGraph) -> list[int]:
    """Lanes of the (split) graph inside the camera's view."""
    return [i for i in g.ids if i < 1000 and i not in SITES[g.site].outside]


def entries(g: LaneGraph) -> list[int]:
    """In-view lanes with no in-view predecessor: where traffic comes into view."""
    view = set(in_view(g))
    return [i for i in view if not any(p in view for p in g.predecessors[i])]


def exits(g: LaneGraph) -> list[int]:
    """In-view lanes with no in-view successor: where traffic leaves view."""
    view = set(in_view(g))
    return [i for i in view if not any(s in view for s in g.successors[i])]


def build(g: LaneGraph) -> Network:
    """In-view lanes as links (id = lane id), plus a virtual link ``in<lane>``
    before each place traffic comes into view."""
    b = NetworkBuilder()
    view = in_view(g)
    used = {int(n) for i in view for n in (g.from_node[g.index(i)], g.to_node[g.index(i)])}
    for n in sorted(used):
        b.add_node(n, *g.nodes[n])
    for i in view:
        k = g.index(i)
        b.add_link(i, int(g.from_node[k]), int(g.to_node[k]), g.length[k], ROUNDABOUT.freespeed,
                   ROUNDABOUT.capacity, 1.0, ("car",))
    for i in entries(g):
        k = g.index(i)
        geo = g.geometry[k]
        back = geo[0] - (geo[1] - geo[0]) / np.linalg.norm(geo[1] - geo[0]) * VIRTUAL["length"]
        b.add_node(f"in{i}", *back)
        b.add_link(f"in{i}", f"in{i}", int(g.from_node[k]), VIRTUAL["length"], VIRTUAL["freespeed"],
                   VIRTUAL["capacity"], VIRTUAL["lanes"], ("car",))
    return b.build()


def view_edges(links: pd.DataFrame, vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where vehicles come into and go out of view, per lane of the unsplit
    graph: first-seen ``s_in`` (vehicles first seen after t = 1 s) and
    last-seen ``s_out`` (vehicles last seen before the recording's end)."""
    v = vehicles.set_index(["rec", "id"])
    end = v.groupby("rec").t_last.max()
    links = links.reset_index(drop=True)
    first = links[links.seq == 0].join(v, on=["rec", "id"])
    first = first[first.t_first > 1]
    last = links.loc[links.groupby(["rec", "id"]).seq.idxmax()].join(v, on=["rec", "id"])
    last = last[last.t_last < last.rec.map(end) - 1]
    q = lambda p: (lambda x: x.quantile(p))  # noqa: E731
    fs = first.groupby("lane").s_in.agg(n="count", p50=q(0.5), p95=q(0.95))
    ls = last.groupby("lane").s_out.agg(n="count", p5=q(0.05), p50=q(0.5))
    return fs.sort_values("n", ascending=False), ls.sort_values("n", ascending=False)
