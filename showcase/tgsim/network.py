"""The I-395 section in view as a mobslim network: one link per stretch of
road between where the lane count changes, with all its lanes.

::

    in_main -> A --up-- D --mid-- M --merge (4 lanes)-- Q --down-- E
                        |         ^
                        off       in_ramp
                        v
                        X

Every node is a line across the road at a ``y`` (:data:`LINES`), measured
from the tracks (``run.view_lines``): where 95% of the vehicles coming in
have been seen (A, M), where 95% of those leaving are still seen (E, X),
where off-ramp vehicles start to leave the left lane (D), and where the
on-ramp's added lane has ended (Q). Each place where traffic comes into view
also gets a virtual link (:data:`VIRTUAL`) to wait on, as for openDD.

Every link has the free speed and capacity per lane of
:data:`showcase.replay.FREEWAY`, counting only the lanes that continue past
its end (:data:`CAPACITY_LANES`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mobslim import NetworkBuilder
from mobslim.network import Network
from showcase.replay import FREEWAY, VIRTUAL

LINES = {  # [m] along y
    "A": 161.0,  # mainline in: 95% of vehicles first seen by 161.0
    "D": 200.0,  # off-ramp vehicles enter the off-ramp lane from 200.1 (5%) to 241.9 (95%)
    "M": 291.0,  # on-ramp in: 95% first seen by 290.8
    "Q": 500.0,  # the added lane ends: 99% of its points are before 500.0 (95% of vehicles leave it by 524)
    "E": 583.0,  # mainline out: 95% of vehicles still seen at 583.5
    "X": 281.0,  # off-ramp out: 95% still seen at 281.1
}
LINKS = {  # id: (from, to, lanes)
    "up": ("A", "D", 3),
    "mid": ("D", "M", 3),
    "off": ("D", "X", 1),
    "merge": ("M", "Q", 4),
    "down": ("Q", "E", 3),
}
ROUTES = {  # (entry, exit): links
    ("main", "through"): ["up", "mid", "merge", "down"],
    ("main", "off"): ["up", "off"],
    ("ramp", "through"): ["merge", "down"],
}
ENTRY_NODE = {"main": "A", "ramp": "M"}
EXIT_NODE = {"through": "E", "off": "X"}
LANE_WIDTH = 3.6  # [m], for drawing only

# A link's flow capacity is its end's: the merge link's added lane ends in it, so it has four lanes of
# storage and three of capacity.
CAPACITY_LANES = {"merge": 3}
TAIL = 80.0  # [m] of road drawn beyond the view at each entry and exit


@dataclass
class Geometry:
    """Per link: centreline (n, 2) in data coordinates and its length [m].
    ``lines["added"]`` is the merge link's fourth lane, for drawing: the
    merge centreline is its three through lanes'. ``tails``: the road
    beyond the view, for drawing: back from each entry line (``main``,
    ``ramp``) and on from each exit line (``through``, ``off``)."""

    lines: dict[str, np.ndarray]
    length: dict[str, float]
    tails: dict[str, np.ndarray]


def _median_line(tr: pd.DataFrame, lanes: list[int], y0: float, y1: float, step: float) -> np.ndarray:
    d = tr[tr.lane.isin(lanes) & tr.y.between(y0 - step, y1 + step)]
    ys = np.arange(y0, y1 + step / 2, step)
    ys[-1] = y1
    xs = [d[(d.y - y).abs() < step / 2].x.median() for y in ys]
    return np.column_stack([pd.Series(xs).interpolate(limit_direction="both"), ys])


def geometry(tr: pd.DataFrame, step: float = 5.0) -> Geometry:
    """Link centrelines from the tracks: the median ``x`` of the lanes each
    link carries, every ``step`` metres of ``y``."""
    lanes = {"up": [-5, -4, -3], "mid": [-5, -4, -3], "merge": [-5, -4, -3], "down": [-5, -4, -3],
             "off": [-6], "added": [-2]}
    lines, length = {}, {}
    for k, (a, b) in {**{k: v[:2] for k, v in LINKS.items()}, "added": ("M", "Q")}.items():
        lines[k] = _median_line(tr, lanes[k], LINES[a], LINES[b], step)
        length[k] = float(np.linalg.norm(np.diff(lines[k], axis=0), axis=1).sum())
    # the off-ramp leaves the left lane at D
    lines["off"][0] = lines["up"][-1] - [LANE_WIDTH, 0]
    length["off"] = float(np.linalg.norm(np.diff(lines["off"], axis=0), axis=1).sum())
    y_lo, y_hi = float(tr.y.min()), float(tr.y.max())
    tails = {
        "main": _median_line(tr, lanes["up"], max(y_lo, LINES["A"] - TAIL), LINES["A"], step)[::-1],
        "ramp": _median_line(tr, [-2], max(y_lo, LINES["M"] - TAIL), LINES["M"], step)[::-1],
        "through": _median_line(tr, lanes["down"], LINES["E"], min(y_hi, LINES["E"] + TAIL), step),
        "off": _median_line(tr, [-6], LINES["X"], min(y_hi, LINES["X"] + TAIL), step),
    }
    return Geometry(lines, length, tails)


def build(geo: Geometry) -> Network:
    """The links of :data:`LINKS` plus a virtual link ``in_main`` / ``in_ramp``
    before each place traffic comes into view."""
    b = NetworkBuilder()
    for n, y in LINES.items():
        pt = next(geo.lines[k][0 if LINKS[k][0] == n else -1] for k in LINKS if n in LINKS[k][:2])
        b.add_node(n, float(pt[0]), float(y))
    for k, (a, c, lanes) in LINKS.items():
        cap = FREEWAY.capacity * CAPACITY_LANES.get(k, lanes)
        b.add_link(k, a, c, geo.length[k], FREEWAY.freespeed, cap, float(lanes), ("car",))
    for entry, n in ENTRY_NODE.items():
        k = ROUTES[(entry, "through")][0]
        x, y = geo.lines[k][0]
        b.add_node(f"in_{entry}", float(x), float(y) - VIRTUAL["length"])
        b.add_link(f"in_{entry}", f"in_{entry}", n, VIRTUAL["length"], VIRTUAL["freespeed"],
                   VIRTUAL["capacity"], VIRTUAL["lanes"], ("car",))
    return b.build()
