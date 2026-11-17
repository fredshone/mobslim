"""A run's section of road as a mobslim network: each lane group, left and
right, is its own chain of three-lane links, from the line where traffic
comes into view to the line where it leaves.

Each group ends on an exit link just out of view (``out_left``,
``out_right``), which stores one vehicle per lane, so a queue behind it
grows back into view. The left lanes are held up beyond the view: their
exit's capacity is the flow seen leaving them (:data:`EXIT`). A sink link follows it (``sink_left``,
``sink_right``), so that the exit link is never a route's last. Each place
where traffic comes into view gets a virtual link to wait on, as for
openDD.

The lines (:class:`Section`) are measured per run from the tracks
(``run.section``), as the view edges are for openDD and I-395.

Every other link has the free speed and capacity per lane of
:data:`showcase.replay.FREEWAY`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mobslim import NetworkBuilder
from mobslim.network import Network
from showcase.replay import EFFECTIVE_CELL, FREEWAY, VIRTUAL

GROUPS = ("left", "right")
LANES = 3  # per group
LINK_LENGTH = 80.0  # [m] target: the section is cut into equal links about this long
LANE_WIDTH = 3.6  # [m], for drawing only
TAIL = 80.0  # [m] of road drawn beyond the view at each end

# [veh/h] per lane: the queued left lanes discharge about this at both sites (983-1,123 per run, measured
# on vehicles crossing the exit line); None: a freeway lane's
EXIT = {"left": 1000.0, "right": None}
OUT = dict(length=EFFECTIVE_CELL, freespeed=1000.0, lanes=float(LANES))  # ~0 s, 3 vehicles


@dataclass
class Section:
    """The road in view for one run: from ``s_in`` to ``s_out`` [m], cut
    into ``n`` equal links, and each group's centreline (``w`` per ``s``,
    measured over the whole track, beyond the view too)."""

    run: int
    s_in: float
    s_out: float
    n: int
    centre: dict[str, tuple[list[float], list[float]]]  # group: (s, w)
    s_lo: float = 0.0  # where tracks start and end, for drawing
    s_hi: float = 0.0

    @property
    def cuts(self) -> np.ndarray:
        return np.linspace(self.s_in, self.s_out, self.n + 1)

    @property
    def length(self) -> float:
        return (self.s_out - self.s_in) / self.n

    def link_index(self, s: float) -> int | None:
        """Which of the ``n`` links holds ``s`` (None: outside the section)."""
        if not self.s_in <= s <= self.s_out:
            return None
        return min(int((s - self.s_in) / self.length), self.n - 1)

    def w_at(self, group: str, s) -> np.ndarray:
        return np.interp(s, *self.centre[group])


def link_id(group: str, k: int) -> str:
    return f"{group}{k}"


def route(group: str, first: int, last: int, out: bool) -> list[str]:
    """Links from the ``first`` to the ``last`` in-view link, then the exit and sink links if ``out``."""
    return [link_id(group, k) for k in range(first, last + 1)] + ([f"out_{group}", f"sink_{group}"] if out else [])


def entry(group: str) -> str:
    return f"in_{group}"


def build(sec: Section) -> Network:
    b = NetworkBuilder()
    cuts = sec.cuts
    for g in GROUPS:
        for k, s in enumerate(cuts):
            b.add_node(f"{g}@{k}", float(s), float(sec.w_at(g, s)))
        b.add_node(f"in_{g}", float(cuts[0] - VIRTUAL["length"]), float(sec.w_at(g, cuts[0])))
        for k in range(sec.n):
            b.add_link(link_id(g, k), f"{g}@{k}", f"{g}@{k + 1}", sec.length, FREEWAY.freespeed,
                       FREEWAY.capacity * LANES, float(LANES), ("car",))
        b.add_link(entry(g), f"in_{g}", f"{g}@0", VIRTUAL["length"], VIRTUAL["freespeed"], VIRTUAL["capacity"],
                   VIRTUAL["lanes"], ("car",))
        b.add_node(f"x_{g}", float(cuts[-1] + OUT["length"]), float(sec.w_at(g, cuts[-1])))
        cap = (EXIT[g] or FREEWAY.capacity) * LANES
        b.add_link(f"out_{g}", f"{g}@{sec.n}", f"x_{g}", OUT["length"], OUT["freespeed"], cap, OUT["lanes"], ("car",))
        b.add_node(f"sink_{g}", float(cuts[-1] + OUT["length"] + VIRTUAL["length"]), float(sec.w_at(g, cuts[-1])))
        b.add_link(f"sink_{g}", f"x_{g}", f"sink_{g}", VIRTUAL["length"], VIRTUAL["freespeed"], VIRTUAL["capacity"],
                   VIRTUAL["lanes"], ("car",))
    return b.build()
