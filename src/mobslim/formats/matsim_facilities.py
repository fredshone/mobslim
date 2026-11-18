"""MATSim facilities XML (``facilities_v1``/``v2``): where activities happen.

MobSlim reads a facility's location only: its coordinates and, if it names
one, its link (``linkRefId``). The plans readers use them for activities
that name a ``facility`` but no node, link or coordinates (MATSim's
``facilitiesSource=fromFile``). ``.gz`` is allowed; the file is streamed.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mobslim.network import Network


@dataclass(frozen=True)
class Facilities:
    """Facility ``ids`` with their coordinates and link (``-1``: none)."""

    ids: tuple
    x: np.ndarray  # float64
    y: np.ndarray  # float64
    link: np.ndarray  # int32 link index, -1: none
    _index: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_index", {f: i for i, f in enumerate(self.ids)})

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, facility_id: str) -> Optional[tuple[float, float, int]]:
        """``(x, y, link index or -1)`` of ``facility_id``, or ``None``."""
        i = self._index.get(facility_id)
        return None if i is None else (float(self.x[i]), float(self.y[i]), int(self.link[i]))


def read_facilities_xml(path, network: Network) -> Facilities:
    """The facilities of a MATSim facilities file; ``linkRefId`` links are
    looked up in ``network`` (``ValueError`` if it lacks one)."""
    path = str(path)
    ids, xs, ys, links = [], [], [], []
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "facility":
                continue
            ids.append(el.get("id"))
            xs.append(float(el.get("x")))
            ys.append(float(el.get("y")))
            link = el.get("linkRefId")
            links.append(network.link_index(link) if link else -1)
            el.clear()
    return Facilities(tuple(ids), np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64),
                      np.array(links, dtype=np.int32))
