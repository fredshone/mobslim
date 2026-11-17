"""Read openDD trajectories and a roundabout's lane graph (``rdb1``, ``rdb2``).

Coordinates are UTM 32N shifted so the mean lane vertex is the origin (as in
``scripts/animate_opendd.py``); every later step works in these coordinates.

Two layouts of ``data/opendd/`` are accepted:

* the full download per roundabout, e.g. ``rdb1/trajectories_rdb1_v3.sqlite``:
  one table per recording (``rdb1_1`` ... ``rdb1_153``);
* the example data, ``example_data/rdb1_4.sqlite``: recording ``rdb1_4`` only.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "opendd"
HZ = 1 / 0.0333667  # recording rate (29.97 Hz)
DROP = ("Pedestrian", "Trailer")  # trailers duplicate their towing vehicle
SNAP = 0.5  # lane ends closer than this [m] are one node


def coords(wkt: str) -> np.ndarray:
    """(n, 2) points of a WKT LINESTRING."""
    return np.array([[float(v) for v in p.split()]
                     for p in re.search(r"\(+(.*?)\)+", wkt).group(1).split(",")])


def _ids(s) -> list[int]:
    return [int(float(v)) for v in re.findall(r"[\d.]+", s)] if s else []


def site_of(recording: str) -> str:
    """``rdb2_160`` -> ``rdb2``."""
    if not re.fullmatch(r"rdb\d+_\d+", recording):
        raise ValueError(f"not an openDD recording: {recording!r}")
    return recording.split("_")[0]


def map_path(site: str = "rdb1", base: Path = DATA) -> Path:
    for p in (base / site / f"map_{site}" / f"map_{site}.sqlite",
              base / "example_data" / f"map_{site}" / f"map_{site}.sqlite"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no map_{site}.sqlite under {base}; see showcase/opendd/__init__.py")


def trajectories_path(recording: str, base: Path = DATA) -> Path:
    site = site_of(recording)
    full = base / site / f"trajectories_{site}_v3.sqlite"
    if full.exists():
        return full
    single = base / "example_data" / f"{recording}.sqlite"
    if single.exists():
        return single
    raise FileNotFoundError(f"no trajectories for {recording} under {base}")


def recordings(site: str = "rdb1", base: Path = DATA) -> list[str]:
    """A roundabout's recording names available, in numeric order."""
    full = base / site / f"trajectories_{site}_v3.sqlite"
    if full.exists():
        with sqlite3.connect(full) as c:
            names = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    else:
        names = [p.stem for p in (base / "example_data").glob(f"{site}_*.sqlite")]
    return sorted(names, key=lambda n: int(n.split("_")[1]))


@dataclass
class LaneGraph:
    """A roundabout's ``trafficLane`` centrelines as a directed graph of nodes and lanes.

    Lane ``k`` (map id ``ids[k]``) runs from node ``from_node[k]`` to node
    ``to_node[k]``. ``successors``/``predecessors`` are the map's own lists,
    by map id.
    """

    ids: list[int]
    geometry: list[np.ndarray]  # per lane, (n, 2) points in shifted coordinates
    length: np.ndarray  # per lane [m]
    from_node: np.ndarray
    to_node: np.ndarray
    nodes: np.ndarray  # (n_nodes, 2)
    successors: dict[int, list[int]]
    predecessors: dict[int, list[int]]
    boundaries: list[np.ndarray]
    origin: np.ndarray  # UTM (x, y) of the shifted origin
    repairs: list[str]  # changes made to the map's lists, for the docs
    site: str = "rdb1"

    def index(self, lane_id: int) -> int:
        return self.ids.index(lane_id)

    @property
    def entries(self) -> list[int]:
        return [i for i in self.ids if not self.predecessors[i]]

    @property
    def exits(self) -> list[int]:
        return [i for i in self.ids if not self.successors[i]]


def _snap(points: np.ndarray, tol: float) -> np.ndarray:
    """Node index per point: points within ``tol`` (transitively) share one."""
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    for i, j in zip(*np.nonzero(np.triu(d < tol, 1))):
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(points))]
    _, node = np.unique(roots, return_inverse=True)
    return node


def load_lanes(site: str = "rdb1", base: Path = DATA) -> LaneGraph:
    with sqlite3.connect(map_path(site, base)) as c:
        mp = pd.read_sql(f"select * from {site}", c)
    lanes = mp[mp.type == "trafficLane"].assign(id=lambda d: d.id.astype(int)).sort_values("id")
    geom = [coords(g) for g in lanes.geometry]
    origin = np.vstack(geom).mean(0)
    geom = [g - origin for g in geom]
    bounds = [coords(g) - origin for g in mp[mp.type == "boundary"].geometry.dropna()]
    n = len(geom)
    ends = np.array([g[0] for g in geom] + [g[-1] for g in geom])
    node = _snap(ends, SNAP)
    nodes = np.array([ends[node == k].mean(0) for k in range(node.max() + 1)])
    ids = lanes.id.tolist()
    succ = dict(zip(ids, map(_ids, lanes.successors)))
    pred = dict(zip(ids, map(_ids, lanes.predecessors)))
    repairs = []
    for lists, kind in ((succ, "successors"), (pred, "predecessors")):
        for i, js in lists.items():  # rdb1: lane 34 lists 36 and 37, which are not in the map
            missing = [j for j in js if j not in succ]
            if missing:
                lists[i] = [j for j in js if j in succ]
                repairs.append(f"lane {i}: dropped {kind} {missing} (not in the map)")
    return LaneGraph(
        ids=ids, geometry=geom,
        length=np.array([np.linalg.norm(np.diff(g, axis=0), axis=1).sum() for g in geom]),
        from_node=node[:n], to_node=node[n:], nodes=nodes,
        successors=succ, predecessors=pred, boundaries=bounds, origin=origin, repairs=repairs, site=site)


def load_trajectories(recording: str = "rdb1_4", base: Path = DATA,
                      origin: np.ndarray | None = None) -> pd.DataFrame:
    """One recording's road users (no pedestrians or trailers), in shifted coordinates.

    Columns: ``t`` [s], ``f`` (frame), ``id``, ``x``, ``y``, ``angle`` [rad],
    ``v`` [m/s], ``acc`` [m/s²], ``cls``, ``width``, ``length`` [m].
    """
    with sqlite3.connect(trajectories_path(recording, base)) as c:
        tr = pd.read_sql(
            "select TIMESTAMP t, OBJID id, UTM_X x, UTM_Y y, UTM_ANGLE angle, V v, ACC acc, "
            f"CLASS cls, WIDTH width, LENGTH length from {recording} "
            f"where CLASS not in ({','.join('?' * len(DROP))}) order by OBJID, TIMESTAMP",
            c, params=DROP)
    if origin is None:
        origin = load_lanes(site_of(recording), base).origin
    tr["x"] -= origin[0]
    tr["y"] -= origin[1]
    tr["f"] = (tr.t * HZ).round().astype(int)
    return tr


def _cut(geo: np.ndarray, s: float) -> np.ndarray:
    """The point ``s`` metres along a polyline."""
    seg = np.diff(geo, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    j = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
    return geo[j] + seg[j] * (s - cum[j]) / (cum[j + 1] - cum[j])


def _piece(geo: np.ndarray, a: float, b: float) -> np.ndarray:
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(geo, axis=0), axis=1))])
    inner = geo[(cum > a) & (cum < b)]
    return np.vstack([_cut(geo, a), inner, _cut(geo, b)])


def split_lanes(g: LaneGraph, cuts: dict[int, tuple[float | None, float | None]]) -> LaneGraph:
    """Split lanes at the edges of the camera's view.

    ``cuts[i] = (lo, hi)``: lane ``i`` keeps ``[lo, hi]`` (either may be
    ``None``: no cut there); the part before ``lo`` becomes lane
    ``1000 + i`` and the part after ``hi`` lane ``2000 + i``.
    """
    ids, geom, succ, pred = [], [], {}, {}
    first, last = {}, {}  # original id -> id of its first / last piece
    for i, geo in zip(g.ids, g.geometry):
        lo, hi = cuts.get(i, (None, None))
        L = g.length[g.index(i)]
        pieces = []
        if lo is not None:
            pieces.append((1000 + i, 0.0, lo))
        pieces.append((i, lo or 0.0, L if hi is None else hi))
        if hi is not None:
            pieces.append((2000 + i, hi, L))
        for (a, s0, s1), nxt in zip(pieces, pieces[1:] + [None]):
            ids.append(a)
            geom.append(geo if len(pieces) == 1 else _piece(geo, s0, s1))
            succ[a] = [nxt[0]] if nxt else None
            pred.setdefault(a, [])
            if nxt:
                pred[nxt[0]] = [a]
        first[i], last[i] = pieces[0][0], pieces[-1][0]
    for i in g.ids:
        succ[last[i]] = [first[j] for j in g.successors[i]]
        pred[first[i]] = [last[j] for j in g.predecessors[i]]
    order = np.argsort(ids)
    ids = [ids[k] for k in order]
    geom = [geom[k] for k in order]
    n = len(geom)
    ends = np.array([q[0] for q in geom] + [q[-1] for q in geom])
    node = _snap(ends, 0.01)
    return LaneGraph(
        ids=ids, geometry=geom,
        length=np.array([np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q in geom]),
        from_node=node[:n], to_node=node[n:],
        nodes=np.array([ends[node == k].mean(0) for k in range(node.max() + 1)]),
        successors=succ, predecessors=pred, boundaries=g.boundaries, origin=g.origin,
        repairs=g.repairs + [f"lane {i}: split at the view edge {cuts[i]}" for i in sorted(cuts)],
        site=g.site)
