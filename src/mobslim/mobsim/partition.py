"""Splitting a network's nodes into partitions for the parallel mobsims.

Two methods, both deterministic:

* ``"morton"`` (:func:`partition_nodes`): nodes in Morton (Z-curve) order of
  their coordinates, cut into runs of equal *traffic* (route links ending
  at each node, plus one). Balances work; ignores the links it cuts.
* ``"metis"`` (:func:`metis_partition`): METIS's k-way partitioning of the
  undirected node graph, as matsim-rust does: fewest cut links within 3 %
  of equal node counts, contiguous partitions, and few neighbours per
  partition. Needs ``pymetis`` (``pip install mobslim[metis]``).

PSim's results do not depend on the partition, nor do DSim's
trajectories; only the speed does.
"""

from typing import Optional

import numpy as np

from mobslim.network import Network
from mobslim.planbatch import PlanBatch

METHODS = ("morton", "metis")


def partition(net: Network, plans: Optional[PlanBatch], k: int, method: str) -> np.ndarray:
    """Each node's partition (``0 .. k-1``) by ``method`` (see :data:`METHODS`)."""
    if method == "morton":
        return partition_nodes(net, plans, k)
    if method == "metis":
        return metis_partition(net, k)
    raise ValueError(f"partitioner must be one of {METHODS}, not {method!r}.")


def partition_nodes(net: Network, plans: Optional[PlanBatch], k: int) -> np.ndarray:
    """Split the nodes into ``k`` spatially compact partitions of about equal
    work: nodes in Morton (Z-curve) order of their coordinates, cut into
    ``k`` runs of equal weight, where a node's weight is the number of route
    links that end at it (the moves it decides), plus one."""
    n = net.n_nodes
    if k <= 1 or n == 0:
        return np.zeros(n, dtype=np.int64)
    weight = np.ones(n)
    if plans is not None and len(plans.route_links):
        weight += np.bincount(net.to_node[plans.route_links.astype(np.int64)], minlength=n)
    xy = np.asarray(net.node_xy, dtype=float)
    lo, span = xy.min(axis=0), np.ptp(xy, axis=0)
    q = ((xy - lo) / np.where(span > 0, span, 1.0) * 65535).astype(np.uint64)
    code = _spread(q[:, 0]) | (_spread(q[:, 1]) << np.uint64(1))
    order = np.argsort(code, kind="stable")
    before = np.cumsum(weight[order]) - weight[order]  # weight ahead of each node
    part = np.empty(n, dtype=np.int64)
    part[order] = np.minimum((before * k / weight.sum()).astype(np.int64), k - 1)
    return part


def _spread(v: np.ndarray) -> np.ndarray:
    """Interleave the low 16 bits of ``v`` with zeros (for Morton codes)."""
    v = v & np.uint64(0xFFFF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
    v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
    v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
    v = (v | (v << np.uint64(1))) & np.uint64(0x55555555)
    return v


def metis_partition(net: Network, k: int, *, imbalance: float = 1.03, contiguous: bool = True,
                    min_conn: bool = True, seed: int = 4711) -> np.ndarray:
    """METIS k-way partition of the nodes, with matsim-rust's defaults.

    Each pair of nodes joined by at least one link (either direction) is one
    edge of weight 1; loops are ignored; every node weighs 1. ``imbalance``
    is the heaviest partition's weight relative to the mean. If the graph is
    not connected, ``contiguous`` is dropped (METIS cannot honour it).
    """
    n = net.n_nodes
    if k <= 1 or n == 0:
        return np.zeros(n, dtype=np.int64)
    try:
        import pymetis
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("partitioner='metis' needs pymetis (pip install mobslim[metis]).") from e
    a = np.asarray(net.from_node, dtype=np.int64)
    b = np.asarray(net.to_node, dtype=np.int64)
    keep = a != b
    pairs = np.unique(np.concatenate([np.stack([a[keep], b[keep]], 1),
                                      np.stack([b[keep], a[keep]], 1)]), axis=0)
    xadj = np.searchsorted(pairs[:, 0], np.arange(n + 1)).astype(np.int64)
    adjncy = pairs[:, 1].astype(np.int64)
    contig = contiguous and _connected(n, xadj, adjncy)
    opts = pymetis.Options(seed=seed, minconn=int(min_conn), contig=int(contig),
                           ufactor=max(1, round((imbalance - 1.0) * 1000)))
    graph = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
    _, parts = pymetis.part_graph(k, graph, options=opts)
    return np.asarray(parts, dtype=np.int64)


def _connected(n: int, xadj: np.ndarray, adjncy: np.ndarray) -> bool:
    seen = np.zeros(n, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adjncy[xadj[u]:xadj[u + 1]].tolist():
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return bool(seen.all())
