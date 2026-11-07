"""Array-backed, multi-modal road network.

The :class:`Network` stores link attributes as flat numpy arrays indexed by a
dense link index ("struct of arrays"). This is the format the hot loops (the
mobsim, routers, travel-time collection) work on, and the format that can be
handed to a compiled extension or a worker process without conversion.

Links are identified by their own id, not by ``(from, to)``, so parallel links
between the same pair of nodes are supported. Each link carries the set of
modes allowed on it; per-mode views (e.g. adjacency for routing) are derived
lazily and cached.

Units: length [m], freespeed [m/s], capacity [veh/h], lanes [-].

Further link attributes (e.g. MATSim's ``type`` or ``osm:way:highway``)
are optional columns, read with :meth:`Network.link_attribute`. A loader
can supply them lazily: :func:`~mobslim.formats.matsim_network.read_network_xml`
does not parse them until one is first asked for, so runs that never use
them pay nothing.
"""

import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Union

import numpy as np

from mobslim.ids import IdMap

DEFAULT_MODE = "car"


@dataclass(frozen=True, eq=False)
class Network:
    """Immutable network. Build with :class:`NetworkBuilder` or a loader."""

    node_ids: IdMap
    node_xy: np.ndarray  # (n_nodes, 2) float
    link_ids: IdMap
    from_node: np.ndarray  # (n_links,) int
    to_node: np.ndarray  # (n_links,) int
    length: np.ndarray  # (n_links,) float, meters
    freespeed: np.ndarray  # (n_links,) float, m/s
    capacity: np.ndarray  # (n_links,) float, vehicles per hour
    lanes: np.ndarray  # (n_links,) float
    link_modes: tuple[frozenset[str], ...]
    # extra link attributes: {name: column}, or a picklable callable returning
    # one, called on first use (lazy loading); None: no extra attributes
    link_attribute_source: Union[None, Mapping[str, Any], Callable[[], Mapping[str, Any]]] = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        n = len(self.link_ids)
        for name in ("from_node", "to_node", "length", "freespeed", "capacity", "lanes"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Network.{name} must have {n} entries.")
        if len(self.link_modes) != n:
            raise ValueError(f"Network.link_modes must have {n} entries.")
        if self.node_xy.shape != (len(self.node_ids), 2):
            raise ValueError("Network.node_xy must have shape (n_nodes, 2).")

    # -- pickling: drop derived caches (they may hold unpicklable objects) --
    def __getstate__(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "_cache"}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_cache", {})

    # -- sizes and lookups --
    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_links(self) -> int:
        return len(self.link_ids)

    @property
    def modes(self) -> frozenset[str]:
        """All modes that appear on at least one link."""
        if "modes" not in self._cache:
            self._cache["modes"] = frozenset().union(*self.link_modes)
        return self._cache["modes"]

    def node_index(self, node_id: Hashable) -> int:
        return self.node_ids.index(node_id)

    def link_index(self, link_id: Hashable) -> int:
        return self.link_ids.index(link_id)

    # -- link attributes --
    _BUILTIN_ATTRIBUTES = ("length", "freespeed", "capacity", "lanes")

    def _link_attributes(self) -> dict:
        if "link_attrs" not in self._cache:
            from mobslim.attributes import column_array

            src = self.link_attribute_source
            if callable(src):
                src = src()
            cols = {}
            for name, values in (src or {}).items():
                col = values if isinstance(values, np.ndarray) else column_array(values)
                if len(col) != self.n_links:
                    raise ValueError(f"Link attribute {name!r} has {len(col)} values "
                                     f"for {self.n_links} links.")
                cols[name] = col
            self._cache["link_attrs"] = cols
        return self._cache["link_attrs"]

    def link_attribute_names(self) -> tuple[str, ...]:
        """Names of the extra link attributes (loads them if lazy)."""
        return tuple(self._link_attributes())

    def has_link_attribute(self, name: str) -> bool:
        return name in self._BUILTIN_ATTRIBUTES or name in self._link_attributes()

    def link_attribute(self, name: str) -> np.ndarray:
        """One value per link: an extra attribute (``None``/``NaN`` where a
        link lacks it), or one of the built-in columns ``length``,
        ``freespeed``, ``capacity``, ``lanes``. Extra attributes are loaded
        on the first call if the network was read lazily."""
        if name in self._BUILTIN_ATTRIBUTES:
            return getattr(self, name)
        attrs = self._link_attributes()
        if name not in attrs:
            raise KeyError(f"Network has no link attribute {name!r} "
                           f"(has: {sorted(attrs) + list(self._BUILTIN_ATTRIBUTES)}).")
        return attrs[name]

    def mode_mask(self, mode: str) -> np.ndarray:
        """Boolean array: is ``mode`` allowed on each link."""
        key = ("mask", mode)
        if key not in self._cache:
            self._cache[key] = np.array(
                [mode in m for m in self.link_modes], dtype=bool
            )
        return self._cache[key]

    def free_speed_travel_times(self) -> np.ndarray:
        """``length / freespeed`` for every link (seconds)."""
        if "fstt" not in self._cache:
            self._cache["fstt"] = self.length / self.freespeed
        return self._cache["fstt"]

    def out_links(self, mode: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """CSR adjacency for ``mode``: ``(offsets, links)``.

        Outgoing links of node ``u`` are ``links[offsets[u]:offsets[u + 1]]``.
        Returned as (cached, immutable) tuples because they are consumed by
        Python loops; use ``np.asarray`` for array code.
        """
        key = ("csr", mode)
        if key not in self._cache:
            idx = np.flatnonzero(self.mode_mask(mode))
            order = idx[np.argsort(self.from_node[idx], kind="stable")]
            counts = np.bincount(self.from_node[order], minlength=self.n_nodes)
            offsets = np.concatenate([[0], np.cumsum(counts)])
            self._cache[key] = (tuple(offsets.tolist()), tuple(order.tolist()))
        return self._cache[key]

    def nearest_node(self, x: float, y: float) -> int:
        """Index of the node nearest to ``(x, y)``."""
        if "qt" not in self._cache:
            from fastquadtree import QuadTreeObjects

            xs, ys = self.node_xy[:, 0], self.node_xy[:, 1]
            bounds = (xs.min() - 1, ys.min() - 1, xs.max() + 1, ys.max() + 1)
            # small leaves so the tree actually splits; f64 because real-world
            # (e.g. UTM) coordinates lose metre precision in f32
            qt = QuadTreeObjects(bounds, capacity=16, dtype="f64")
            qt.insert_many([tuple(p) for p in self.node_xy.tolist()], list(range(self.n_nodes)))
            self._cache["qt"] = qt
        return self._cache["qt"].nearest_neighbor((x, y)).obj

    def beeline_distance(self, a: int, b: int) -> float:
        """Euclidean distance between two nodes."""
        (x0, y0), (x1, y1) = self.node_xy[a], self.node_xy[b]
        return math.hypot(x1 - x0, y1 - y0)

    def to_networkx(self, mode: Optional[str] = None):
        """A ``networkx.MultiDiGraph`` view, keyed by external ids.

        Intended for analysis and plotting, not for the hot loops.
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        for i, nid in enumerate(self.node_ids):
            g.add_node(nid, x=float(self.node_xy[i, 0]), y=float(self.node_xy[i, 1]))
        for k, lid in enumerate(self.link_ids):
            if mode is not None and mode not in self.link_modes[k]:
                continue
            g.add_edge(
                self.node_ids.id(int(self.from_node[k])),
                self.node_ids.id(int(self.to_node[k])),
                key=lid,
                length=float(self.length[k]),
                freespeed=float(self.freespeed[k]),
                capacity=float(self.capacity[k]),
                lanes=float(self.lanes[k]),
                modes=self.link_modes[k],
            )
        return g

    def __repr__(self) -> str:
        return f"Network(nodes={self.n_nodes}, links={self.n_links}, modes={sorted(self.modes)})"


class NetworkBuilder:
    """Incrementally build a :class:`Network`."""

    def __init__(self):
        self._nodes = IdMap()
        self._xy: list[tuple[float, float]] = []
        self._links = IdMap()
        self._rows: list[tuple] = []
        self._attrs: dict[str, dict[int, Any]] = {}

    def add_node(self, node_id: Hashable, x: float, y: float) -> int:
        if node_id in self._nodes:
            raise ValueError(f"Duplicate node id {node_id!r}.")
        self._xy.append((float(x), float(y)))
        return self._nodes.add(node_id)

    def add_link(
        self,
        link_id: Hashable,
        from_node: Hashable,
        to_node: Hashable,
        length: float,
        freespeed: float,
        capacity: float,
        lanes: float = 1.0,
        modes: Iterable[str] = (DEFAULT_MODE,),
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Add a link between two existing nodes. ``capacity`` is veh/h;
        ``attributes`` are extra link attributes (e.g. ``{"type": "primary"}``)."""
        if link_id in self._links:
            raise ValueError(f"Duplicate link id {link_id!r}.")
        if length <= 0 or freespeed <= 0 or capacity <= 0 or lanes <= 0:
            raise ValueError(
                f"Link {link_id!r}: length, freespeed, capacity and lanes must be positive."
            )
        u, v = self._nodes.index(from_node), self._nodes.index(to_node)
        modes = frozenset(modes)
        if not modes:
            raise ValueError(f"Link {link_id!r} has no modes.")
        self._rows.append((u, v, float(length), float(freespeed), float(capacity), float(lanes), modes))
        k = self._links.add(link_id)
        for name, value in (attributes or {}).items():
            self._attrs.setdefault(name, {})[k] = value
        return k

    def build(self) -> Network:
        cols = list(zip(*self._rows)) if self._rows else [()] * 7
        return Network(
            node_ids=self._nodes,
            node_xy=np.array(self._xy, dtype=float).reshape(-1, 2),
            link_ids=self._links,
            from_node=np.array(cols[0], dtype=np.int64),
            to_node=np.array(cols[1], dtype=np.int64),
            length=np.array(cols[2], dtype=float),
            freespeed=np.array(cols[3], dtype=float),
            capacity=np.array(cols[4], dtype=float),
            lanes=np.array(cols[5], dtype=float),
            link_modes=tuple(cols[6]),
            link_attribute_source={name: [vals.get(k) for k in range(len(self._rows))]
                                   for name, vals in self._attrs.items()} or None,
        )


# -- loaders and synthetic networks ------------------------------------------


def grid_network(
    size: int = 10,
    spacing: float = 50.0,
    freespeed: float = 10.0,
    capacity: float = 900.0,
    lanes: float = 1.0,
    modes: Iterable[str] = (DEFAULT_MODE,),
) -> Network:
    """A ``(size + 1) x (size + 1)`` grid with links in both directions.

    Node ids are ``(row, col)`` tuples; link ids are ``(from_id, to_id)``.
    The conventional demo origin is ``(0, 0)`` and destination ``(size, size)``.
    """
    b = NetworkBuilder()
    for i in range(size + 1):
        for j in range(size + 1):
            b.add_node((i, j), j * spacing, i * spacing)
    for i in range(size + 1):
        for j in range(size + 1):
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni > size or nj > size:
                    continue
                for u, v in (((i, j), (ni, nj)), ((ni, nj), (i, j))):
                    b.add_link((u, v), u, v, spacing, freespeed, capacity, lanes, modes)
    return b.build()


def linear_network(
    size: int = 10,
    spacing: float = 50.0,
    freespeed: float = 10.0,
    capacity: float = 900.0,
    lanes: float = 1.0,
    modes: Iterable[str] = (DEFAULT_MODE,),
) -> Network:
    """Nodes ``0..size`` on a line, links in both directions."""
    b = NetworkBuilder()
    for i in range(size + 1):
        b.add_node(i, i * spacing, 0.0)
    for i in range(size):
        b.add_link((i, i + 1), i, i + 1, spacing, freespeed, capacity, lanes, modes)
        b.add_link((i + 1, i), i + 1, i, spacing, freespeed, capacity, lanes, modes)
    return b.build()
