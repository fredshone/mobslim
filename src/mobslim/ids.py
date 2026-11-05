"""Dense integer indexing for external ids.

Every entity that the hot loops touch (nodes, links, persons) is referred to
internally by a dense ``int`` index. External ids (strings from MATSim XML,
tuples for synthetic grids, ...) are kept in an :class:`IdMap` so they can be
recovered for I/O and reporting.

Dense integer indices are what make the core data model array-friendly: they
can be shipped to numpy, to worker processes, or across a Rust (PyO3) boundary
without translation.
"""

from typing import Hashable, Iterable, Iterator


class IdMap:
    """Bidirectional mapping between external ids and dense ``int`` indices.

    Indices are assigned in insertion order, starting at 0.
    """

    __slots__ = ("_ids", "_index")

    def __init__(self, ids: Iterable[Hashable] = ()):
        self._ids: list[Hashable] = []
        self._index: dict[Hashable, int] = {}
        for ext in ids:
            if ext in self._index:
                raise ValueError(f"Duplicate id {ext!r}.")
            self.add(ext)

    def add(self, ext: Hashable) -> int:
        """Return the index for ``ext``, assigning a new one if unseen."""
        idx = self._index.get(ext)
        if idx is None:
            idx = len(self._ids)
            self._ids.append(ext)
            self._index[ext] = idx
        return idx

    def index(self, ext: Hashable) -> int:
        """Index of an external id. Raises ``KeyError`` if unknown."""
        try:
            return self._index[ext]
        except KeyError:
            raise KeyError(f"Unknown id {ext!r}.") from None

    def get(self, ext: Hashable, default=None):
        return self._index.get(ext, default)

    def id(self, idx: int) -> Hashable:
        """External id of an index."""
        return self._ids[idx]

    def __contains__(self, ext: Hashable) -> bool:
        return ext in self._index

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._ids)

    def __getstate__(self):
        return self._ids

    def __setstate__(self, ids):
        self._ids = list(ids)
        self._index = {ext: i for i, ext in enumerate(self._ids)}

    def __repr__(self) -> str:
        return f"IdMap(n={len(self)})"
