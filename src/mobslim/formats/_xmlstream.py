"""Streaming over large MATSim XML files (populations, attribute files)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import IO, Iterator


def iter_root_children(f: IO[bytes], tag: str) -> Iterator[ET.Element]:
    """Each direct child of the root element named ``tag``, in file order,
    as a complete element (what ``ET.parse(f).getroot().findall(tag)``
    yields), without holding the whole document.

    Each child is cleared and dropped from the root once the caller asks for
    the next one, so use (or copy out of) it before advancing. Memory is
    about one child's XML at a time.
    """
    events = ET.iterparse(f, events=("start", "end"))
    _, root = next(events)
    depth = 0  # below the root
    for event, el in events:
        if event == "start":
            depth += 1
            continue
        depth -= 1
        if depth == 0:  # a child of the root (the root's own end leaves -1)
            if el.tag == tag:
                yield el
            el.clear()
            del root[:]
