"""MATSim network XML (``network_v1``/``network_v2``) -> :class:`~mobslim.network.Network`."""

import os
import xml.etree.ElementTree as ET
from typing import Any, Iterable

from mobslim.network import DEFAULT_MODE, Network, NetworkBuilder
from mobslim.timeutils import parse_time


class MatsimLinkAttributes:
    """Lazy source of a MATSim network's extra link attributes.

    Called (by :meth:`Network.link_attribute`) the first time an attribute
    is needed, it parses the file again and returns ``{name: values}``: a
    link element's ``type`` and ``origid`` (``network_v1``) and its
    ``<attributes>`` children (``network_v2``, typed by their ``class``).
    Picklable (it holds only the path), so worker processes can load too.
    """

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def __call__(self) -> dict[str, list[Any]]:
        from mobslim.formats.matsim_attributes import read_attributes_element

        cols: dict[str, dict[int, Any]] = {}
        k = 0
        in_links = False
        for event, el in ET.iterparse(self.path, events=("start", "end")):
            if event == "start":
                if el.tag == "links":
                    in_links = True
                continue
            if el.tag == "link" and in_links:
                for name in ("type", "origid"):
                    if (v := el.get(name)) is not None:
                        cols.setdefault(name, {})[k] = v
                attrs = el.find("attributes")
                if attrs is not None:
                    for name, v in read_attributes_element(attrs).items():
                        cols.setdefault(name, {})[k] = v
                k += 1
                el.clear()
            elif el.tag == "node":
                el.clear()
        return {name: [vals.get(i) for i in range(k)] for name, vals in cols.items()}

    def __repr__(self) -> str:
        return f"MatsimLinkAttributes({self.path!r})"


def read_network_xml(path: str, default_modes: Iterable[str] = (DEFAULT_MODE,)) -> Network:
    """Read a MATSim ``network_v1``/``v2`` XML file.

    ``capacity`` is normalised from ``capperiod`` to vehicles per hour. Links
    without a ``modes`` attribute get ``default_modes`` (MATSim uses ``car``).
    Extra link attributes (``type``, ``<attributes>``) are not parsed here
    but on first use (:class:`MatsimLinkAttributes`).
    """
    root = ET.parse(path).getroot()
    b = NetworkBuilder()
    for node in root.find("nodes"):
        b.add_node(node.get("id"), float(node.get("x")), float(node.get("y")))
    links = root.find("links")
    capperiod = parse_time(links.get("capperiod", "01:00:00"))
    scale = 3600.0 / capperiod
    for link in links:
        modes = link.get("modes")
        modes = [m.strip() for m in modes.split(",")] if modes else default_modes
        b.add_link(
            link.get("id"),
            link.get("from"),
            link.get("to"),
            length=float(link.get("length")),
            freespeed=float(link.get("freespeed")),
            capacity=float(link.get("capacity")) * scale,
            lanes=float(link.get("permlanes", 1)),
            modes=modes,
        )
    import dataclasses

    return dataclasses.replace(b.build(), link_attribute_source=MatsimLinkAttributes(path))
