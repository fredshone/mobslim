"""MATSim network XML (``network_v1``/``network_v2``) -> :class:`~mobslim.network.Network`."""

import xml.etree.ElementTree as ET
from typing import Iterable

from mobslim.network import DEFAULT_MODE, Network, NetworkBuilder
from mobslim.timeutils import parse_time


def read_network_xml(path: str, default_modes: Iterable[str] = (DEFAULT_MODE,)) -> Network:
    """Read a MATSim ``network_v1``/``v2`` XML file.

    ``capacity`` is normalised from ``capperiod`` to vehicles per hour. Links
    without a ``modes`` attribute get ``default_modes`` (MATSim uses ``car``).
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
    return b.build()
