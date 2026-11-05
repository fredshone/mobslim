"""MATSim attributes: typed ``<attribute>`` elements and ``objectAttributes`` files."""

from typing import Any, Optional

_JAVA_INT = {"java.lang.Integer", "java.lang.Long", "java.lang.Short", "java.lang.Byte"}
_JAVA_FLOAT = {"java.lang.Double", "java.lang.Float"}


def parse_matsim_attribute(cls: Optional[str], text: Optional[str]):
    """The value of a MATSim ``<attribute class="...">text</attribute>``.

    Strings, numbers and booleans are converted; other classes (maps,
    custom types) keep their text.
    """
    text = "" if text is None else text
    if cls in _JAVA_INT:
        return int(text.strip())
    if cls in _JAVA_FLOAT:
        return float(text.strip())
    if cls == "java.lang.Boolean":
        return text.strip().lower() == "true"
    return text


def read_attributes_element(el) -> dict[str, Any]:
    """``{name: value}`` of an ``<attributes>`` (or ``<object>``) element's
    ``<attribute>`` children."""
    return {a.get("name"): parse_matsim_attribute(a.get("class"), a.text)
            for a in el.findall("attribute")}


def read_person_attributes_xml(path) -> dict[str, dict[str, Any]]:
    """A MATSim ``objectAttributes`` file (``plans.inputPersonAttributesFile``,
    ``.gz`` allowed) -> ``{person_id: {name: value}}``."""
    import gzip
    import xml.etree.ElementTree as ET

    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        root = ET.parse(f).getroot()
    return {obj.get("id"): read_attributes_element(obj) for obj in root.findall("object")}
