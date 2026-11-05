"""File formats: every reader and writer lives in this package.

==========================  ===================================================
:mod:`.matsim_network`      MATSim network XML -> :class:`~mobslim.network.Network`
:mod:`.matsim_plans`        MATSim population XML: read and write
:mod:`.matsim_attributes`   MATSim person attributes (``objectAttributes``)
:mod:`.matsim_transit`      MATSim transit schedule and vehicles
:mod:`.parquet`             MobSlim Parquet population directory
==========================  ===================================================

Populations can be read and written in either format, chosen by path:

=================================  ==========================================
``*.xml``, ``*.xml.gz``            MATSim plans (``plans_v4``/``population_v6``)
a directory (``persons.parquet``)  MobSlim Parquet (:mod:`.parquet`)
=================================  ==========================================

:func:`read_population` and :func:`write_population` dispatch on the path;
``python -m mobslim.formats convert`` converts between them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mobslim.network import Network
from mobslim.population import Population

FORMATS = ("xml", "parquet")


def population_format(path, format: Optional[str] = None) -> str:
    """``"xml"`` or ``"parquet"`` for ``path`` (``format`` wins if given).

    A path ending in ``.xml``/``.xml.gz`` is XML; an existing directory with
    ``persons.parquet``, or any other path that does not exist yet (to
    write), is Parquet."""
    if format is not None:
        if format not in FORMATS:
            raise ValueError(f"Unknown population format {format!r}; expected one of {FORMATS}.")
        return format
    name = Path(path).name.lower()
    if name.endswith(".xml") or name.endswith(".xml.gz"):
        return "xml"
    p = Path(path)
    if p.is_dir() or not p.exists() or name.endswith(".parquet"):
        return "parquet"
    raise ValueError(f"Cannot tell the population format of {str(path)!r}: expected *.xml, "
                     f"*.xml.gz or a Parquet population directory.")


def read_population(path, network: Network, *, format: Optional[str] = None,
                    keep_routes: bool = True, attributes_file=None,
                    transit_schedule=None) -> Population:
    """Read a population in any supported format.

    ``attributes_file``: extra person attributes, a MATSim
    ``objectAttributes`` XML or a Parquet persons table; they win where both
    name a value. ``transit_schedule`` keeps Parquet pt rides (MATSim pt
    routes are always routed again)."""
    fmt = population_format(path, format)
    if fmt == "xml":
        from mobslim.formats.matsim_plans import read_population_xml

        population = read_population_xml(path, network, keep_routes=keep_routes)
    else:
        from mobslim.formats.parquet import read_population_parquet

        population = read_population_parquet(path, network, keep_routes=keep_routes,
                                             transit_schedule=transit_schedule)
    if attributes_file is not None:
        population = population.with_attributes(
            population.attributes.merged(read_person_attributes(attributes_file, population)))
    return population


def read_person_attributes(path, population: Population):
    """Person attributes from a MATSim ``objectAttributes`` XML or a Parquet
    persons table, aligned with ``population``'s persons."""
    from mobslim.attributes import PersonAttributes
    from mobslim.formats.matsim_attributes import read_person_attributes_xml

    name = Path(path).name.lower()
    if name.endswith(".xml") or name.endswith(".xml.gz"):
        records = read_person_attributes_xml(path)
    else:
        from mobslim.formats.parquet import read_person_attributes_parquet

        records = read_person_attributes_parquet(path)
    # ids in files may be strings of the population's ids (or the reverse)
    by_str = {str(k): v for k, v in records.items()}
    rows = {pid: records.get(pid) or by_str.get(str(pid), {}) for pid in population.ids}
    return PersonAttributes.from_records(population.ids, rows)


def write_population(population: Population, path, network: Network, *,
                     format: Optional[str] = None, transit_schedule=None) -> Path:
    """Write a population as MATSim XML or Parquet (by ``path``, or ``format``)."""
    fmt = population_format(path, format)
    if fmt == "xml":
        from mobslim.formats.matsim_plans import write_population_xml

        return write_population_xml(population, path, network)
    from mobslim.formats.parquet import write_population_parquet

    return write_population_parquet(population, path, network, transit_schedule=transit_schedule)
