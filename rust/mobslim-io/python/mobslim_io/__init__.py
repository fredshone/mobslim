"""Compiled file readers for :mod:`mobslim.formats`.

``read_population_columns`` reads a MATSim ``plans_v4``/``population_v6``
file into the columns ``mobslim.formats.matsim_plans.read_population_xml``
builds; it raises ``FallBack`` for anything it does not read exactly as that
reader does (which then reads the file itself).
"""

from mobslim_io._kernel import FallBack, read_population_columns

__all__ = ["FallBack", "read_population_columns"]
