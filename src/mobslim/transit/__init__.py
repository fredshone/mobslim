"""Scheduled public transport (design: ``docs/design.md`` section 5).

* Data: :class:`TransitSchedule`, :class:`TransitVehicles`, put together with
  a :class:`TransitParams` in a :class:`Transit` (``Scenario.transit``).
  Build them with :class:`TransitBuilder` or read MATSim files with
  :func:`read_transit`.
* Plans: a pt leg carries a :class:`TransitPassengerRoute` (its rides).
* Routing: :class:`RaptorRouter` (least-cost RAPTOR, as SwissRailRaptor), and
  :class:`RustRaptorRouter`, the same on the compiled kernel.
* Simulation: ``QSim`` and ``HermesSim`` run transit vehicles on the schedule
  or the network.
"""

from mobslim.formats.matsim_transit import (
    read_transit,
    read_transit_schedule_xml,
    read_transit_vehicles_xml,
)
from mobslim.population import TransitPassengerRoute
from mobslim.transit.builder import TransitBuilder
from mobslim.transit.core import Transit, TransitParams
from mobslim.transit.raptor import (
    PersonRaptorWeights,
    RaptorData,
    RaptorParams,
    RaptorRouter,
    RaptorWeights,
    RustRaptorRouter,
    rust_raptor_available,
)
from mobslim.transit.schedule import TransitSchedule, build_schedule
from mobslim.transit.vehicles import TransitVehicles, build_vehicles

__all__ = ["PersonRaptorWeights", "TransitPassengerRoute", "RaptorParams", "RaptorData", "RaptorWeights", "RaptorRouter", "RustRaptorRouter", "Transit", "TransitBuilder",
           "TransitParams", "TransitSchedule", "TransitVehicles", "build_schedule",
           "build_vehicles", "read_transit", "read_transit_schedule_xml",
           "read_transit_vehicles_xml", "rust_raptor_available"]
