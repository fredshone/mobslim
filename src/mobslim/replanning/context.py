"""Read-only context handed to replanning.

Everything a strategy may consult (network, learned travel times, mode
definitions, the iteration number, person attributes) lives here, so a
strategy is a pure function of ``(person, context, rng)``. The context is
picklable: routers are built lazily from ``router_factory`` and cached per
process, never pickled.

Person attributes: ``ctx.attributes`` is the population's
:class:`~mobslim.attributes.PersonAttributes` table. The runners hand each
person's strategy a context bound to that person (``ctx.for_person(i)``), so
strategies and modules read the person being replanned with
``ctx.attribute(name)`` or ``ctx.person_attributes()``.

Person-specific transit routing: ``ctx.transit_weights`` (a
:class:`~mobslim.transit.PersonRaptorWeights`, computed once per run from the
attributes) gives each person's :class:`~mobslim.transit.RaptorWeights`;
``Reroute``, the runners' batch routing and ``route_scenario`` route each
person's pt legs with them.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from mobslim.attributes import PersonAttributes
from mobslim.modes import Modes
from mobslim.network import Network
from mobslim.routing.base import Router, RouterFactory, TravelTime
from mobslim.routing.compiled import default_router


@dataclass(frozen=True)
class TransitRouterFactory:
    """Transit router factory: :class:`~mobslim.transit.RustRaptorRouter` when
    the compiled router is installed, else :class:`~mobslim.transit.RaptorRouter`
    (identical routes either way), with ``params`` (a
    :class:`~mobslim.transit.RaptorParams`; ``None``: its defaults).

    ``TransitRouterFactory.from_scoring(params)`` routes with the utilities
    the scorer uses (see :meth:`RaptorParams.from_scoring
    <mobslim.transit.RaptorParams.from_scoring>`). Picklable, so it can be
    sent to replanning worker processes.
    """

    params: Optional[object] = None

    @classmethod
    def from_scoring(cls, params, mode: str = "pt", **kw) -> "TransitRouterFactory":
        from mobslim.transit.raptor import RaptorParams

        return cls(RaptorParams.from_scoring(params, mode, **kw))

    def __call__(self, network, transit):
        from mobslim.transit.raptor import (
            RaptorParams,
            RaptorRouter,
            RustRaptorRouter,
            rust_raptor_available,
        )

        params = self.params if self.params is not None else RaptorParams()
        if rust_raptor_available():
            return RustRaptorRouter(network, transit, params)
        return RaptorRouter(network, transit, params)


# RaptorParams's own defaults (not tied to any scoring)
default_transit_router = TransitRouterFactory()


@dataclass(frozen=True, eq=False)
class ReplanningContext:
    """``modes`` is the scenario's :class:`~mobslim.modes.Modes`; use
    ``ctx.modes.is_network_mode(...)`` / ``ctx.modes.teleport(...)``."""

    modes: Modes
    travel_time: TravelTime
    iteration: int = 0
    router_factory: RouterFactory = default_router
    transit_router_factory: object = default_transit_router
    # set by the replanning runners: Reroute only clears routes and the runner
    # routes all new plans in one batch (see Reroute, route_plans)
    batch_routing: bool = False
    attributes: Optional[PersonAttributes] = field(default=None, repr=False)
    # the person being replanned (set by for_person); None outside replanning
    person_index: Optional[int] = None
    # per-person transit weights (PersonRaptorWeights); None: the router's own
    transit_weights: Optional[object] = field(default=None, repr=False)
    _routers: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_routers"] = {}
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    @classmethod
    def from_scenario(cls, scenario, travel_time: TravelTime, iteration: int = 0,
                      router_factory: RouterFactory = default_router,
                      transit_router_factory=default_transit_router,
                      transit_weights=None) -> "ReplanningContext":
        return cls(scenario.modes, travel_time, iteration, router_factory, transit_router_factory,
                   attributes=scenario.population.attributes, transit_weights=transit_weights)

    def for_person(self, index: int) -> "ReplanningContext":
        """This context bound to the person with person index ``index``
        (sharing everything else, including the routers). Cheap: one
        shallow copy, no validation."""
        if self.person_index == index:
            return self
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__["person_index"] = index
        return new

    def _bound(self) -> int:
        if self.person_index is None:
            raise RuntimeError("ReplanningContext is not bound to a person (use for_person).")
        return self.person_index

    def attribute(self, name: str, default: Any = None) -> Any:
        """The current person's value of attribute ``name`` (``default`` if
        missing, or if there are no attributes)."""
        if self.attributes is None:
            return default
        return self.attributes.value(self._bound(), name, default)

    def person_transit_weights(self):
        """The current person's :class:`~mobslim.transit.RaptorWeights`, or
        ``None`` (the router's own; also when not bound to a person)."""
        if self.transit_weights is None or self.person_index is None:
            return None
        return self.transit_weights.of(self.person_index)

    def person_attributes(self) -> dict[str, Any]:
        """The current person's attributes (missing values left out)."""
        if self.attributes is None:
            return {}
        return self.attributes.row(self._bound())

    def batched(self) -> "ReplanningContext":
        """This context with ``batch_routing`` on (sharing its routers)."""
        if self.batch_routing:
            return self
        return replace(self, batch_routing=True)

    @property
    def network(self) -> Network:
        return self.modes.network

    def router(self, mode: str) -> Router:
        """The router for ``mode``: the transit router (one for every transit
        mode) or the mode's network router. Built lazily, cached per process."""
        if self.modes.is_transit_mode(mode):
            key = ("transit",)
            if key not in self._routers:
                self._routers[key] = self.transit_router_factory(self.network, self.modes.transit)
            return self._routers[key]
        if mode not in self._routers:
            self._routers[mode] = self.router_factory(self.network, mode, self.travel_time)
        return self._routers[mode]
