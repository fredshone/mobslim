"""Preparing plans for simulation (MATSim's ``PrepareForSim``)."""

from typing import Optional

import numpy as np

from mobslim.activity_links import ActivityLinkParams, assign_activity_links
from mobslim.population import locate_activities
from mobslim.replanning.context import ReplanningContext, default_transit_router
from mobslim.replanning.modules import route_plans, routing_groups
from mobslim.routing.base import RouterFactory, TravelTime
from mobslim.routing.compiled import default_router
from mobslim.routing.traveltime import TravelTimeTable
from mobslim.scenario import Scenario


def route_scenario(
    scenario: Scenario,
    travel_time: Optional[TravelTime] = None,
    router_factory: RouterFactory = default_router,
    transit_router_factory=default_transit_router,
    transit_weights=None,
    activity_links: Optional[ActivityLinkParams] = ActivityLinkParams(),
) -> Scenario:
    """Validate leg modes, give every activity coordinates (its node's where
    the plans had none, as MATSim's ``PrepareForSim`` does), assign every
    activity without a link to one (:func:`~mobslim.activity_links.assign_activity_links`
    with ``activity_links``; ``None``: leave activities without links), and
    route every network and transit leg that has no route.

    Applies to every stored plan (once per plan, however many persons share
    it), all in one batch (:func:`route_plans`). Scores and selection are
    kept. ``travel_time`` defaults to free speed.

    ``transit_weights`` (a :class:`~mobslim.transit.PersonRaptorWeights`)
    routes each person's transit legs with their own weights: a stored plan
    is then routed once per distinct weights among the persons holding it
    (and split into one plan per weights if they differ). The same holds for
    per-person mode speeds (``scenario.person_max_speeds``), which time
    teleported legs and so the departures of the legs after them.
    """
    scenario.validate_modes()
    if activity_links is not None:
        scenario = assign_activity_links(scenario, activity_links)
    if travel_time is None:
        travel_time = TravelTimeTable.free_speed(scenario.network)
    ctx = ReplanningContext.from_scenario(scenario, travel_time, 0, router_factory,
                                          transit_router_factory, transit_weights)
    net = scenario.network
    pop = scenario.population
    groups = routing_groups(ctx)
    if groups is None:
        population = pop.map_plan_store(
            lambda plans: route_plans([locate_activities(p, net) for p in plans], ctx))
        return scenario.with_population(population)
    group_codes, group_weights, group_persons = groups

    # one (stored plan, routing group) pair per distinct combination in use
    slots = np.arange(pop.plan_ids.shape[1])[None, :] < pop.n_plans[:, None]
    codes = np.broadcast_to(group_codes[:, None], pop.plan_ids.shape)
    pairs = np.unique(np.stack([pop.plan_ids[slots], codes[slots]], axis=1), axis=0).tolist()
    store = pop.plans
    routed = route_plans([locate_activities(store[j], net) for j, _ in pairs], ctx,
                         [group_weights[c] for _, c in pairs], [group_persons[c] for _, c in pairs])
    lookup = {(j, c): plan for (j, c), plan in zip(pairs, routed)}
    if len({j for j, _ in pairs}) == len(pairs):  # each plan has one weights: keep the store
        by_plan = {j: plan for (j, _), plan in lookup.items()}
        live = pop.live_plan_rows().tolist()  # the rows map_plan_store passes, in order
        population = pop.map_plan_store(
            lambda plans: [by_plan.get(j, p) for j, p in zip(live, plans)])
        return scenario.with_population(population)
    ids, n = pop.plan_ids.tolist(), pop.n_plans.tolist()
    persons = [person.with_plans([lookup[(j, c)] for j in ids[i][:n[i]]])
               for i, (person, c) in enumerate(zip(pop, group_codes.tolist()))]
    return scenario.with_population(pop.replace(persons))
