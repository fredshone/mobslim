"""Preparing plans for simulation (MATSim's ``PrepareForSim``)."""

from dataclasses import replace
from typing import Optional

import numpy as np

from mobslim.activity_links import ActivityLinkParams, assign_activity_links
from mobslim.population import locate_activities
from mobslim.replanning.context import ReplanningContext, default_transit_router
from mobslim.replanning.modules import route_batch, route_plans, routing_groups
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
    it), all in one batch, on the plan arrays (:func:`route_batch`, which
    gives the routes :func:`route_plans` gives). Scores and selection are
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
        return scenario.with_population(_route_held_plans(pop, ctx))
    group_codes, group_weights, group_persons = groups

    # one (stored plan, routing group) pair per distinct combination in use
    slots = np.arange(pop.plan_ids.shape[1])[None, :] < pop.n_plans[:, None]
    codes = np.broadcast_to(group_codes[:, None], pop.plan_ids.shape)
    pair_rows = np.unique(np.stack([pop.plan_ids[slots], codes[slots]], axis=1), axis=0)
    if len(np.unique(pair_rows[:, 0])) == len(pair_rows):
        # each stored plan is held by persons who route alike: route it as
        # its group's first person (their transit weights and mode speeds)
        _, first = np.unique(group_codes, return_index=True)
        return scenario.with_population(_route_held_plans(pop, ctx, first[pair_rows[:, 1]]))
    pairs = pair_rows.tolist()
    store = pop.plans
    routed = route_plans([locate_activities(store[j], net) for j, _ in pairs], ctx,
                         [group_weights[c] for _, c in pairs], [group_persons[c] for _, c in pairs])
    lookup = {(j, c): plan for (j, c), plan in zip(pairs, routed)}
    ids, n = pop.plan_ids.tolist(), pop.n_plans.tolist()
    persons = [person.with_plans([lookup[(j, c)] for j in ids[i][:n[i]]])
               for i, (person, c) in enumerate(zip(pop, group_codes.tolist()))]
    return scenario.with_population(pop.replace(persons))


def _route_held_plans(pop, ctx: ReplanningContext, persons=None):
    """``pop`` with every plan a person holds given coordinates (its
    activities' nodes' where missing, as :func:`locate_activities`) and
    routed with :func:`route_batch`. ``persons``: per held plan (in
    :meth:`~mobslim.population.Population.live_plan_rows` order), the person
    whose transit weights and mode speeds route it."""
    live = pop.live_plan_rows()
    batch = pop.plans.take(live)
    xy = batch.resolved_act_xy(ctx.network)
    located = batch if xy is batch.act_xy else replace(batch, act_xy=xy)
    routed = route_batch(located, ctx, persons)
    if routed is batch:  # nothing to do: keep the store (compacted to the held plans)
        return pop.map_plan_store(lambda plans: plans)
    return pop.with_live_batch(live, routed)
