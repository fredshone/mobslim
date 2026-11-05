"""Plan innovation modules: functions ``Plan -> Plan``.

A module must not mutate its input; plans are immutable and modules build new
ones. Modules compose: a strategy applies its modules in order (e.g.
``ChangeTripMode`` then ``Reroute``).

Two forms of the same module:

* ``mutate(plan, ctx, rng)``: one :class:`Plan`, a ``random.Random`` (the
  per-person path). This is all a custom module needs.
* ``mutate_batch(batch, persons, ctx, rng)``: many plans at once, as a
  :class:`~mobslim.planbatch.PlanBatch` (``persons``: the person index of
  each plan; ``rng`` a ``numpy.random.Generator``), returning a batch of the
  same plans changed (:class:`~mobslim.replanning.vectorised.VectorRunner`).
  A module without it is applied plan by plan with ``mutate``.
"""

import math
from dataclasses import replace
from random import Random
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from mobslim.planbatch import _I32, PlanBatch
from mobslim.planstore import csr_gather
from mobslim.population import Activity, Leg, Plan, Route, TransitPassengerRoute
from mobslim.replanning.context import ReplanningContext
from mobslim.routing.base import route_many_to_links


@runtime_checkable
class PlanInnovationModule(Protocol):
    def mutate(self, plan: Plan, ctx: ReplanningContext, rng: Random) -> Plan: ...


class Reroute:
    """Route every network leg with the context's routers.

    Departure times are estimated by walking through the plan using activity
    end times/durations and expected leg travel times, so time-dependent
    travel times are respected. Teleported legs keep ``route=None``.

    With ``only_missing=True`` only legs without a route are routed (used to
    prepare initial plans).

    When ``ctx.batch_routing`` is set (the replanning runners set it), the
    legs are not routed here: their routes are cleared and the runner routes
    every new plan afterwards in one batch (:func:`route_plans`), which gives
    the same routes. Modules *after* ``Reroute`` in a strategy then see legs
    without routes, so put ``Reroute`` last.

    A network leg to an activity on a link (``Activity.link``) ends by
    driving through that link, as in MATSim
    (:func:`~mobslim.routing.base.route_many_to_links`).
    """

    def __init__(self, only_missing: bool = False):
        self.only_missing = only_missing

    def mutate(self, plan: Plan, ctx: ReplanningContext, rng: Random) -> Plan:
        if ctx.batch_routing:
            return _clear_routes(plan, ctx, keep_network=self.only_missing)
        if not self.only_missing:
            plan = _clear_routes(plan, ctx, keep_network=False)
        w = ctx.person_transit_weights()
        return route_plans([plan], ctx, None if w is None else [w], [ctx.person_index])[0]

    def mutate_batch(self, batch: PlanBatch, persons, ctx: ReplanningContext, rng) -> PlanBatch:
        if ctx.batch_routing:
            return clear_routes_batch(batch, ctx, keep_network=self.only_missing)
        if not self.only_missing:
            batch = clear_routes_batch(batch, ctx, keep_network=False)
        return route_batch(batch, ctx, persons)


def _clear_routes(plan: Plan, ctx: ReplanningContext, keep_network: bool) -> Plan:
    """Drop the routes :func:`route_plans` should (re)compute: every network
    leg's (unless ``keep_network``) and any route on a teleported leg."""
    modes = ctx.modes
    elements = list(plan.elements)
    changed = False
    for i in range(1, len(elements), 2):
        leg = elements[i]
        if leg.route is not None and not (keep_network and _has_route(leg, modes)):
            elements[i] = replace(leg, route=None)
            changed = True
    return plan.with_elements(elements) if changed else plan


def _has_route(leg: Leg, modes) -> bool:
    """Whether a routed-mode leg has a route of the right type."""
    route = leg.route
    if route is None:
        return False
    if modes.is_transit_mode(leg.mode):
        return route.__class__ is TransitPassengerRoute
    return route.__class__ is Route


def needs_routing(plan: Plan, ctx: ReplanningContext) -> bool:
    """Whether :func:`route_plans` would change ``plan``."""
    modes = ctx.modes
    for leg in plan.elements[1::2]:
        if modes.is_routed_mode(leg.mode):
            if not _has_route(leg, modes):
                return True
        elif leg.route is not None:
            return True
    return False


def routing_groups(ctx: ReplanningContext):
    """Persons whose plans route alike: same transit weights and same
    per-person mode speeds. ``(codes, weights, persons)``: each person's group
    code, and per group its :class:`~mobslim.transit.RaptorWeights` (or
    ``None``) and one person index (for speeds). ``None`` if everyone routes
    alike."""
    tw = ctx.transit_weights
    wcodes = tw.codes if tw is not None and tw.any else None
    scodes, _ = ctx.modes.speed_groups()
    if wcodes is None and scodes is None:
        return None
    n = len(wcodes) if wcodes is not None else len(scodes)
    w = np.asarray(wcodes, dtype=np.int64) if wcodes is not None else np.full(n, -1, dtype=np.int64)
    s = scodes if scodes is not None else np.zeros(n, dtype=np.int64)
    pairs, first, codes = np.unique(np.stack([w, s], axis=1), axis=0, return_index=True,
                                    return_inverse=True)
    weights = [None if wcodes is None else tw.of_code(int(a)) for a, _ in pairs.tolist()]
    persons = first.tolist() if scodes is not None else [None] * len(pairs)
    return codes.reshape(-1), weights, persons


def route_plans(plans: Sequence[Plan], ctx: ReplanningContext,
                weights: Optional[Sequence] = None,
                persons: Optional[Sequence[Optional[int]]] = None) -> list[Plan]:
    """Route every network leg without a route, in all ``plans`` at once.

    Each plan is walked as :class:`Reroute` walks it (a leg departs when the
    previous activity ends, given the expected travel times of the legs
    before it). Legs are routed in rounds: every plan's first unrouted leg,
    then every plan's next one, and so on, with one ``route_many`` call per
    mode per round. Each route is the one ``router.route`` would return, so
    the result equals routing plan by plan, only faster. Routes on teleported
    legs are dropped. Plans that need nothing are returned as they are.

    ``weights``: one :class:`~mobslim.transit.RaptorWeights` (or ``None``:
    the router's own) per plan, for its transit legs. ``persons``: one person
    index (or ``None``) per plan, whose mode speeds time its teleported legs.
    """
    if weights is not None and all(w is None for w in weights):
        weights = None
    modes = ctx.modes
    out = list(plans)
    # per plan being routed: [index into out, elements, next element, time, changed, person]
    active = [[k, list(p.elements), 0, 0.0, False, None if persons is None else persons[k]]
              for k, p in enumerate(plans) if needs_routing(p, ctx)]
    pending = [s for s in active if _advance(s, modes)]
    while pending:
        by_mode: dict[str, list] = {}
        for s in pending:
            by_mode.setdefault(s[1][s[2]].mode, []).append(s)
        pending = []
        for mode, states in by_mode.items():
            router = ctx.router(mode)
            if modes.is_transit_mode(mode):
                net = modes.network
                args = ([s[1][s[2] - 1].xy(net) for s in states],
                        [s[1][s[2] + 1].xy(net) for s in states],
                        [s[3] for s in states])
                if weights is None:
                    routes = router.route_many_xy(*args)
                else:
                    routes = router.route_many_xy(*args, weights=[weights[s[0]] for s in states])
            else:
                origins = [s[1][s[2] - 1] for s in states]
                dests = [s[1][s[2] + 1] for s in states]
                routes = route_many_to_links(
                    router, modes.network, mode,
                    [a.node for a in origins], [a.link for a in origins],
                    [a.node for a in dests], [a.link for a in dests],
                    [s[3] for s in states],
                )
            for s, route in zip(states, routes):
                elements, i = s[1], s[2]
                elements[i] = replace(elements[i], route=route)
                s[2], s[3], s[4] = i + 1, s[3] + route.travel_time, True
                if _advance(s, modes):
                    pending.append(s)
    for k, elements, _, _, changed, _ in active:
        if changed:
            out[k] = out[k].with_elements(elements)
    return out


def _advance(state: list, modes) -> bool:
    """Walk ``state`` to its next network leg without a route (``True``) or
    to the end of the plan (``False``), updating the clock."""
    _, elements, i, time, changed, person = state
    n = len(elements)
    while i < n:
        element = elements[i]
        if i % 2 == 0:  # activity
            end = element.planned_end(time)
            time = end if end is not None else time
        elif modes.is_routed_mode(element.mode):
            if not _has_route(element, modes):
                state[2], state[3], state[4] = i, time, changed
                return True
            time += element.route.travel_time
        else:
            if element.route is not None:
                elements[i] = replace(element, route=None)
                changed = True
            net = modes.network
            time += modes.teleport(
                element.mode, elements[i - 1].xy(net), elements[i + 1].xy(net), element.travel_time,
                person,
            )[0]
        i += 1
    state[2], state[3], state[4] = i, time, changed
    return False


class TimeAllocationMutator:
    """Shift each activity's end time (or duration) by ``U(-range, +range)``.

    The last activity is untouched. Times are clamped at 0.
    """

    def __init__(self, mutation_range: float = 1800.0):
        self.mutation_range = mutation_range

    def mutate(self, plan: Plan, ctx: ReplanningContext, rng: Random) -> Plan:
        r = self.mutation_range
        elements = list(plan.elements)
        for i in range(0, len(elements) - 1, 2):
            act: Activity = elements[i]
            shift = rng.uniform(-r, r)
            if act.end_time is not None:
                elements[i] = replace(act, end_time=max(0.0, act.end_time + shift))
            elif act.duration is not None:
                elements[i] = replace(act, duration=max(0.0, act.duration + shift))
        return plan.with_elements(elements)

    def mutate_batch(self, batch: PlanBatch, persons, ctx: ReplanningContext, rng) -> PlanBatch:
        r = self.mutation_range
        shift = rng.uniform(-r, r, batch.n_acts)
        last = np.zeros(batch.n_acts, dtype=bool)
        last[batch.act_offsets[1:] - 1] = True
        end, dur = batch.act_end_time, batch.act_duration
        has_end = ~last & ~np.isnan(end)
        has_dur = ~last & np.isnan(end) & ~np.isnan(dur)
        new_end = np.where(has_end, np.maximum(0.0, end + shift), end)
        new_dur = np.where(has_dur, np.maximum(0.0, dur + shift), dur)
        return replace(batch, act_end_time=new_end, act_duration=new_dur)


class ChangeTripMode:
    """Switch all legs to a randomly chosen different mode from ``modes``.

    Routes are dropped; combine with :class:`Reroute` in the same strategy.

    With ``ignore_car_availability=False`` (MATSim's
    ``changeMode.ignoreCarAvailability=false``), ``car_mode`` is not chosen
    for persons without a car; if no mode is left, the plan is unchanged.
    Who has none: with ``availability_attribute``, persons whose boolean
    attribute of that name is false (the mobslim config's
    ``persons.car_availability``); otherwise, as MATSim, persons whose
    ``carAvail`` attribute is ``"never"`` or whose ``hasLicense`` is ``"no"``.
    """

    def __init__(self, modes: Sequence[str], ignore_car_availability: bool = True,
                 car_mode: str = "car", availability_attribute: Optional[str] = None):
        if len(modes) < 2:
            raise ValueError("ChangeTripMode needs at least two modes.")
        self.modes = tuple(modes)
        self.ignore_car_availability = ignore_car_availability
        self.car_mode = car_mode
        self.availability_attribute = availability_attribute

    def _car_unavailable(self, ctx: ReplanningContext) -> bool:
        if self.availability_attribute is not None:
            return ctx.attribute(self.availability_attribute, True) is False
        return (ctx.attribute("carAvail") == "never"
                or ctx.attribute("hasLicense") == "no")

    def mutate(self, plan: Plan, ctx: ReplanningContext, rng: Random) -> Plan:
        legs = plan.legs
        if not legs:
            return plan
        current = legs[0].mode
        choices = [m for m in self.modes if m != current]
        if (not self.ignore_car_availability and self.car_mode in choices
                and ctx.attributes is not None and self._car_unavailable(ctx)):
            choices.remove(self.car_mode)
            if not choices:
                return plan
        mode = rng.choice(choices)
        elements = [
            Leg(mode) if isinstance(e, Leg) else e for e in plan.elements
        ]
        return plan.with_elements(elements)

    def _car_unavailable_many(self, ctx: ReplanningContext, persons: np.ndarray) -> np.ndarray:
        attrs = ctx.attributes

        def equals(name, value):
            col = attrs.get(name)
            if col is None:
                return np.zeros(len(persons), dtype=bool)
            return np.array([v == value and v is not None and type(v) is type(value)
                             for v in col[persons].tolist()], dtype=bool)

        if self.availability_attribute is not None:
            return equals(self.availability_attribute, False)
        return equals("carAvail", "never") | equals("hasLicense", "no")

    def mutate_batch(self, batch: PlanBatch, persons, ctx: ReplanningContext, rng) -> PlanBatch:
        persons = np.asarray(persons, dtype=np.int64)
        n = batch.n_plans
        n_legs = np.diff(batch.leg_offsets)
        u = rng.random(n)
        if not batch.n_legs:
            return batch
        modes = list(batch.modes)
        for m in self.modes:
            if m not in modes:
                modes.append(m)
        code = {m: i for i, m in enumerate(modes)}
        own = np.array([code[m] for m in self.modes], dtype=np.int64)
        has_legs = n_legs > 0
        first_leg = np.minimum(batch.leg_offsets[:-1], max(batch.n_legs - 1, 0))
        current = np.where(has_legs, batch.leg_mode[first_leg], -1)
        allowed = own[None, :] != current[:, None]  # (plans, modes): the choices
        if (not self.ignore_car_availability and self.car_mode in self.modes
                and ctx.attributes is not None):
            no_car = self._car_unavailable_many(ctx, persons)
            allowed[no_car, self.modes.index(self.car_mode)] = False
        count = allowed.sum(axis=1)
        change = has_legs & (count > 0)
        k = np.minimum((u * count).astype(np.int64), np.maximum(count - 1, 0))
        pick = (np.cumsum(allowed, axis=1) <= k[:, None]).sum(axis=1)  # k-th allowed mode
        new_mode = own[np.minimum(pick, len(own) - 1)]
        legs = batch.plan_of_leg()
        changed = change[legs]
        if not changed.any():
            return batch
        leg_mode = np.where(changed, new_mode[legs], batch.leg_mode).astype(np.int32)
        return _set_routes(replace(batch, modes=tuple(modes), leg_mode=leg_mode,
                                   leg_travel_time=np.where(changed, np.nan, batch.leg_travel_time)),
                           clear=changed)



# ------------------------------------------------------------------ batches

def _set_routes(batch: PlanBatch, clear: np.ndarray, new: Optional[dict] = None) -> PlanBatch:
    """``batch`` with the routes of legs ``clear`` (bool per leg) removed and
    ``new`` (``{leg: Route or TransitPassengerRoute}``) set; other legs keep theirs."""
    new = new or {}
    n_legs = batch.n_legs
    is_new = np.zeros(n_legs, dtype=bool)
    if new:
        is_new[list(new)] = True
    keep = batch.leg_routed & ~clear & ~is_new
    keep_net = keep & ~batch.leg_pt_route
    keep_pt = keep & batch.leg_pt_route
    new_legs = sorted(new)
    new_pt = np.zeros(n_legs, dtype=bool)
    new_pt[[leg for leg in new_legs if new[leg].__class__ is TransitPassengerRoute]] = True
    new_net = is_new & ~new_pt

    def rebuild(old_offsets, keep_mask, new_mask, old_values, new_data, width):
        lengths = np.where(keep_mask, np.diff(old_offsets), 0)
        new_values = np.frombuffer(b"".join(new_data), dtype=_I32).reshape(-1, width)
        if new_mask.any():
            lengths[new_mask] = [len(d) // (4 * width) for d in new_data]
        offsets = np.zeros(n_legs + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        values = np.empty((offsets[-1], width), dtype=np.int32)
        dst, _ = csr_gather(offsets, np.flatnonzero(keep_mask))
        src, _ = csr_gather(old_offsets, np.flatnonzero(keep_mask))
        values[dst] = old_values[src]
        dst, _ = csr_gather(offsets, np.flatnonzero(new_mask))
        values[dst] = new_values
        return offsets, values

    route_offsets, links = rebuild(
        batch.route_offsets, keep_net, new_net, batch.route_links.reshape(-1, 1),
        [new[leg].data for leg in new_legs if new_net[leg]], 1)
    old_rides = np.column_stack([batch.ride_board_stop, batch.ride_alight_stop])
    ride_offsets, rides = rebuild(
        batch.ride_offsets, keep_pt, new_pt, old_rides,
        [new[leg].data for leg in new_legs if new_pt[leg]], 2)
    routed = keep | is_new
    rtime = np.where(keep, batch.leg_route_time, np.nan)
    rdist = np.where(keep, batch.leg_route_distance, np.nan)
    if new_legs:
        rtime[new_legs] = [new[leg].travel_time for leg in new_legs]
        rdist[new_legs] = [new[leg].distance for leg in new_legs]
    return replace(
        batch, leg_routed=routed, leg_pt_route=(keep_pt | new_pt), route_offsets=route_offsets,
        route_links=links[:, 0].copy(), ride_offsets=ride_offsets,
        ride_board_stop=rides[:, 0].copy(), ride_alight_stop=rides[:, 1].copy(),
        leg_route_time=rtime, leg_route_distance=rdist)


def _mode_kinds(batch: PlanBatch, modes) -> tuple[np.ndarray, np.ndarray]:
    """Per leg: is its mode routed (network or transit), is it transit."""
    routed = np.array([modes.is_routed_mode(m) for m in batch.modes], dtype=bool)
    transit = np.array([modes.is_transit_mode(m) for m in batch.modes], dtype=bool)
    if not batch.n_legs:
        return np.zeros(0, bool), np.zeros(0, bool)
    return routed[batch.leg_mode], transit[batch.leg_mode]


def clear_routes_batch(batch: PlanBatch, ctx: ReplanningContext, keep_network: bool) -> PlanBatch:
    """Vectorised ``_clear_routes``: drop every route, except (with
    ``keep_network``) routed-mode legs' routes of the right type."""
    _, transit = _mode_kinds(batch, ctx.modes)
    # as _has_route: a route of the mode's type (teleported legs' routes are
    # dropped later, by routing)
    valid = batch.leg_routed & (batch.leg_pt_route == transit)
    clear = batch.leg_routed & ~(keep_network & valid)
    return _set_routes(batch, clear) if clear.any() else batch


def route_batch(batch: PlanBatch, ctx: ReplanningContext, persons=None) -> PlanBatch:
    """Vectorised :func:`route_plans`: route every routed-mode leg without a
    route of the right type and drop teleported legs' routes, for all plans
    of ``batch`` (``persons``: each plan's person index, for per-person
    transit weights and teleport speeds).

    Every plan's clock is walked as :func:`route_plans` walks it (with the
    same arithmetic, so departure times are equal), leg position by leg
    position across plans; the legs to route at each position are routed
    with one ``route_many`` call per mode. The routes are therefore the ones
    :func:`route_plans` returns.
    """
    modes = ctx.modes
    net = modes.network
    routed, transit = _mode_kinds(batch, modes)
    valid = batch.leg_routed & (batch.leg_pt_route == transit)
    need = routed & ~valid
    drop = ~routed & batch.leg_routed
    if not (need.any() or drop.any()):
        return batch
    plan_of_leg = batch.plan_of_leg()
    active = np.zeros(batch.n_plans, dtype=bool)
    active[plan_of_leg[need | drop]] = True
    tw = ctx.transit_weights
    use_weights = tw is not None and tw.any and persons is not None
    persons = None if persons is None else np.asarray(persons, dtype=np.int64)
    xy = batch.resolved_act_xy(net)
    n_legs = np.diff(batch.leg_offsets)
    act_off, leg_off = batch.act_offsets, batch.leg_offsets
    end, dur = batch.act_end_time, batch.act_duration
    clock = np.zeros(batch.n_plans)
    new: dict[int, object] = {}
    teleport_params = {m: modes.teleported[m] for m in batch.modes if m in modes.teleported}
    for j in range(int(n_legs[active].max(initial=0))):
        p = np.flatnonzero(active & (n_legs > j))
        a = act_off[p] + j
        t = clock[p]
        e, d = end[a], dur[a]
        t = np.where(~np.isnan(e), np.maximum(t, e), np.where(~np.isnan(d), t + d, t))
        legs = leg_off[p] + j
        # legs that keep their route
        ok = routed[legs] & ~need[legs]
        t[ok] = t[ok] + batch.leg_route_time[legs[ok]]
        # teleported legs (their routes, if any, are dropped below)
        tele = np.flatnonzero(~routed[legs])
        if len(tele):
            t[tele] = t[tele] + _teleport_times(batch, legs[tele], a[tele], xy, modes,
                                                teleport_params,
                                                None if persons is None else persons[p[tele]])
        # legs to route now, one route_many call per mode
        todo = np.flatnonzero(need[legs])
        by_mode: dict[int, list] = {}
        for k in todo.tolist():
            by_mode.setdefault(int(batch.leg_mode[legs[k]]), []).append(k)
        for code, ks in by_mode.items():
            mode = batch.modes[code]
            ks = np.array(ks, dtype=np.int64)
            o, dst = a[ks], a[ks] + 1
            router = ctx.router(mode)
            if modes.is_transit_mode(mode):
                args = ([tuple(v) for v in xy[o].tolist()], [tuple(v) for v in xy[dst].tolist()],
                        t[ks].tolist())
                if use_weights:
                    w = [tw.of(int(i)) for i in persons[p[ks]]]
                    routes = (router.route_many_xy(*args, weights=w) if any(x is not None for x in w)
                              else router.route_many_xy(*args))
                else:
                    routes = router.route_many_xy(*args)
            else:
                links = batch.act_link
                routes = route_many_to_links(
                    router, net, mode,
                    batch.act_node[o].tolist(), [None if v < 0 else v for v in links[o].tolist()],
                    batch.act_node[dst].tolist(), [None if v < 0 else v for v in links[dst].tolist()],
                    t[ks].tolist())
            for k, route in zip(ks.tolist(), routes):
                new[int(legs[k])] = route
                t[k] = t[k] + route.travel_time
        clock[p] = t
    return _set_routes(batch, clear=drop | need, new=new)


def _teleport_times(batch, legs, origin_acts, xy, modes, params, persons) -> np.ndarray:
    """Travel times of teleported ``legs`` (as ``Modes.teleport``: the fixed
    time if the leg has one, else beeline * factor / speed)."""
    out = np.empty(len(legs))
    fixed = batch.leg_travel_time[legs]
    o, d = xy[origin_acts], xy[origin_acts + 1]
    for k, (leg, f) in enumerate(zip(legs.tolist(), fixed.tolist())):
        if f == f:
            out[k] = f
            continue
        mode = batch.modes[batch.leg_mode[leg]]
        p = params[mode] if persons is None else modes.teleport_params(mode, int(persons[k]))
        distance = math.hypot(d[k, 0] - o[k, 0], d[k, 1] - o[k, 1]) * p.beeline_distance_factor
        out[k] = distance / p.speed
    return out
