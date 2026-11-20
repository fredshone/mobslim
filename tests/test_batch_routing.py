"""Batched and compiled routing: ``route_many``, ``RustRouter`` and batched
rerouting in replanning. Everything here must give the same routes as one
``DijkstraRouter.route`` call per leg."""

import dataclasses
import pickle
import random
from functools import partial

import numpy as np
import pytest

from mobslim import Scenario, grid_network, route_scenario
from mobslim.population import Leg, Person, Plan, Population
from mobslim.replanning import (
    ChangeTripMode,
    PlanStrategy,
    RandomSelector,
    ReplanningContext,
    Reroute,
    SerialRunner,
    StrategyManager,
    TimeAllocationMutator,
)
from mobslim.replanning.modules import route_plans
from mobslim.routing import (
    DijkstraRouter,
    FreeSpeedTravelTime,
    NoRouteError,
    RustRouter,
    TravelTimeTable,
    default_router,
    route_many,
    rust_router_available,
)
from mobslim.routing.disutility import TravelDisutility
from tests.conftest import routed

needs_rust = pytest.mark.skipif(not rust_router_available(),
                                reason="mobslim_router (Rust router) not built")


def congested(network, seed=0, peak=(28, 40), factor=3.0):
    """Free speed, slowed by up to ``factor`` on random links in ``peak`` bins."""
    free = TravelTimeTable.free_speed(network)
    times = free.times.copy()
    rng = np.random.default_rng(seed)
    a, b = peak
    times[:, a:b] *= rng.uniform(1.0, factor, size=(network.n_links, b - a))
    return TravelTimeTable(times, free.bin_size)


def irregular_grid(size=12, seed=0):
    """A grid with random lengths and some fast links: few tied routes."""
    g = grid_network(size, spacing=100.0)
    rng = np.random.default_rng(seed)
    length = g.length * rng.uniform(1.0, 1.6, g.n_links)
    speed = np.where(rng.random(g.n_links) < 0.15, g.freespeed * 2.5, g.freespeed)
    return dataclasses.replace(g, length=length, freespeed=speed, _cache={})


def queries(network, n=300, seed=1, shared=True):
    """Random (origin, destination, departure) triples, including o == d and
    groups sharing an origin and departure time."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        o, d = rng.randrange(network.n_nodes), rng.randrange(network.n_nodes)
        out.append((o, d, rng.choice([25200.0, 28800.0, 30000.5]) if shared else rng.uniform(0, 86400)))
    out += [(5, 5, 100.0), (0, 1, 7 * 3600.0), (0, 1, 7 * 3600.0)]
    return out


# ------------------------------------------------------- Python route_many


@pytest.mark.parametrize("static", [True, False])
def test_python_route_many_equals_route(static):
    net = irregular_grid()
    table = TravelTimeTable.free_speed(net) if static else congested(net)
    router = DijkstraRouter(net, "car", table)
    assert router.is_static == static
    qs = queries(net)
    o, d, t = zip(*qs)
    assert router.route_many(o, d, t) == [router.route(*q) for q in qs]


def test_static_routes_ignore_departure_time():
    net = irregular_grid()
    router = DijkstraRouter(net, "car", TravelTimeTable.free_speed(net))
    assert router.route(0, 77, 0.0) == router.route(0, 77, 12345.6)
    assert FreeSpeedTravelTime(net).is_static and not congested(net).is_static


def test_route_many_falls_back_to_route():
    class OneByOne:
        def route(self, o, d, t):
            return (o, d, t)

    assert route_many(OneByOne(), [1, 2], [3, 4], [5.0, 6.0]) == [(1, 3, 5.0), (2, 4, 6.0)]


# ------------------------------------------------------------- RustRouter


@needs_rust
@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("threads", [1, 4])
def test_rust_matches_python(static, shared, threads):
    net = irregular_grid()
    table = TravelTimeTable.free_speed(net) if static else congested(net)
    qs = queries(net, shared=shared)
    o, d, t = zip(*qs)
    expected = [DijkstraRouter(net, "car", table).route(*q) for q in qs]
    rust = RustRouter(net, "car", table, threads=threads)
    assert rust.route_many(o, d, t) == expected
    assert [rust.route(*q) for q in qs] == expected


@needs_rust
def test_rust_matches_python_on_a_tied_grid():
    """A regular grid has many equal-cost routes: tie-breaking must match too."""
    net = grid_network(8, spacing=50.0)
    table = congested(net, factor=1.5)
    qs = queries(net, shared=False)
    o, d, t = zip(*qs)
    assert RustRouter(net, "car", table).route_many(o, d, t) == [
        DijkstraRouter(net, "car", table).route(*q) for q in qs
    ]


@needs_rust
def test_rust_free_speed_travel_time():
    net = irregular_grid()
    qs = queries(net)
    py = DijkstraRouter(net, "car", FreeSpeedTravelTime(net))
    assert RustRouter(net, "car", FreeSpeedTravelTime(net)).route_many(*zip(*qs)) == [
        py.route(*q) for q in qs
    ]


@needs_rust
def test_rust_no_route():
    from tests.test_routing import diamond

    net = diamond()
    router = RustRouter(net, "car", FreeSpeedTravelTime(net))
    c = net.node_index("c")
    with pytest.raises(NoRouteError):
        router.route(c, 0, 0)
    with pytest.raises(NoRouteError, match=f"node {c} to node 0"):
        router.route_many([0, c], [c, 0], [0.0, 0.0])


@needs_rust
def test_rust_rejects_other_travel_times():
    class Custom:
        def link_travel_time(self, link, time):
            return 1.0

        def link_travel_times(self, time):
            return [1.0] * 10

    net = irregular_grid(4)
    with pytest.raises(TypeError, match="TravelTimeTable"):
        RustRouter(net, "car", Custom())
    assert isinstance(default_router(net, "car", Custom()), DijkstraRouter)
    assert isinstance(default_router(net, "car", FreeSpeedTravelTime(net)), RustRouter)


@needs_rust
@pytest.mark.parametrize("static", [True, False])
def test_alt_finds_the_same_least_travel_times(static):
    net = irregular_grid(15)
    table = TravelTimeTable.free_speed(net) if static else congested(net, factor=1.5)
    qs = queries(net, shared=False)
    dijkstra = [DijkstraRouter(net, "car", table).route(*q) for q in qs]
    alt = RustRouter(net, "car", table, algorithm="alt", landmarks=8)
    assert len(alt.landmark_nodes) == 8
    for a, b in zip(alt.route_many(*zip(*qs)), dijkstra):
        assert a.travel_time == pytest.approx(b.travel_time, rel=1e-9, abs=1e-9)


@needs_rust
def test_alt_landmarks_are_built_once_per_network_and_mode():
    net = irregular_grid(6)
    a = RustRouter(net, "car", congested(net, 1), algorithm="alt", landmarks=4)
    b = RustRouter(net, "car", congested(net, 2), algorithm="alt", landmarks=4)
    assert a.landmark_nodes == b.landmark_nodes
    assert sum(1 for k in net._cache if k[0] == "rust_router") == 1


def _route_plan_by_plan(scenario, transit_weights=None):
    """``route_scenario`` as it was: every held plan as Plan objects, given
    coordinates by ``locate_activities`` and routed by ``route_plans``."""
    from mobslim.activity_links import ActivityLinkParams, assign_activity_links
    from mobslim.population import locate_activities
    from mobslim.replanning.modules import routing_groups

    scenario = assign_activity_links(scenario, ActivityLinkParams())
    ctx = ReplanningContext.from_scenario(scenario, TravelTimeTable.free_speed(scenario.network), 0,
                                          transit_weights=transit_weights)
    pop, net = scenario.population, scenario.network
    weights = persons = None
    if (groups := routing_groups(ctx)) is not None:
        codes, group_weights, group_persons = groups
        code_of = {int(j): int(codes[i]) for i in range(len(pop))
                   for j in pop.plan_ids[i, :pop.n_plans[i]]}
        live = pop.live_plan_rows().tolist()
        weights = [group_weights[code_of[j]] for j in live]
        persons = [group_persons[code_of[j]] for j in live]
    return pop.map_plan_store(
        lambda plans: route_plans([locate_activities(p, net) for p in plans], ctx, weights, persons))


def _held_plans(population):
    return [list(person.plans) for person in population]


@pytest.mark.parametrize("case", ["car", "car_without_coordinates", "car_routed", "pt", "pt_weights"])
def test_route_scenario_equals_routing_plan_by_plan(case, equil_scenario):
    from tests.test_transit_router_scoring import _pt_tutorial_split
    from tests.test_transit_scenarios import load

    weights = None
    if case.startswith("car"):
        scenario = equil_scenario
        if case == "car_without_coordinates":  # node-only activities, several plans each
            net = irregular_grid(6)
            rng = random.Random(4)
            plans = []
            for _ in range(40):
                a, b = rng.sample(range(net.n_nodes), 2)
                plans.append([Plan.builder().act("home", a, end_time=rng.uniform(6, 9) * 3600)
                              .leg("car").act("work", b, duration=8 * 3600.0).leg("car")
                              .act("home", a).build() for _ in range(2)])
            scenario = Scenario(net, Population([Person(i, p) for i, p in enumerate(plans)]))
        if case == "car_routed":
            scenario = route_scenario(scenario)
    elif case == "pt":
        scenario = load("pt_tutorial")
    else:
        scenario, weights = _pt_tutorial_split()
    got = route_scenario(scenario, transit_weights=weights).population
    assert _held_plans(got) == _held_plans(_route_plan_by_plan(scenario, weights))
    assert got.n_plans.tolist() == scenario.population.n_plans.tolist()


@needs_rust
def test_a_tables_router_columns_are_computed_once(monkeypatch):
    """Routers of several modes on one table share its columns; a pickled
    table (as sent to workers) computes them anew."""
    from mobslim.routing import compiled

    net = irregular_grid(5)
    table = congested(net)
    calls = []
    real = compiled.travel_time_columns
    monkeypatch.setattr(compiled, "travel_time_columns", lambda *a: calls.append(1) or real(*a))
    first = RustRouter(net, "car", table)
    second = RustRouter(net, "car", table, algorithm="dijkstra")
    assert len(calls) == 1 and first.algorithm == "alt"
    assert second.route_many([0], [24], [8 * 3600.0]) == first.route_many([0], [24], [8 * 3600.0])
    RustRouter(net, "car", pickle.loads(pickle.dumps(table)))
    assert len(calls) == 2


@needs_rust
def test_alt_landmarks_when_node_0_has_no_links_of_the_mode():
    """A car + pt network whose first nodes are pt stops: the landmarks are
    still chosen (and bound) on the car links, not seeded from node 0."""
    from mobslim.network import NetworkBuilder

    grid = irregular_grid(8)
    b = NetworkBuilder()
    for s in ("stop0", "stop1"):
        b.add_node(s, -500.0, 0.0 if s == "stop0" else 100.0)
    b.add_link("pt0", "stop0", "stop1", 100.0, 10.0, 1000.0, modes=("pt",))
    b.add_link("pt1", "stop1", "stop0", 100.0, 10.0, 1000.0, modes=("pt",))
    for i in range(grid.n_nodes):
        b.add_node(grid.node_ids.id(i), *grid.node_xy[i])
    for k in range(grid.n_links):
        b.add_link(grid.link_ids.id(k), grid.node_ids.id(int(grid.from_node[k])),
                   grid.node_ids.id(int(grid.to_node[k])), grid.length[k], grid.freespeed[k],
                   grid.capacity[k], grid.lanes[k], grid.link_modes[k])
    net = b.build()
    table = congested(net, factor=1.5)
    alt = RustRouter(net, "car", table, algorithm="alt", landmarks=4)
    assert len(alt.landmark_nodes) == 4 and min(alt.landmark_nodes) >= 2  # not the stops
    rng = random.Random(2)
    qs = [(rng.randrange(2, net.n_nodes), rng.randrange(2, net.n_nodes), rng.uniform(0, 86400))
          for _ in range(200)]
    dijkstra = [DijkstraRouter(net, "car", table).route(*q) for q in qs]
    for a, d in zip(alt.route_many(*zip(*qs)), dijkstra):
        assert a.travel_time == pytest.approx(d.travel_time, rel=1e-9, abs=1e-9)
    with pytest.raises(NoRouteError):  # a stop has no car links
        alt.route(0, 5, 0.0)


@needs_rust
def test_alt_rejects_times_below_the_lower_bound():
    net = irregular_grid(4)
    free = TravelTimeTable.free_speed(net)
    with pytest.raises(ValueError, match="free-speed"):
        RustRouter(net, "car", TravelTimeTable(free.times * 0.5), algorithm="alt")
    with pytest.raises(ValueError, match="algorithm"):
        RustRouter(net, "car", free, algorithm="bogus")


@needs_rust
def test_rust_router_factories_pickle():
    net = irregular_grid(4)
    factory = pickle.loads(pickle.dumps(partial(RustRouter, algorithm="alt", threads=2)))
    router = factory(net, "car", TravelTimeTable.free_speed(net))
    assert router.algorithm == "alt" and router.threads == 2
    ctx = ReplanningContext.from_scenario(
        Scenario(net, routed_population(net)), TravelTimeTable.free_speed(net), 1, factory
    )
    ctx.router("car")
    assert pickle.loads(pickle.dumps(ctx)).router("car").algorithm == "alt"


def routed_population(net):
    from mobslim import Plan, Population

    return Population.from_plans({
        "p": Plan.builder().act("h", 0, end_time=3600).leg("car").act("w", 5).build()
    })


# ------------------------------------------------- batched rerouting


ROUTERS = ["python", pytest.param("rust", marks=needs_rust)]
FACTORIES = {"python": DijkstraRouter, "rust": RustRouter}


@pytest.fixture
def equil_ctx(equil_scenario):
    scenario = routed(equil_scenario)
    return scenario, congested(scenario.network)


@pytest.mark.parametrize("router", ROUTERS)
def test_route_plans_equals_rerouting_plan_by_plan(equil_ctx, router):
    scenario, table = equil_ctx
    ctx = ReplanningContext.from_scenario(scenario, table, 1, FACTORIES[router])
    reference = ReplanningContext.from_scenario(scenario, table, 1, DijkstraRouter)
    plans = scenario.population.selected_plans()
    # mixed modes: some plans walk their second leg (teleported, keeps no route)
    mixed = [
        p.with_elements([Leg("walk") if i == 3 else e for i, e in enumerate(p.elements)])
        if k % 3 == 0 and len(p.elements) > 3 else p
        for k, p in enumerate(plans)
    ]
    cleared = [Reroute().mutate(p, ctx.batched(), None) for p in mixed]
    assert all(leg.route is None for p in cleared for leg in p.legs)
    batched = route_plans(cleared, ctx)
    assert batched == [Reroute().mutate(p, reference, None) for p in mixed]


def test_route_plans_returns_routed_plans_unchanged(equil_ctx):
    scenario, table = equil_ctx
    ctx = ReplanningContext.from_scenario(scenario, table, 1)
    plans = scenario.population.selected_plans()
    assert all(a is b for a, b in zip(route_plans(plans, ctx), plans))


def mixed_manager():
    return StrategyManager([
        (0.4, PlanStrategy(RandomSelector())),
        (0.2, PlanStrategy(RandomSelector(), [Reroute()])),
        (0.2, PlanStrategy(RandomSelector(), [TimeAllocationMutator(), Reroute()])),
        (0.2, PlanStrategy(RandomSelector(), [ChangeTripMode(["car", "walk"]), Reroute()])),
    ])


@pytest.mark.parametrize("router", ROUTERS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_batched_replanning_equals_unbatched(equil_ctx, router, seed):
    scenario, table = equil_ctx
    pop = scenario.population.with_selected_scores([0.0] * len(scenario.population))
    sc = Scenario(scenario.network, pop)
    manager = mixed_manager()
    ctx = ReplanningContext.from_scenario(sc, table, 1, FACTORIES[router])
    reference = ReplanningContext.from_scenario(sc, table, 1, DijkstraRouter)
    batched = SerialRunner().run(pop, manager, ctx, seed)
    unbatched = SerialRunner(batch_routing=False).run(pop, manager, reference, seed)
    assert list(batched) == list(unbatched)
    assert list(batched) != list(pop)
    assert all(
        leg.route is not None
        for person in batched for plan in person.plans for leg in plan.legs
        if sc.modes.is_network_mode(leg.mode)
    )


@pytest.mark.parametrize("router", ROUTERS)
def test_route_scenario_with_either_router(equil_scenario, router):
    expected = route_scenario(equil_scenario, router_factory=DijkstraRouter)
    got = route_scenario(equil_scenario, router_factory=FACTORIES[router])
    assert list(got.population) == list(expected.population)


# ------------------------------------------- time and distance (MATSim's disutility)

COST = TravelDisutility({"car": 6.0 / 3600}, {"car": 0.004})


def _factors(n, seed=3):
    """Distance factors with repeats (queries that may share a search)."""
    return np.random.default_rng(seed).choice([0.05, 1.0, 1.0, 7.0], size=n)


@needs_rust
@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("threads", [1, 4])
def test_rust_matches_python_with_a_distance_cost(static, threads):
    net = irregular_grid()
    table = TravelTimeTable.free_speed(net) if static else congested(net)
    qs = queries(net)
    o, d, t = zip(*qs)
    f = _factors(len(qs))
    py = DijkstraRouter(net, "car", table, disutility=COST)
    expected = [py.route(*q, factor=x) for q, x in zip(qs, f)]
    assert py.route_many(o, d, t, f) == expected
    rust = RustRouter(net, "car", table, algorithm="dijkstra", threads=threads, disutility=COST)
    assert rust.route_many(o, d, t, f) == expected
    assert [rust.route(*q, factor=x) for q, x in zip(qs, f)] == expected


@needs_rust
@pytest.mark.parametrize("static", [True, False])
def test_alt_finds_the_same_least_costs_with_a_distance_cost(static):
    net = irregular_grid(15)
    table = TravelTimeTable.free_speed(net) if static else congested(net, factor=1.5)
    qs = queries(net, shared=False)
    f = _factors(len(qs))
    (a, b) = COST.costs("car")
    dijkstra = DijkstraRouter(net, "car", table, disutility=COST)
    alt = RustRouter(net, "car", table, algorithm="alt", landmarks=8, disutility=COST)
    for q, x, r in zip(qs, f, alt.route_many(*zip(*qs), f)):
        best = dijkstra.route(*q, factor=x)
        assert a * r.travel_time + b * x * r.distance == pytest.approx(
            a * best.travel_time + b * x * best.distance, rel=1e-9, abs=1e-9)


@needs_rust
def test_matsims_bound_factor_gives_dearer_routes_only_when_it_overestimates():
    net = irregular_grid(15)
    table = congested(net, factor=1.5)
    qs = queries(net, shared=False)
    o, d, t = zip(*qs)
    (a, b) = COST.costs("car")
    matsim = dataclasses.replace(COST, bound_factor=1.0)

    def costs(disutility, f, algorithm="alt"):
        r = RustRouter(net, "car", table, algorithm=algorithm, landmarks=8, disutility=disutility)
        return np.array([a * x.travel_time + b * y * x.distance
                         for x, y in zip(r.route_many(o, d, t, f), f)])

    low = np.full(len(qs), 0.05)  # below 1: the bound overestimates
    exact = costs(COST, low)
    assert np.all(costs(matsim, low) >= exact - 1e-9)
    assert np.any(costs(matsim, low) > exact + 1e-6)
    high = np.full(len(qs), 7.0)  # above 1: still a lower bound, still least-cost
    assert costs(matsim, high) == pytest.approx(costs(COST, high), rel=1e-9, abs=1e-9)
    # Dijkstra has no bound: unaffected
    assert costs(matsim, low, "dijkstra") == pytest.approx(exact, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("router", ROUTERS)
def test_a_distance_cost_takes_shorter_slower_routes(router):
    net = irregular_grid(15)
    qs = queries(net, shared=False)
    by_time = FACTORIES[router](net, "car", TravelTimeTable.free_speed(net)).route_many(*zip(*qs))
    by_cost = FACTORIES[router](net, "car", TravelTimeTable.free_speed(net),
                                disutility=COST).route_many(*zip(*qs))
    assert sum(r.distance for r in by_cost) < sum(r.distance for r in by_time)
    assert sum(r.travel_time for r in by_cost) > sum(r.travel_time for r in by_time)


def test_travel_disutility_from_scoring_and_its_factors():
    from mobslim import ModeParams, ScoringParams

    params = ScoringParams(performing=6.0, marginal_utility_of_money=2.0,
                           modes={"car": ModeParams(marginal_utility_of_traveling=-1.0,
                                                    monetary_distance_rate=-2e-4)})
    d = TravelDisutility.from_scoring(params, ["car", "walk"], randomness=3.0, seed=5)
    assert d.costs("car") == pytest.approx((7.0 / 3600, 4e-4))
    assert d.costs("bike") == (1.0, 0.0)  # not listed: travel time
    f = d.factors(np.arange(200_000), 1)
    # lognormal, normalised to mean 1 as MATSim does: log f ~ N(-sigma^2 / 2, sigma)
    assert np.log(f).mean() == pytest.approx(-4.5, abs=0.03)
    assert np.log(f).std() == pytest.approx(3.0, abs=0.03)
    np.testing.assert_array_equal(f[:10], d.factors(np.arange(10), 1))  # per person, deterministic
    assert not np.array_equal(f[:10], d.factors(np.arange(10), 2))  # redrawn each iteration
    assert dataclasses.replace(d, randomness=0.0).factors(np.arange(10), 1) is None


@pytest.mark.parametrize("router", ROUTERS)
def test_batch_routing_draws_each_persons_factor_as_routing_plan_by_plan(equil_ctx, router):
    from mobslim import PlanBatch
    from mobslim.replanning.modules import route_batch
    from tests.test_planstore import batches_equal

    scenario, table = equil_ctx
    cost = TravelDisutility({"car": 6.0 / 3600}, {"car": 1e-3}, randomness=2.0, seed=1)
    ctx = ReplanningContext.from_scenario(scenario, table, 3,
                                          partial(FACTORIES[router], disutility=cost))
    plans = [Reroute().mutate(p, ctx.batched(), None)
             for p in scenario.population.selected_plans()]
    persons = np.arange(len(plans))
    batched = route_batch(PlanBatch.from_plans(plans), ctx, persons)
    assert batches_equal(batched, PlanBatch.from_plans(route_plans(plans, ctx,
                                                                   persons=list(persons))))
