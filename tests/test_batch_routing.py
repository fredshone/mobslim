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
from mobslim.population import Leg
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
