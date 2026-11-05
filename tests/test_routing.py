import pickle

import numpy as np
import pytest

from mobslim import (
    NetworkBuilder,
    Plan,
    Population,
    Scenario,
    grid_network,
    route_scenario,
)
from mobslim.events import LinkEnter, LinkLeave
from mobslim.planbatch import PlanBatch
from mobslim.population import Leg, Route
from mobslim.routing import (
    DijkstraRouter,
    FreeSpeedTravelTime,
    NoRouteError,
    RustRouter,
    TravelTimeCollector,
    TravelTimeTable,
    arrival_link,
    route_many_to_links,
    rust_router_available,
)


def diamond():
    """Two parallel links a->b ('fast', 'slow') then b->c."""
    b = NetworkBuilder()
    for i, n in enumerate("abc"):
        b.add_node(n, i * 100, 0)
    b.add_link("fast", "a", "b", 100, 20, 1000)  # 5 s
    b.add_link("slow", "a", "b", 100, 10, 1000)  # 10 s
    b.add_link("bc", "b", "c", 100, 10, 1000)  # 10 s
    return b.build()


def test_shortest_path_uses_fastest_parallel_link():
    net = diamond()
    router = DijkstraRouter(net, "car", FreeSpeedTravelTime(net))
    route = router.route(net.node_index("a"), net.node_index("c"), 0)
    assert [net.link_ids.id(k) for k in route.links] == ["fast", "bc"]
    assert route.travel_time == pytest.approx(15)
    assert route.distance == pytest.approx(200)


def test_time_dependent_routing():
    net = diamond()
    table = TravelTimeTable.free_speed(net, bin_size=900, horizon=3600)
    times = table.times.copy()
    times[net.link_index("fast"), 1] = 1000  # congested 00:15-00:30
    table = TravelTimeTable(times, 900)
    router = DijkstraRouter(net, "car", table)
    a, c = net.node_index("a"), net.node_index("c")
    assert net.link_ids.id(router.route(a, c, 0).links[0]) == "fast"
    assert net.link_ids.id(router.route(a, c, 1000).links[0]) == "slow"


def test_same_origin_and_destination():
    net = diamond()
    route = DijkstraRouter(net, "car", FreeSpeedTravelTime(net)).route(0, 0, 0)
    assert route.links == () and route.travel_time == 0


def test_no_route():
    net = diamond()
    with pytest.raises(NoRouteError):
        DijkstraRouter(net, "car", FreeSpeedTravelTime(net)).route(net.node_index("c"), 0, 0)


def test_grid_route_length():
    net = grid_network(size=4, spacing=50)
    route = DijkstraRouter(net, "car", FreeSpeedTravelTime(net)).route(
        net.node_index((0, 0)), net.node_index((4, 4)), 0
    )
    assert len(route.links) == 8 and route.distance == 400


def test_travel_time_collector_bins_by_entry_time():
    net = diamond()
    c = TravelTimeCollector(net, bin_size=900, horizon=3600)
    fast = net.link_index("fast")
    for person, (t0, t1) in enumerate([(0, 20), (100, 140), (1000, 1100)]):
        c.handle(LinkEnter(t0, person, fast))
        c.handle(LinkLeave(t1, person, fast))
    table = c.table()
    assert table.link_travel_time(fast, 0) == pytest.approx(30)  # mean of 20, 40
    assert table.link_travel_time(fast, 950) == pytest.approx(100)
    assert table.link_travel_time(fast, 2000) == pytest.approx(5)  # unobserved: free speed
    assert table.link_travel_time(fast, 1e9) == pytest.approx(5)  # beyond horizon: last bin


def test_travel_time_never_below_free_speed():
    net = diamond()
    c = TravelTimeCollector(net, horizon=3600)
    c.handle(LinkEnter(0, 0, 0))
    c.handle(LinkLeave(1, 0, 0))  # faster than free speed (5 s)
    assert c.table().link_travel_time(0, 0) == pytest.approx(5)


def test_table_blend_and_pickle():
    net = diamond()
    a = TravelTimeTable.free_speed(net, horizon=3600)
    b = TravelTimeTable(a.times * 3, a.bin_size)
    mid = a.blend(b, 0.5)
    assert np.allclose(mid.times, a.times * 2)
    clone = pickle.loads(pickle.dumps(mid))
    assert clone.link_travel_time(0, 0) == mid.link_travel_time(0, 0)


def test_link_travel_times_vector_matches_scalar():
    import numpy as np

    times = np.arange(12, dtype=float).reshape(3, 4) + 1
    table = TravelTimeTable(times, bin_size=10.0)
    for t in (-5.0, 0.0, 9.99, 10.0, 35.0, 1e9):
        vec = table.link_travel_times(t)
        assert len(vec) == 3
        assert [table.link_travel_time(k, t) for k in range(3)] == list(vec)
    assert table.link_travel_times(-5.0) == (1.0, 5.0, 9.0)  # before 0: first bin
    assert table.link_travel_times(1e9) == (4.0, 8.0, 12.0)  # after horizon: last bin


# -- legs end on their activity's link (MATSim's routes; docs/design.md section 5)

def side_entry():
    """Node h has two ways in: straight from a ('ah', 10 s) or via b ('ab'
    then 'bh', 10 s each). An activity on link 'bh' must be reached via b."""
    b = NetworkBuilder()
    for i, n in enumerate("abh"):
        b.add_node(n, i * 100, 0)
    b.add_link("ah", "a", "h", 200, 20, 1000)  # 10 s
    b.add_link("ab", "a", "b", 100, 10, 1000)  # 10 s
    b.add_link("bh", "b", "h", 100, 10, 1000)  # 10 s
    b.add_link("ha", "h", "a", 200, 20, 1000)  # 10 s
    b.add_link("bus", "b", "h", 100, 10, 1000, modes=("bus",))
    return b.build()


ROUTERS = {"python": DijkstraRouter, "rust": RustRouter}
ROUTER_KINDS = ["python", pytest.param("rust", marks=pytest.mark.skipif(
    not rust_router_available(), reason="mobslim_router (Rust router) not built"))]


@pytest.mark.parametrize("kind", ROUTER_KINDS)
def test_leg_ends_by_driving_through_the_activity_link(kind):
    net = side_entry()
    a, h = net.node_index("a"), net.node_index("h")
    ha, bh, ah = (net.link_index(k) for k in ("ha", "bh", "ah"))
    by_node, by_link, no_link = route_many_to_links(
        ROUTERS[kind](net, "car", FreeSpeedTravelTime(net)), net, "car",
        [a, a, a], [ha, ha, ha], [h, h, h], [ah, bh, None], [0.0, 0.0, 0.0])
    assert [net.link_ids.id(k) for k in by_node.links] == ["ah"]  # the quickest way in
    assert [net.link_ids.id(k) for k in by_link.links] == ["ab", "bh"]
    assert by_link.travel_time == pytest.approx(20) and by_link.distance == pytest.approx(200)
    assert no_link == by_node


def test_when_a_leg_does_not_use_the_activity_link():
    net = side_entry()
    a, h = net.node_index("a"), net.node_index("h")
    bh, bus, ha = net.link_index("bh"), net.link_index("bus"), net.link_index("ha")
    assert arrival_link(net, "car", a, ha, h, bh) == bh
    assert arrival_link(net, "car", h, bh, h, bh) is None  # starts on it: empty route
    assert arrival_link(net, "car", h, None, h, bh) is None  # already at the node
    assert arrival_link(net, "car", a, ha, h, None) is None  # no link
    assert arrival_link(net, "car", a, ha, h, bus) is None  # not a car link


@pytest.mark.parametrize("kind", ROUTER_KINDS)
def test_route_scenario_ends_legs_on_activity_links(kind):
    net = side_entry()
    a, h = net.node_index("a"), net.node_index("h")
    ha, bh = net.link_index("ha"), net.link_index("bh")
    plan = (Plan.builder().act("w", a, end_time=0.0, link=ha).leg("car")
            .act("h", h, link=bh).build())
    scenario = route_scenario(Scenario(net, Population.from_plans({"p": plan})),
                              router_factory=ROUTERS[kind])
    routed = scenario.population.by_id("p").selected_plan
    assert [net.link_ids.id(k) for k in routed.legs[0].route.links] == ["ab", "bh"]
    assert PlanBatch.from_plans([routed]).invalid_legs(net).size == 0

    # reaching the node by another link is not a valid route to the activity
    other = routed.with_elements([plan.elements[0], Leg("car", Route([net.link_index("ah")], 10, 200)),
                                  plan.elements[2]])
    assert PlanBatch.from_plans([other]).invalid_legs(net).tolist() == [0]


def test_an_activity_link_must_end_at_its_node():
    net = side_entry()
    a, h = net.node_index("a"), net.node_index("h")
    plan = (Plan.builder().act("w", a, end_time=0.0, link=net.link_index("bh")).leg(
        "car", Route([net.link_index("ah")], 10, 200)).act("h", h).build())
    assert PlanBatch.from_plans([plan]).invalid_legs(net).tolist() == [0]
