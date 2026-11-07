"""Link attributes and activity-to-link assignment (nearest link midpoint)."""

import pickle

import numpy as np
import pytest

from mobslim.activity_links import (
    ActivityLinkParams,
    LinkFilter,
    LinkSnapper,
    SnapTier,
    assign_activity_links,
    link_midpoints,
    link_snapper,
    rust_snapper_available,
)
from mobslim.formats.matsim_network import read_network_xml
from mobslim.network import NetworkBuilder, grid_network
from mobslim.population import Activity, Leg, Plan, Population
from mobslim.replanning.prepare import route_scenario
from mobslim.scenario import Scenario

# ------------------------------------------------------------ link attributes

NETWORK_XML = """<?xml version="1.0"?>
<network>
  <nodes>
    <node id="a" x="0" y="0"/><node id="b" x="1000" y="0"/><node id="c" x="1000" y="1000"/>
  </nodes>
  <links capperiod="01:00:00">
    <link id="ab" from="a" to="b" length="1000" freespeed="30" capacity="2000" permlanes="2"
          type="motorway"/>
    <link id="ba" from="b" to="a" length="1000" freespeed="30" capacity="2000" permlanes="2">
      <attributes>
        <attribute name="osm:way:highway" class="java.lang.String">motorway</attribute>
        <attribute name="lit" class="java.lang.Boolean">true</attribute>
      </attributes>
    </link>
    <link id="bc" from="b" to="c" length="1000" freespeed="10" capacity="600" permlanes="1"
          type="residential"/>
  </links>
</network>
"""


@pytest.fixture
def xml_network(tmp_path):
    path = tmp_path / "network.xml"
    path.write_text(NETWORK_XML)
    return read_network_xml(str(path))


def test_matsim_link_attributes_are_lazy(xml_network):
    net = xml_network
    assert "link_attrs" not in net._cache  # nothing parsed yet
    assert net.link_attribute("type").tolist() == ["motorway", None, "residential"]
    assert net.link_attribute("osm:way:highway").tolist() == [None, "motorway", None]
    assert net.link_attribute("lit").tolist() == [None, True, None]
    assert net.link_attribute("freespeed").tolist() == [30, 30, 10]  # built-in columns
    assert set(net.link_attribute_names()) == {"type", "osm:way:highway", "lit"}
    with pytest.raises(KeyError, match="no link attribute 'nope'"):
        net.link_attribute("nope")


def test_lazy_link_attributes_survive_pickling(xml_network):
    back = pickle.loads(pickle.dumps(xml_network))
    assert back.link_attribute("type").tolist() == ["motorway", None, "residential"]


def test_builder_link_attributes():
    b = NetworkBuilder()
    for n, x in (("a", 0), ("b", 100)):
        b.add_node(n, x, 0)
    b.add_link("ab", "a", "b", 100, 10, 600, attributes={"type": "primary", "lit": True})
    b.add_link("ba", "b", "a", 100, 10, 600)
    net = b.build()
    assert net.link_attribute("type").tolist() == ["primary", None]
    assert grid_network(2).link_attribute_names() == ()


def test_link_filter(xml_network):
    assert LinkFilter("type", values=("motorway",)).mask(xml_network).tolist() == [True, False, False]
    assert LinkFilter("freespeed", min=20).mask(xml_network).tolist() == [True, True, False]
    assert LinkFilter("missing", values=("x",)).mask(xml_network).tolist() == [False] * 3
    with pytest.raises(ValueError):
        LinkFilter("type")


# ------------------------------------------------------------ snapping

def _highway_network():
    """A motorway along y=0 and a residential street along y=1500, both two-way,
    plus a car-free path along y=-200."""
    b = NetworkBuilder()
    for n, x, y in (("m0", 0, 0), ("m1", 5000, 0), ("r0", 0, 1500), ("r1", 5000, 1500),
                    ("p0", 0, -200), ("p1", 5000, -200)):
        b.add_node(n, x, y)
    for u, v, t, modes in (("m0", "m1", "motorway", ("car",)), ("r0", "r1", "residential", ("car",)),
                           ("p0", "p1", "path", ("walk",))):
        b.add_link(f"{u}{v}", u, v, 5000, 10, 600, modes=modes, attributes={"type": t})
        b.add_link(f"{v}{u}", v, u, 5000, 10, 600, modes=modes, attributes={"type": t})
    return b.build()


def _types(net, links):
    return [net.link_attribute("type")[k] for k in links]


def test_link_midpoints():
    net = _highway_network()
    np.testing.assert_allclose(link_midpoints(net)[:2], [[2500, 0], [2500, 0]])


def test_snap_to_car_links_by_default():
    net = _highway_network()
    # (2500, -150) is nearest the path's midpoint, but the path does not allow car
    snap = link_snapper(net).snap
    assert _types(net, snap([[2500, -150], [2500, 1400]])) == ["motorway", "residential"]
    assert _types(net, link_snapper(net, ActivityLinkParams(modes=None)).snap([[2500, -150]])) \
        == ["path"]


def test_avoid_motorway_unless_nothing_within():
    net = _highway_network()
    params = ActivityLinkParams.avoiding(LinkFilter("type", values=("motorway",)),
                                         unless_beyond=1000.0)
    links = link_snapper(net, params).snap([[2500, 600], [2500, 400], [2500, -20]])
    # 900 m from the street: avoid the motorway; 1100 m: fall back to it
    assert _types(net, links) == ["residential", "motorway", "motorway"]
    assert link_snapper(net, params) is link_snapper(net, params)  # cached per network


def test_tiers_in_order():
    net = _highway_network()
    params = ActivityLinkParams(tiers=(
        SnapTier((LinkFilter("type", values=("motorway", "residential")),), 10.0),  # nothing left
        SnapTier((LinkFilter("type", values=("motorway",)),), 2000.0),
    ))
    assert _types(net, link_snapper(net, params).snap([[2500, 10]])) == ["residential"]


needs_rust = pytest.mark.skipif(not rust_snapper_available(),
                                reason="needs the mobslim_router extension")


def _brute_force(net, xy, params=ActivityLinkParams()):
    """Nearest midpoint per tier by exhaustive search (ties: lowest index)."""
    mid, eligible = link_midpoints(net), params.eligible(net)
    tiers = []
    for tier in params.tiers:
        mask = eligible.copy()
        for f in tier.exclude:
            mask &= ~f.mask(net)
        tiers.append((mask, tier.max_distance))
    tiers.append((eligible, np.inf))
    out = []
    for x, y in xy:
        d = np.hypot(mid[:, 0] - x, mid[:, 1] - y)
        for mask, max_distance in tiers:
            if mask.any() and d[mask].min() <= max_distance:
                out.append(np.flatnonzero(mask & (d == d[mask].min()))[0])
                break
    return np.array(out)


@pytest.mark.parametrize("backend", ["quadtree", pytest.param("rust", marks=needs_rust)])
def test_backends_match_brute_force(backend):
    # a grid: opposite links share a midpoint, and lattice queries tie between links
    net = grid_network(12, spacing=100.0)
    rng = np.random.default_rng(3)
    xy = np.concatenate([rng.uniform(-200, 1400, (400, 2)),
                         rng.integers(-2, 25, (200, 2)) * 50.0])  # on midpoints and nodes
    got, want = LinkSnapper(net, backend=backend, threads=3).snap(xy), _brute_force(net, xy)
    mid = link_midpoints(net)
    np.testing.assert_array_equal(np.hypot(*(mid[got] - xy).T), np.hypot(*(mid[want] - xy).T))
    if backend == "rust":  # the quadtree breaks ties between distinct points arbitrarily
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("backend", ["quadtree", pytest.param("rust", marks=needs_rust)])
def test_backends_match_brute_force_with_tiers(backend):
    net = _highway_network()
    params = ActivityLinkParams(tiers=(
        SnapTier((LinkFilter("type", values=("motorway",)),), 900.0),
        SnapTier((LinkFilter("type", values=("residential",)),), 1200.0),
    ))
    xy = np.random.default_rng(4).uniform(-1000, 6000, (300, 2))
    snapper = LinkSnapper(net, params, backend=backend)
    np.testing.assert_array_equal(snapper.snap(xy), _brute_force(net, xy, params))


def test_snap_rejects_non_finite_points():
    with pytest.raises(ValueError, match="finite"):
        link_snapper(grid_network(2)).snap([[np.nan, 0.0]])


def test_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        LinkSnapper(grid_network(2), backend="kdtree")


# ------------------------------------------------------------ assignment

def _plan(net, xy_home, xy_work, home_node=0, link=None):
    return Plan((
        Activity("h", home_node, end_time=8 * 3600, x=xy_home[0], y=xy_home[1], link=link),
        Leg("car"),
        Activity("w", home_node, end_time=17 * 3600, x=xy_work[0], y=xy_work[1]),
        Leg("walk"),
        Activity("h", home_node, x=xy_home[0], y=xy_home[1], link=link),
    ))


def test_assign_activity_links():
    net = grid_network(4, spacing=100.0)  # nodes (row, col) at (100 col, 100 row)
    keep = net.link_index(((0, 0), (0, 1)))
    pop = Population.from_plans({
        "a": _plan(net, (5, 52), (395, 248)),
        "b": _plan(net, (5, 52), (395, 248), home_node=net.node_index((0, 1)), link=keep),
        "c": Plan((Activity("h", net.node_index((2, 2)), end_time=3600.0), Leg("car"),
                   Activity("w", net.node_index((4, 4))))),
    })
    out = assign_activity_links(Scenario(net, pop)).population
    a, b, c = (p.selected_plan.activities for p in out)
    # (5, 52) is nearest the midpoint (0, 50) of the links between (0,0) and (1,0)
    assert set(net.link_ids.id(a[0].link)) == {(0, 0), (1, 0)}
    assert a[0].node == net.to_node[a[0].link]
    assert (a[0].x, a[0].y) == (5, 52)  # coordinates kept
    assert set(net.link_ids.id(a[1].link)) == {(2, 4), (3, 4)}
    assert a[2].link == a[0].link
    assert b[0].link == keep and b[1].link == a[1].link  # an existing link is kept
    assert all(x.link is None for x in c)  # placed at nodes: kept there
    forced = assign_activity_links(Scenario(net, pop), ActivityLinkParams(snap_node_activities=True))
    c = forced.population[2].selected_plan.activities
    assert all(x.link is not None and x.node == net.to_node[x.link] for x in c)


def test_assignment_drops_stale_routes_and_routing_ends_on_the_link():
    net = grid_network(4, spacing=100.0)
    pop = Population.from_plans({"a": _plan(net, (5, 52), (395, 248))})
    routed = route_scenario(Scenario(net, pop), activity_links=None)  # node to node
    assert routed.population[0].selected_plan.legs[0].route is not None
    redone = route_scenario(routed)
    plan = redone.population[0].selected_plan
    home, work = plan.activities[:2]
    route = plan.legs[0].route.links
    assert route[-1] == work.link  # ends by driving through the activity's link
    assert net.from_node[route[0]] == home.node
    assert plan.legs[1].route is None  # walk: teleported


def test_assignment_is_a_no_op_when_every_activity_has_a_link():
    net = grid_network(2)
    pop = Population.from_plans({"a": Plan((Activity("h", 0, end_time=1.0), Leg("car"),
                                            Activity("w", 3)))})
    scenario = Scenario(net, pop)
    assert assign_activity_links(scenario) is scenario


def test_config_activity_links(tmp_path):
    from mobslim.config.mobslim import MobslimConfig

    (tmp_path / "n.xml").write_text(NETWORK_XML)
    (tmp_path / "p.xml").write_text("<population/>")
    path = tmp_path / "c.yaml"
    path.write_text("inputs: {network: n.xml, plans: p.xml}\n"
                    "activity_links:\n"
                    "  avoid: {attribute: osm:way:highway, values: [motorway, trunk], "
                    "unless_beyond: 500}\n")
    params = MobslimConfig.load(path).controller_params().activity_links
    assert params == ActivityLinkParams(
        tiers=(SnapTier((LinkFilter("osm:way:highway", ("motorway", "trunk")),), 500.0),))
    path.write_text("inputs: {network: n.xml, plans: p.xml}\n")
    assert MobslimConfig.load(path).controller_params().activity_links == ActivityLinkParams()
    path.write_text("inputs: {network: n.xml, plans: p.xml}\n"
                    "activity_links: {avoid: {attribute: type}}\n")
    with pytest.raises(ValueError, match="needs values or min_value"):
        MobslimConfig.load(path).controller_params()
