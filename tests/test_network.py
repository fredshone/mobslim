import pickle

import numpy as np
import pytest

from mobslim import (
    NetworkBuilder,
    grid_network,
    linear_network,
    read_network_xml,
)
from tests.conftest import FIXTURES


def test_read_equil(equil_paths):
    net = read_network_xml(equil_paths[0])
    assert net.n_nodes == 15
    assert net.n_links == 23
    assert net.modes == {"car"}  # MATSim default when no modes attribute
    link1 = net.link_index("1")
    assert net.capacity[link1] == 36000  # capperiod 01:00:00
    assert net.node_ids.id(int(net.from_node[link1])) == "1"


def test_capperiod_is_normalised_to_hourly():
    net = read_network_xml(FIXTURES.parent.parent / "scenarios" / "pt-simple" / "network.xml")
    # capacity="2000" with capperiod="10:00:00" -> 200 veh/h
    assert np.allclose(net.capacity, 200.0)
    assert net.modes == {"pt"}


def test_parallel_links_are_kept():
    b = NetworkBuilder()
    b.add_node("a", 0, 0)
    b.add_node("b", 100, 0)
    b.add_link("fast", "a", "b", 100, 20, 1000)
    b.add_link("slow", "a", "b", 100, 5, 1000)
    net = b.build()
    assert net.n_links == 2
    offsets, links = net.out_links("car")
    assert sorted(links[offsets[0]:offsets[1]]) == [0, 1]


def test_builder_validates():
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 1, 0)
    with pytest.raises(ValueError):
        b.add_link("x", 0, 1, 10, 10, 0)  # zero capacity
    with pytest.raises(KeyError):
        b.add_link("x", 0, 99, 10, 10, 100)
    b.add_link("x", 0, 1, 10, 10, 100)
    with pytest.raises(ValueError):
        b.add_link("x", 0, 1, 10, 10, 100)


def test_mode_specific_adjacency():
    b = NetworkBuilder()
    for i in range(3):
        b.add_node(i, i, 0)
    b.add_link("car", 0, 1, 1, 1, 1, modes=["car"])
    b.add_link("both", 1, 2, 1, 1, 1, modes=["car", "bike"])
    net = b.build()
    assert net.modes == {"car", "bike"}
    assert net.mode_mask("bike").tolist() == [False, True]
    offsets, links = net.out_links("bike")
    assert links == (1,) and offsets[1] == offsets[0]


def test_grid_and_linear():
    g = grid_network(size=2, spacing=10)
    assert g.n_nodes == 9
    assert g.n_links == 2 * 2 * 3 * 2  # 12 undirected edges, both directions
    assert g.node_index((2, 2)) == 8
    lin = linear_network(size=3)
    assert lin.n_links == 6


def test_nearest_node_and_beeline(equil_scenario):
    net = equil_scenario.network
    assert net.node_ids.id(net.nearest_node(-20000, 0)) == "1"
    a, b = net.node_index("1"), net.node_index("2")
    assert net.beeline_distance(a, b) == pytest.approx(5000)


def test_pickle_roundtrip_drops_caches(equil_scenario):
    net = equil_scenario.network
    net.nearest_node(0, 0)  # populate quadtree cache
    clone = pickle.loads(pickle.dumps(net))
    assert clone.n_links == net.n_links
    assert np.array_equal(clone.length, net.length)
    assert clone.node_ids.index("7") == net.node_ids.index("7")


def test_to_networkx(equil_scenario):
    g = equil_scenario.network.to_networkx()
    assert g.number_of_edges() == 23
    assert g.has_edge("1", "2", key="1")


def test_nearest_node_matches_brute_force_at_utm_scale():
    import numpy as np

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 5000, (2000, 2)) + [600_000, 5_700_000]  # float32 loses metres here
    b = NetworkBuilder()
    for i, (x, y) in enumerate(xy):
        b.add_node(i, x, y)
    b.add_link("l", 0, 1, 1, 1, 1)
    net = b.build()
    for x, y in rng.uniform(0, 5000, (300, 2)) + [600_000, 5_700_000]:
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        assert d[net.nearest_node(x, y)] == pytest.approx(d.min(), abs=1e-6)


def test_out_links_are_immutable():
    offsets, links = grid_network(size=2).out_links("car")
    assert isinstance(offsets, tuple) and isinstance(links, tuple)
