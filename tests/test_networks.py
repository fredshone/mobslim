from mobslim import Networks
from mobslim.entities.networks import Linear, RoadGrid


class TestNetworks:
    """Test Networks class."""

    def test_networks_init(self):
        """Test Networks initialization."""
        networks = Networks()
        assert networks._networks == {}
        assert networks.node_locations == {}

    def test_load_xml(self, tmp_path):
        """Test loading network from XML."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">
<network>
    <nodes>
        <node id="1" x="0" y="0"/>
        <node id="2" x="100" y="0"/>
    </nodes>
    <links>
        <link id="1-2" from="1" to="2" length="100" capacity="1800" freespeed="10" permlanes="1" modes="road"/>
    </links>
</network>"""
        xml_file = tmp_path / "network.xml"
        xml_file.write_text(xml_content)

        networks = Networks()
        networks.load_xml(str(xml_file))

        assert "1" in networks.node_locations
        assert "2" in networks.node_locations
        assert networks.node_locations["1"] == (0, 0)
        assert networks.node_locations["2"] == (100, 0)
        assert "road" in networks._networks
        assert networks["road"].has_edge("1", "2")

    def test_getitem_setitem(self):
        """Test getting and setting network modes."""
        networks = Networks()
        from networkx import DiGraph

        graph = DiGraph()
        graph.add_edge("A", "B")
        networks["test"] = graph

        assert "test" in networks._networks
        assert networks["test"] == graph
        assert networks["test"].has_edge("A", "B")

    def test_items(self):
        """Test iterating over network modes."""
        networks = Networks()
        from networkx import DiGraph

        graph1 = DiGraph()
        graph2 = DiGraph()
        networks["mode1"] = graph1
        networks["mode2"] = graph2

        items = list(networks.items())
        assert len(items) == 2
        assert ("mode1", graph1) in items
        assert ("mode2", graph2) in items

    def test_node_quad_tree(self):
        """Test creating quad tree from node locations."""
        networks = Networks()
        networks.node_locations = {
            "1": (0, 0),
            "2": (100, 0),
            "3": (100, 100),
            "4": (0, 100),
        }
        qt = networks.node_quad_tree()

        # Find nearest neighbor to (50, 50)
        nearest = qt.nearest_neighbor((50, 50))
        assert nearest.obj in ["2", "3", "4", "1"]


class TestRoadGrid:
    """Test RoadGrid class."""

    def test_road_grid_init(self):
        """Test RoadGrid initialization."""
        grid = RoadGrid(size=3)
        assert grid.size == 3
        assert "road" in grid._networks
        assert grid.get_start() == (0, 0)
        assert grid.get_end() == (3, 3)

    def test_road_grid_edges(self):
        """Test RoadGrid edges."""
        grid = RoadGrid(size=2, length=50, freespeed=10)

        # Check that grid has bidirectional edges
        assert grid["road"].has_edge((0, 0), (0, 1))
        assert grid["road"].has_edge((0, 1), (0, 0))
        assert grid["road"].has_edge((0, 0), (1, 0))
        assert grid["road"].has_edge((1, 0), (0, 0))

    def test_road_grid_node_positions(self):
        """Test RoadGrid node positions."""
        grid = RoadGrid(size=2, length=50)

        assert grid.node_locations[(0, 0)] == (0, 0)
        assert grid.node_locations[(0, 1)] == (50, 0)
        assert grid.node_locations[(1, 0)] == (0, 50)
        assert grid.node_locations[(2, 2)] == (100, 100)

    def test_road_grid_edge_attributes(self):
        """Test RoadGrid edge attributes."""
        grid = RoadGrid(
            size=2, length=100, lanes=2, freespeed=20, flow_capacity=0.5
        )

        edge_data = grid["road"].get_edge_data((0, 0), (0, 1))
        assert edge_data["length"] == 100
        assert edge_data["lanes"] == 2
        assert edge_data["freespeed"] == 20
        assert edge_data["flow_capacity"] == 0.5

    def test_road_grid_repr(self):
        """Test RoadGrid string representation."""
        grid = RoadGrid(size=2)
        repr_str = repr(grid)

        assert "O" in repr_str
        assert "D" in repr_str
        assert "X" in repr_str


class TestLinear:
    """Test Linear network class."""

    def test_linear_init(self):
        """Test Linear network initialization."""
        linear = Linear(size=5)
        assert linear.size == 5
        assert linear.get_start() == 0
        assert linear.get_end() == 5

    def test_linear_edges(self):
        """Test Linear network edges."""
        linear = Linear(size=3)

        # Check edges exist
        assert linear._networks["road"].has_edge(0, 1)
        assert linear._networks["road"].has_edge(1, 2)
        assert linear._networks["road"].has_edge(2, 3)

    def test_linear_node_positions(self):
        """Test Linear network node positions."""
        linear = Linear(size=3, length=100)

        assert linear.node_positions[0] == (0, 0)
        assert linear.node_positions[1] == (100, 0)
        assert linear.node_positions[2] == (200, 0)
        assert linear.node_positions[3] == (300, 0)

    def test_linear_repr(self):
        """Test Linear network string representation."""
        linear = Linear(size=3)
        repr_str = repr(linear)

        assert repr_str == "X---X---X---X"
