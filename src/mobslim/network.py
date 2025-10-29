import xml.etree.ElementTree as ET

from networkx import DiGraph, Graph


class Network:

    def __init__(self):
        self.G = DiGraph()
        self.node_positions = {}

    def nodes(self):
        """Get the nodes of the grid graph.

        Returns:
            list: A list of nodes in the grid graph.
        """
        return self.G.nodes

    def edges(self):
        """Get the edges of the grid graph.

        Returns:
            list: A list of edges in the grid graph.
        """
        return self.G.edges

    def load_xml(self, path: str):
        """Load a network from an XML file.

        Args:
            path (str): The path to the XML file.
        """

        tree = ET.parse(path)
        root = tree.getroot()

        # Load nodes
        for node in root.find("nodes"):
            node_id = int(node.get("id"))
            x = float(node.get("x"))
            y = float(node.get("y"))
            self.G.add_node(node_id)
            self.node_positions[node_id] = (x, y)

        # Load links
        for link in root.find("links"):
            link_id = int(link.get("id"))
            from_node = int(link.get("from"))
            to_node = int(link.get("to"))
            length = float(link.get("length"))
            capacity = float(link.get("capacity"))
            freespeed = float(link.get("freespeed"))
            permlanes = int(link.get("permlanes"))
            self.G.add_edge(
                from_node,
                to_node,
                id=link_id,
                length=length,
                flow_capacity=capacity / 3600,
                freespeed=freespeed,
                lanes=permlanes,
            )

    def minimum_durations(self) -> dict:
        """Get the minimum durations for all edges in the network.

        Returns:
            dict: A dictionary with edges as keys and minimum durations as values.
        """
        min_durations = {}
        for u, v, data in self.G.edges(data=True):
            length = data.get("length", 0)
            freespeed = data.get("freespeed", 1)
            duration = length / freespeed
            min_durations[(u, v)] = duration
        return min_durations


class Grid(Network):
    """A grid graph with nodes arranged in a 10x10 grid.
    Node ids are tuples (i, j) where i and j are the row and column indices.
    """

    def __init__(
        self,
        **kwargs: dict,
    ):
        self.G, self.node_positions = self.build_grid_graph(**kwargs if kwargs else {})

    def build_grid_graph(
        self,
        size: int = 10,
        length: float = 50,  # meters
        lanes: int = 1,
        freespeed: float = 10,  # m/s
        flow_capacity: float = 0.25,  # vehicles per second
    ) -> Graph:
        """Create a grid graph with 100 nodes. length between nodes is 50.
        Returns:
            Graph: A grid graph with 100 nodes.
        """
        self.size = size
        G = Graph()
        node_positions = {}
        for i in range(size + 1):
            for j in range(size + 1):
                G.add_node((i, j))
                node_positions[(i, j)] = (j * length, i * length)
                if i > 0:
                    G.add_edge(
                        (i, j),
                        (i - 1, j),
                        length=length,
                        lanes=lanes,
                        freespeed=freespeed,
                        flow_capacity=flow_capacity,
                    )
                if j > 0:
                    G.add_edge(
                        (i, j),
                        (i, j - 1),
                        length=length,
                        lanes=lanes,
                        freespeed=freespeed,
                        flow_capacity=flow_capacity,
                    )
        return G, node_positions

    def get_start(self):
        """Get the top-left node of the grid graph.

        Returns:
            tuple: The coordinates of the top-left node.
        """
        return (0, 0)

    def get_end(self):
        """Get the bottom-right node of the grid graph.

        Returns:
            tuple: The coordinates of the bottom-right node.
        """
        return (self.size, self.size)

    def __repr__(self):
        
        o_row = "O" + "---X" * self.size + "\n"
        row = "X---" * self.size + "X\n"
        d_row = "X---" * self.size + "D\n"
        empty_row = "|   " * self.size + "|\n"

        string = d_row + empty_row
        for _ in range(self.size - 1):
            string += row
            string += empty_row
        string += o_row
        return string


class Linear(Network):
    def __init__(
        self,
        **kwargs: dict,
    ):
        self.G, self.node_positions= self.build_linear_graph(**kwargs if kwargs else {})

    def build_linear_graph(
        self,
        size: int = 10,
        length: float = 50,  # meters
        lanes: int = 1,
        freespeed: float = 10,  # m/s
        flow_capacity: float = 0.25,  # vehicles per second
    ) -> Graph:
        """Create a linear graph with a specified number of nodes.
        Returns:
            Graph: A linear graph with the specified number of nodes.
        """
        self.size = size
        G = Graph()
        node_positions = {}
        for i in range(size + 1):
            G.add_node(i)
            node_positions[i] = (i * length, 0)
            if i > 0:
                G.add_edge(
                    i,
                    i - 1,
                    length=length,
                    lanes=lanes,
                    freespeed=freespeed,
                    flow_capacity=flow_capacity,
                )
        return G, node_positions
    
    def get_start(self):
        """Get the start node of the linear graph.

        Returns:
            int: The start node.
        """
        return 0

    def get_end(self):
        """Get the end node of the linear graph.

        Returns:
            int: The end node.
        """
        return self.size

    def __repr__(self):
        string = "X"
        for r in range(self.size):
            string += "---X"
        return string
