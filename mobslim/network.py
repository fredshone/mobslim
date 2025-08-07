from networkx import Graph


class Network:

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


class Grid(Network):
    """A grid graph with nodes arranged in a 10x10 grid.
    Node ids are tuples (i, j) where i and j are the row and column indices.
    """

    def __init__(
        self,
        **kwargs: dict,
    ):
        self.G = self.build_grid_graph(**kwargs if kwargs else {})
        self.expected_durations = {edge: 1 for edge in self.G.edges}

    def build_grid_graph(
        self,
        size: int = 10,
        distance: float = 50,  # meters
        lanes: int = 1,
        freespeed: float = 10,  # m/s
        flow_capacity: float = 0.25,  # vehicles per second
    ) -> Graph:
        """Create a grid graph with 100 nodes. Distance between nodes is 50.
        Returns:
            Graph: A grid graph with 100 nodes.
        """
        self.size = size
        G = Graph()
        for i in range(size):
            for j in range(size):
                G.add_node((i, j))
                if i > 0:
                    G.add_edge(
                        (i, j),
                        (i - 1, j),
                        distance=distance,
                        lanes=lanes,
                        freespeed=freespeed,
                        flow_capacity=flow_capacity,
                    )
                if j > 0:
                    G.add_edge(
                        (i, j),
                        (i, j - 1),
                        distance=distance,
                        lanes=lanes,
                        freespeed=freespeed,
                        flow_capacity=flow_capacity,
                    )
        return G

    def get_top_left(self):
        """Get the top-left node of the grid graph.

        Returns:
            tuple: The coordinates of the top-left node.
        """
        return (0, 0)

    def get_bottom_right(self):
        """Get the bottom-right node of the grid graph.

        Returns:
            tuple: The coordinates of the bottom-right node.
        """
        return (self.size - 1, self.size - 1)
