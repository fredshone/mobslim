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
        kwargs: dict = None,
    ):
        self.G = self.build_grid_graph(**kwargs if kwargs else {})
        self.expected_durations = {edge: 1 for edge in self.G.edges}

    def build_grid_graph(
        self,
        size: int = 10,
        distance: float = 50,
        lanes: int = 1,
        freespeed: float = 10,
    ) -> Graph:
        """Create a grid graph with 100 nodes. Distance between nodes is 1.
        Returns:
            Graph: A grid graph with 100 nodes.
        """
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
                    )
                if j > 0:
                    G.add_edge(
                        (i, j),
                        (i, j - 1),
                        distance=distance,
                        lanes=lanes,
                        freespeed=freespeed,
                    )
        return G
