from networkx import Graph


class Durations:
    """A class to handle expected durations for edges in a graph."""

    def __init__(self, graph: Graph):
        raise NotImplementedError("This class is not implemented yet.")

    def get(edge: tuple, time: float) -> float:
        """Get the expected duration for a given edge at a specific time.

        Args:
            edge (tuple): The edge for which to get the expected duration.
            time (float): The time at which to get the expected duration.

        Returns:
            float: The expected duration for the edge at the specified time.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def update(edge: tuple, time: float, duration: float):
        """Update the expected duration for a given edge at a specific time.

        Args:
            edge (tuple): The edge for which to update the expected duration.
            time (float): The time at which to update the expected duration.
            duration (float): The new expected duration for the edge.
        """
        raise NotImplementedError("This method is not implemented yet.")


class SimpleDurations(Durations):
    """A simple implementation of expected durations for edges in a graph."""

    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.expected_durations = {edge: 1 for edge in graph.edges}

    def get(self, edge: tuple, time: int) -> float:
        """Get the expected duration for a given edge at a specific time."""
        return self.expected_durations[edge]

    def update(self, edge: tuple, time: int, duration: float):
        """Update the expected duration for a given edge at a specific time."""
        self.expected_durations[edge] = duration
