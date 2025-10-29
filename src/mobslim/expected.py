from networkx import Graph

from mobslim.network import Network
from mobslim.processs_events import expected_link_durations


class ExpectedLinkDurations:
    """A class to handle expected durations for edges in a graph."""

    def __init__(self, graph: Graph):
        raise NotImplementedError("This class is not implemented yet.")

    def get(self, edge: tuple, time: float) -> float:
        """Get the expected duration for a given edge at a specific time.

        Args:
            edge (tuple): The edge for which to get the expected duration.
            time (float): The time at which to get the expected duration.

        Returns:
            float: The expected duration for the edge at the specified time.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def update(self, plans: dict, network: Network, events: list):
        raise NotImplementedError("This method is not implemented yet.")

    def update_link(edge: tuple, time: float, duration: float):
        """Update the expected duration for a given edge at a specific time.

        Args:
            edge (tuple): The edge for which to update the expected duration.
            time (float): The time at which to update the expected duration.
            duration (float): The new expected duration for the edge.
        """
        raise NotImplementedError("This method is not implemented yet.")


class SimpleExpectedDurations(ExpectedLinkDurations):
    """A simple implementation of expected durations for edges in a graph."""

    def __init__(self, network: Network):
        self.edge_durations = network.minimum_durations()
        if None in self.edge_durations.values():
            raise ValueError("All edges must have a duration attribute.")

    def get(self, edge: tuple, time: int) -> float:
        """Get the expected duration for a given edge at a specific time."""
        return self.edge_durations[edge]

    def update(self, plans: dict, network: Network, events: list, alpha: float = 1.0):
        durations = expected_link_durations(plans, network, events)
        for edge, duration in durations.items():
            if duration is not None:
                self.update_link(edge, None, duration, alpha=alpha)

    def update_link(self, edge: tuple, time: int, duration: float, alpha: float = 0.5):
        """Update the expected duration for a given edge at a specific time."""
        self.edge_durations[edge] = (1 - alpha) * self.edge_durations[
            edge
        ] + alpha * duration

    def av_duration(self) -> float:
        """Calculate the average expected duration across all edges."""
        return sum(self.edge_durations.values()) / len(self.edge_durations)
