from mobslim.expected import ExpectedLinkDurations
from mobslim.network import Network


class BaseRouter:
    def __init__(self, network: Network, link_durations: ExpectedLinkDurations):
        """Router for the network.
        Args:
            network (Network): The network to route through.
            link_durations (Durations): The expected durations for the links in the network.
        """
        raise NotImplementedError("This class is not implemented yet.")

    def route(self, source, target, time):
        """Find the shortest path between source and target nodes.
        Args:
            source: The starting node.
            target: The destination node.
            time: The time at which to find the shortest path.
        Returns:
            tuple: A list of edges representing the shortest path and exit times for each edge.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def update(self, events: list):
        """Update the router's expected durations based on simulation events.
        Args:
            events (list): A list of events from the simulation.
        """
        raise NotImplementedError("This method is not implemented yet.")