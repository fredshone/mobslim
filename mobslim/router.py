from networkx import shortest_path

from mobslim.network import Network
from expected import Durations


class Router:
    def __init__(self, network: Network, link_durations: Durations):
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


class StaticRouter:
    def __init__(
        self,
        network: Network,
        link_durations: Durations,
    ):
        """Static because it does not change over time.
        Args:
            network (Network): The network to route through.
            link_durations (Durations): The expected durations for the links in the network.
        """
        self.G = network.G.copy()
        self.link_durations = link_durations
        for edge in self.G.edges:
            self.G[edge[0]][edge[1]]["expected_duration"] = link_durations.get(edge)

    def route(self, source, target, time):
        """Find the shortest path between source and target nodes.
        Args:
            source: The starting node.
            target: The destination node.
        Returns:
            tuple: A list of edges representing the shortest path and exit times for each edge.
        """
        path = shortest_path(
            self.G, source=source, target=target, weight="expected_duration"
        )
        link_ids = [
            self.G[edge[0]][edge[1]].get("id") for edge in zip(path[:-1], path[1:])
        ]

        exits = []
        t = 0
        for u, v in link_ids:
            duration = self.link_durations.get((u, v), time + t)
            t += duration
            exits.append(t)

        return link_ids, exits
