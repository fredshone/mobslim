from networkx import shortest_path

from mobslim.network import Network
from mobslim.expected import ExpectedLinkDurations
from mobslim.planners.rerouters.core import BaseRouter

    

class StaticRouter(BaseRouter):
    def __init__(
        self,
        network: Network,
        expectations: ExpectedLinkDurations,
    ):
        """Static because it does not change over time.
        Args:
            network (Network): The network to route through.
            link_durations (Durations): The expected durations for the links in the network.
        """
        self.G = network.G.copy()
        # calc minimum durations based on length and freespeed
        for edge in self.G.edges:
            length = self.G[edge[0]][edge[1]]["length"]
            freespeed = self.G[edge[0]][edge[1]]["freespeed"]
            minduration = length / freespeed if freespeed > 0 else 0
            self.G[edge[0]][edge[1]]["minimum_duration"] = minduration
        self.expectations = expectations
        for edge in self.G.edges:
            self.G[edge[0]][edge[1]]["expected_duration"] = expectations.get(edge, None)

    def get_route(self, source, target, time):
        """Find the shortest path between source and target nodes.
        Args:
            source: The starting node.
            target: The destination node.
        Returns:
            tuple: A list of edges representing the shortest path and expected duration.
        """

        path = shortest_path(
            self.G, source=source, target=target, weight="expected_duration"
        )
        link_ids = [(u, v) for u, v in zip(path[:-1], path[1:])]
        expected_durations = [self.G[u][v]["expected_duration"] for u, v in link_ids]
        minimum_durations = [self.G[u][v]["minimum_duration"] for u, v in link_ids]

        return list(zip(link_ids, expected_durations, minimum_durations)), sum(
            expected_durations
        )

    def update(self, plans: dict, network: Network, events: list, alpha: float = 1.0):
        self.expectations.update(plans, network, events, alpha=alpha)
        for edge in self.G.edges:
            self.G[edge[0]][edge[1]]["expected_duration"] = self.expectations.get(
                edge, None
            )
