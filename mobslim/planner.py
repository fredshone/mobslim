from random import random
from mobslim.agents import Plan, Trip, Activity

from networkx import shortest_path

from mobslim.network import Network
from mobslim.expected import ExpectedLinkDurations
from typing import Dict, Hashable


class Router:
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


class Planner:
    """Currently just a trip planner.
    Assumes start of day at 0 and end at 86400 (24 hours)
    """

    def __init__(self):
        pass

    def plan(self, plans: Dict[Hashable, Plan], router: Router, p: float = 0.5):
        if p < 0 or p > 1:
            raise ValueError("Probability p must be between 0 and 1.")
        for _, plan in plans.items():
            if random() > p:
                continue  # Skip planning for this agent
            time = 0
            for _, component in enumerate(plan.components):

                if isinstance(component, Trip):
                    component.route, component.duration = router.get_route(
                        component.origin, component.destination, time
                    )
                    time += component.duration

                if isinstance(component, Activity):
                    if component.duration is None:  # hopefully just end of day
                        component.duration = 86400 - time
                        if component.duration < 0:
                            raise ValueError("Activity duration cannot be negative.")
                    time += component.duration


class StaticRouter(Router):
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

        return link_ids, sum(expected_durations)

    def update(self, plans: dict, network: Network, events: list, alpha: float = 1.0):
        self.expectations.update(plans, network, events, alpha=alpha)
        for edge in self.G.edges:
            self.G[edge[0]][edge[1]]["expected_duration"] = self.expectations.get(
                edge, None
            )
