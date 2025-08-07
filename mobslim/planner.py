from mobslim.agents import Plan, Trip, Instruction

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

    def update_from_events(self, events: list):
        """Update the router's expected durations based on simulation events.
        Args:
            events (list): A list of events from the simulation.
        """
        raise NotImplementedError("This method is not implemented yet.")


class Planner:
    """Currently just a trip planner."""

    def __init__(self):
        pass

    def plan(self, plans: Dict[Hashable, Plan], router: Router):
        for _, plan in plans.items():
            for _, component in enumerate(plan.components):
                if isinstance(component, Trip):
                    component.route = router.get_route(
                        component.origin, component.destination, component.start_time
                    )
                else:
                    raise ValueError(f"Unknown component type: {type(component)}")


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
            tuple: A list of edges representing the shortest path and exit times for each edge.
        """

        path = shortest_path(
            self.G, source=source, target=target, weight="expected_duration"
        )
        link_ids = [(u, v) for u, v in zip(path[:-1], path[1:])]
        return link_ids

    def update_from_events(self, events: list):
        enters = {}
        for event, _, time, uv in events:
            if event == Instruction.EnterLink:
                enters[uv] = time
            elif event == Instruction.ExitLink:
                if uv in enters:
                    duration = time - enters[uv]
                    self.expectations.update(uv, time, duration)
