from mobslim.planner import RePlanner
from mobslim.network import Network
from mobslim.expected import Durations


class Sim:
    def __init__(
        self, network: Network, link_durations: Durations, replanner: RePlanner
    ):
        """
        Initialize the simulation with a network and expected link durations.

        :param network: The network to simulate.
        :param link_durations: The expected durations for the links in the network.
        :param replanner: The replanner to manage the planning process.
        """
        self.network = network
        self.link_durations = link_durations
        self.replanner = replanner
        self.time = 0

        self.sim_links = {
            edge: SimLink(attributes) for edge, attributes in network.G.edges.items()
        }

    def reset(self):
        self.time = 0


class SimLink:
    def __init__(self, attributes: dict):
        """
        Initialize a simulated link with a given distance.

        :param attributes: A dictionary containing the attributes of the link, including 'distance'.
        """
        self.distance = attributes["distance"]  # Distance of the link
        self.lanes = attributes["lanes"]  # Number of lanes on the link
        self.freespeed = attributes["freespeed"]  # Free speed on the link

        self.capacity = self.distance * self.lanes
        self.queue = []
