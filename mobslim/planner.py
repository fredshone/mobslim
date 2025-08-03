from mobslim.network import Network
from mobslim.expected import Durations
from mobslim.router import Router


class RePlanner:
    """
    A class to handle the planning of the Mobslim system.
    This class is responsible for managing the planning process.
    """

    def __init__(self, network: Network, link_durations: Durations, router: Router):
        self.network = network
        self.link_durations = link_durations
        self.router = router
        self.trips = {}
        self.routes = {}

    def add_trip(self, plan_id, o, d, t):
        """
        Add a trip to the planner.

        :param plan_id: The ID of the plan.
        :param o: The origin node.
        :param d: The destination node.
        :param t: The start time of the trip.
        """
        self.trips[plan_id] = (o, d)
        self.routes[plan_id] = self.router.route(o, d, t)
