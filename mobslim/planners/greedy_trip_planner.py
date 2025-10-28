from random import random

from mobslim.planners.core import BasePlanner
from mobslim.agents import Trip, Activity
from mobslim.network import Network
from mobslim.planners.rerouters.simple_rerouter import BaseRouter
from mobslim.processs_events import events_to_plans


class GreedyTripPlanner(BasePlanner):
    """Currently just a trip planner.
    Assumes start of day at 0 and end at 86400 (24 hours)
    """

    def __init__(self, plans, router: BaseRouter, network: Network):
        self.plans = plans
        self.router = router
        self.network = network

    def update(self, events):
        # parse events into plans and overwrite previous
        self.plans = events_to_plans(events)
        # update router
        self.router.update(plans = self.plans, network = self.network, events = events)

    def plan(self):
        self.replan(p = 1.0)
        
    def replan(self, p: float = 0.5, ):
        if p < 0 or p > 1:
            raise ValueError("Probability p must be between 0 and 1.")
        for plan in self.plans.values():
            if random() > p:
                continue  # Skip planning for this agent
            time = 0
            for component in plan.components:

                if isinstance(component, Trip):
                    component.route, component.expected_duration = self.router.get_route(
                        component.origin, component.destination, time
                    )
                    time += component.expected_duration

                if isinstance(component, Activity):
                    time += component.duration


