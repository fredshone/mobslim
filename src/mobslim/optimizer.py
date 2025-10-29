from typing import Dict, Hashable

from mobslim.agents import Plan
from mobslim.planners.core import BasePlanner
from mobslim.processs_events import (
    expected_link_durations,
    trip_durations,
    trip_lengths,
)
from mobslim.sim import Sim


class Optimizer:
    def __init__(
        self, sim: Sim, plans: Dict[Hashable, Plan], planner: BasePlanner
    ):
        self.sim = sim
        self.plans = plans
        self.planner = planner

    def run(self, max_runs: int = 100, verbose: bool = False):

        print("--- Initial simulation ---")
        self.sim.set(plans=self.plans)
        events = self.sim.run()
        self.report(0, events)

        print("--- Starting optimization ---")
        for i in range(1, max_runs):
            
            self.planner.update(events)
            self.planner.replan()

            self.sim.set(plans=self.planner.plans)
            events = self.sim.run()

            self.report(i ,events)

        print("--- Optimization complete ---")
        return events
    
    def report(self, i, events):
        durations = trip_durations(events)
        avg_trip_duration = sum(durations) / len(durations)

        # calculate average trip distances
        distances = trip_lengths(self.sim.network, events)
        avg_trip_length = sum(distances) / len(distances)

        # calculate average link durations
        link_durations = expected_link_durations(self.plans, self.sim.network, events)
        avg_link_duration = sum(link_durations.values()) / len(link_durations)

        print(
            f"{i}: Av. trip duration: {avg_trip_duration}, Av. trip length: {avg_trip_length}, Av. link duration: {avg_link_duration}"
        )
