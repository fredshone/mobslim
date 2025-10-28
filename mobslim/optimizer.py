from mobslim.sim import Sim
from mobslim.planners.core import BasePlanner
from mobslim.agents import Plan
from mobslim.processs_events import (
    trip_durations,
    av_link_speeds,
    trip_lengths,
)
from typing import Dict, Hashable


class Optimizer:
    def __init__(
        self, sim: Sim, plans: Dict[Hashable, Plan], planner: BasePlanner
    ):
        self.sim = sim
        self.plans = plans
        self.planner = planner

    def run(self, max_runs: int = 100, p: float = 0.5, verbose: bool = False):
        print("--- Starting optimization ---")
        for i in range(max_runs):
            
            # sim
            self.sim.set(plans=self.plans)
            events = self.sim.run()

            # replan
            self.planner.update(events)
            self.planner.replan(p=p)

            # calculate trip lengths and link speeds
            durations = trip_durations(events)
            avg_trip_duration = sum(durations) / len(durations)

            # calculate average trip distances
            distances = trip_lengths(self.sim.network, events)
            avg_trip_length = sum(distances) / len(distances)

            # calculate average link speeds
            link_speeds = av_link_speeds(self.plans, self.sim.network, events)
            avg_link_speed = sum(link_speeds.values()) / len(link_speeds)

            # log results
            print(
                f"Run {i}: Av. trip duration: {avg_trip_duration}, Av. trip length: {avg_trip_length}, Av. link speed: {avg_link_speed}"
            )
        print("--- Optimization complete ---")
        return events
