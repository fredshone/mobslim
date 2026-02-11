from typing import Dict, Hashable

from mobslim.entities.agents import Plan
from mobslim.planners.core import BasePlanner
from mobslim.processs_events import (
    expected_link_durations,
    trip_durations,
    trip_lengths,
)
from mobslim.sim import Sim


class Optimizer:
    def __init__(
        self,
        sim: Sim,
        plans: Dict[Hashable, Plan],
        planner: BasePlanner,
        network_modes: list[str] = ["road"],
    ):
        self.sim = sim
        self.plans = plans
        self.planner = planner
        self.network_modes = network_modes

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

            self.report(i, events)

        print("--- Optimization complete ---")
        return events

    def report(self, i, events):
        for mode in self.network_modes:
            durations = trip_durations(events, network_mode=mode)
            avg_trip_duration = sum(durations) / len(durations)

            # calculate average trip distances
            distances = trip_lengths(
                self.sim.networks, events, network_mode=mode
            )
            avg_trip_length = sum(distances) / len(distances)

            # calculate average link durations
            link_durations = expected_link_durations(
                self.plans, self.sim.networks, events, network_mode=mode
            )
            avg_link_duration = sum(link_durations.values()) / len(
                link_durations
            )

            print(
                f"{i}: Av. {mode} trip duration: {avg_trip_duration}, Av. {mode} trip length: {avg_trip_length}, Av. {mode} link duration: {avg_link_duration}"
            )
