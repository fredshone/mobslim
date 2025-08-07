from mobslim.sim import Sim
from mobslim.planner import Planner, Router
from mobslim.agents import Plan
from mobslim.processs_events import (
    trip_durations,
    av_link_speeds,
    trip_lengths,
    filter_agent,
)
from typing import Dict, Hashable


class Optimizer:
    def __init__(
        self, sim: Sim, plans: Dict[Hashable, Plan], router: Router, replanner: Planner
    ):
        self.sim = sim
        self.plans = plans
        self.router = router
        self.replanner = replanner

    def run(self, max_runs: int = 100):
        for i in range(max_runs):
            # init plans
            self.replanner.plan(self.plans, self.router)

            self.sim.reset(plans=self.plans)
            events = self.sim.run()

            # update expected durations based on events
            self.router.update_from_events(events)

            # calculate trip lengths and link speeds
            durations = trip_durations(events)
            avg_trip_duration = sum(durations) / len(durations)

            distances = trip_lengths(self.sim.network, events)
            avg_trip_length = sum(distances) / len(distances)
            link_speeds = av_link_speeds(self.sim.network, events)
            avg_link_speed = sum(link_speeds.values()) / len(link_speeds)
            # log results
            print(
                f"Run {i}: Av. trip duration: {avg_trip_duration}, Av. trip length: {avg_trip_length}, Av. link speed: {avg_link_speed}"
            )
            agent_0_log = filter_agent(events, 0)
            print("Agent 0 log:")
            for log in agent_0_log:
                print("\t", log)
            print()
