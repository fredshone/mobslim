from mobslim.agents import Plan, Instruction
from mobslim.network import Network
from mobslim.planner import Plan
from mobslim.listener import EventListener
from typing import Dict, Hashable


VEH_SIZE = 4  # Size of the vehicle in meters


class Sim:
    def __init__(
        self,
        network: Network,
        listener: EventListener,
    ):
        """
        Initialize the simulation with a network and expected link durations.

        :param network: The network to simulate.
        :param plans: A dictionary of plans for each agent.
        :param listener: An event listener to handle events during the simulation.
        """
        self.network = network
        self.event_listener = listener

    def reset(self, plans: Dict[Hashable, Plan]):

        self.instructions = {
            agent_id: plan.get_instructions() for agent_id, plan in plans.items()
        }
        self.time = 0
        self.agent_state = {agent_id: 0 for agent_id in plans}
        self.previous_state = {agent_id: None for agent_id in plans}

        self.sim_links = {
            edge: SimLink(attributes)
            for edge, attributes in self.network.G.edges.items()
        }
        self.event_listener.reset()

    def end_simulation(self):
        return all(
            instruction == Instruction.End for instruction in self.agent_state.values()
        )

    def run(self, steps: int = 86400):
        for _ in range(steps):
            self.step()
            if self.end_simulation():
                break
        return self.event_listener.log

    def step(self):

        # stage 1: Process all agents

        for agent_id, state in self.agent_state.items():
            instruction, uv = self.instructions[agent_id][state]
            if instruction == Instruction.EnterLink:
                previous_state = self.previous_state[agent_id]

                if previous_state is None:
                    # Check if the agent can enter the link
                    if self.sim_links[uv].can_enter(VEH_SIZE, self.time):
                        self.sim_links[uv].enter(agent_id, VEH_SIZE, self.time)
                        self.agent_state[agent_id] = state + 1
                        self.previous_state[agent_id] = state
                        self.event_listener.add(
                            Instruction.EnterLink, agent_id, self.time, uv
                        )
                    else:
                        # Agent cannot enter the link, stay in the same state
                        continue

                else:
                    instruction, puv = self.instructions[agent_id][previous_state]
                    if self.sim_links[puv].can_exit(self.time) and self.sim_links[
                        uv
                    ].can_enter(VEH_SIZE, self.time):
                        self.sim_links[puv].exit(agent_id, self.time)
                        self.sim_links[uv].enter(agent_id, VEH_SIZE, self.time)
                        self.agent_state[agent_id] = state + 1
                        self.previous_state[agent_id] = state
                        self.event_listener.add(
                            Instruction.EnterLink, agent_id, self.time, uv
                        )
                    else:
                        # Agent cannot enter the link, stay in the same state
                        continue

            elif instruction == Instruction.ExitLink:
                if self.sim_links[uv].can_exit(self.time):
                    self.sim_links[uv].exit(agent_id, self.time)
                    self.agent_state[agent_id] = state + 1
                    self.previous_state[agent_id] = state
                    self.event_listener.add(
                        Instruction.ExitLink, agent_id, self.time, uv
                    )
                else:
                    # Agent cannot exit the link, stay in the same state
                    continue

        # stage 2: Update the simulation time
        self.time += 1


class SimLink:
    def __init__(self, attributes: dict):
        """
        Initialize a simulated link

        :param attributes: A dictionary containing the attributes of the link, including 'distance'.
        """
        distance = attributes["length"]  # Distance of the link
        lanes = attributes["lanes"]  # Number of lanes on the link
        freespeed = attributes["freespeed"]  # Free speed on the link
        flow_capacity = attributes["flow_capacity"]  # Flow capacity of the link

        self.storage_capacity = distance * lanes  # meters
        self.flow_capacity = int(1 / (flow_capacity * lanes))  # seconds per vehicle
        self.min_duration = int(distance / freespeed)  # seconds

        self.queue = []
        self.earliest_next_exit = 0

    def reset(self):
        self.queue = []
        self.earliest_next_exit = 0

    def can_exit(self, time: int) -> bool:
        _, _, earliest_exit = self.queue[0]
        return earliest_exit <= time and self.has_flow_capacity(time)

    def exit(self, agent_id: Hashable, time: int) -> tuple:
        self.earliest_next_exit = time + self.flow_capacity
        return self.queue.pop(0)

    def can_enter(self, size: int, time: int) -> bool:
        return self.has_storage_capacity(size)

    def enter(self, agent_id: Hashable, size: int, time: int):
        self.add_to_queue(agent_id, size, time + self.min_duration)

    def has_storage_capacity(self, size: int) -> bool:
        request_size = sum([m for _, m, _ in self.queue]) + size
        return request_size <= self.storage_capacity

    def has_flow_capacity(self, time: int) -> bool:
        return time >= self.earliest_next_exit

    def add_to_queue(self, agent_id: Hashable, size: int, time: int):
        earliest_exit = time + self.min_duration
        self.queue.append((agent_id, size, earliest_exit))
