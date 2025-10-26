import heapq
from mobslim.agents import Plan, InstructionType
from mobslim.network import Network
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

    def set(self, plans: Dict[Hashable, Plan]):

        self.instructions = {
            agent_id: plan.get_instructions() for agent_id, plan in plans.items()
        }

        self.queue = []
        for agent_id, instruction_q in self.instructions.items():
            instruction_a, instruction_b = next(instruction_q)
            min_duration = instruction_a[3]
            heapq.heappush(
                self.queue, (min_duration, agent_id, (instruction_a, instruction_b))
            )

        self.time = 0

        self.sim_links = {
            edge: SimLink(attributes)
            for edge, attributes in self.network.G.edges.items()
        }
        self.event_listener.reset()

    def run(self, steps: int = 86400):
        while self.queue and self.time < steps:
            self.step_instruction()
        return self.event_listener.log

    def can_exit(self, agent_id, instruction_a):
        if instruction_a[0] == InstructionType.ExitLink:
            _, _, uv, _ = instruction_a
            if self.sim_links[uv].can_exit(self.time):
                return True
            return False
        return True

    def can_enter(self, agent_id, instruction_b):
        if instruction_b[0] == InstructionType.EnterLink:
            _, _, uv, _ = instruction_b
            if self.sim_links[uv].can_enter(VEH_SIZE, self.time):
                return True
            return False
        return True

    def step_instruction(self):

        self.time, agent_id, (instruction_a, instruction_b) = heapq.heappop(self.queue)

        if not self.can_exit(agent_id, instruction_a):
            # cannot exit, requeue with a small delay
            heapq.heappush(
                self.queue, (self.time + 1, agent_id, (instruction_a, instruction_b))
            )
            return

        if not self.can_enter(agent_id, instruction_b):
            # cannot enter, requeue with a small delay
            heapq.heappush(
                self.queue, (self.time + 1, agent_id, (instruction_a, instruction_b))
            )
            return

        # do link exit and entry
        if instruction_a[0] == InstructionType.ExitLink:
            _, _, uv, _ = instruction_a
            self.sim_links[uv].exit(agent_id, self.time)

        if instruction_b[0] == InstructionType.EnterLink:
            _, _, uv, _ = instruction_b
            self.sim_links[uv].enter(agent_id, VEH_SIZE, self.time)

        self.event_listener.add(self.time, agent_id, instruction_a)
        self.event_listener.add(self.time, agent_id, instruction_b)

        if instruction_b[0] == InstructionType.EOS:
            # end of simulation for this agent
            return

        # schedule next instruction after activity duration
        next_instruction = next(self.instructions[agent_id])
        min_duration = next_instruction[0][3]
        heapq.heappush(
            self.queue, (self.time + min_duration, agent_id, next_instruction)
        )

        return


class SimLink:
    def __init__(self, attributes: dict):
        """
        Initialize a simulated link

        :param attributes: A dictionary containing the attributes of the link, including 'length'.
        """
        length = attributes["length"]  # Distance of the link
        lanes = attributes["lanes"]  # Number of lanes on the link
        freespeed = attributes["freespeed"]  # Free speed on the link
        flow_capacity = attributes["flow_capacity"]  # Flow capacity of the link

        self.storage_capacity = length * lanes  # meters
        self.flow_capacity = int(1 / (flow_capacity * lanes))  # seconds per vehicle
        self.min_duration = int(length / freespeed)  # seconds

        self.queue = []
        self.earliest_next_exit = 0

    def reset(self):
        self.queue = []
        self.earliest_next_exit = 0

    def can_exit(self, time: int) -> bool:
        _, _, earliest_exit = self.queue[
            0
        ]  # todo: this is a duplicate check in sim loop
        return earliest_exit <= time and self.has_flow_capacity(time)

    def exit(self, agent_id: Hashable, time: int) -> tuple:
        self.earliest_next_exit = time + self.flow_capacity
        return self.queue.pop(0)

    def can_enter(self, size: int, time: int) -> bool:
        return self.has_storage_capacity(size)

    def enter(self, agent_id: Hashable, size: int, time: int):
        self.add_to_queue(agent_id, size, time)

    def has_storage_capacity(self, size: int) -> bool:
        request_size = sum([m for _, m, _ in self.queue]) + size
        return request_size <= self.storage_capacity

    def has_flow_capacity(self, time: int) -> bool:
        return time >= self.earliest_next_exit

    def add_to_queue(self, agent_id: Hashable, size: int, time: int):
        earliest_exit = time + self.min_duration
        self.queue.append((agent_id, size, earliest_exit))
