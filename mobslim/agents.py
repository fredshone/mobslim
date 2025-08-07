from enum import Enum


class Instruction(Enum):
    EnterLink = 1
    ExitLink = 2
    End = 3


class Plan:
    def __init__(self):
        self.components = []

    def add_trip(self, origin, destination, start_time):
        """Add a trip to the agent's plan."""
        trip = Trip(origin, destination, start_time)
        self.components.append(trip)

    def get_instructions(self):
        instructions = [
            list(c.get_instructions()) for c in self.components
        ]  # (enum, time)
        return [
            item for sublist in instructions for item in sublist
        ]  # Flatten the list

    def copy(self):
        new_plan = Plan()
        new_plan.components = [c.copy() for c in self.components]
        return new_plan


class Trip:
    def __init__(self, origin, destination, start_time):
        self.origin = origin
        self.destination = destination
        self.start_time = start_time
        self.route = None

    def get_instructions(self):
        """Get the route instructions for the trip."""
        if self.route is None:
            raise ValueError("Route has not been planned yet.")
        for edge in self.route:
            yield (Instruction.EnterLink, edge)
        yield (Instruction.ExitLink, edge)
        yield (Instruction.End, None)
