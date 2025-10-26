from enum import Enum
import xml.etree.ElementTree as ET

from typing import Optional


# instruction will be composed of (InstructionType, asset_id, duration)


class InstructionType(Enum):
    SOS = 0
    EnterActivity = 1  # (ActivityType, facility_id, duration)
    ExitActivity = 2  # (ActivityType, facility_id, duration)
    EnterLink = 3  # (link_id)
    ExitLink = 4  # (link_id)
    EOS = 5  # ()


class ActivityType(Enum):
    HOME = "h"
    WORK = "w"


class Plan:
    def __init__(self):
        self.components = [SOS()]

    def add_activity(self, type: ActivityType, location, duration):
        """Add an activity to the agent's plan."""
        activity = Activity(type, location, duration)
        self.components.append(activity)

    def add_trip(self, origin, destination, start_time):
        """Add a trip to the agent's plan."""
        trip = Trip(origin, destination, start_time)
        self.components.append(trip)

    def finish(self):
        self.components.append(EOS())

    def get_instructions(self):
        if len(self.components) == 0:
            raise ValueError("Plan has no components.")
        if self.components[-1] != InstructionType.EOS:
            self.finish()
        instructions = []
        for component in self.components:
            for instruction in component.get_instructions():
                instructions.append(instruction)

        # yield pairs from instructions
        for i in range(0, len(instructions) - 1, 2):
            yield (instructions[i], instructions[i + 1])

    def copy(self):
        new_plan = Plan()
        new_plan.components = [c.copy() for c in self.components]
        return new_plan

    def __repr__(self):
        return f"Plan({self.components})"


class SOS:
    def get_instructions(self):
        yield (InstructionType.SOS, None, None, 0)

    def __repr__(self):
        return "SOS()"


class Activity:
    def __init__(self, type: ActivityType, location, duration):
        self.type = type
        self.location = location
        self.duration = duration

    def get_instructions(self):
        for instruction in [
            (InstructionType.EnterActivity, self.type, self.location, self.duration),
            (InstructionType.ExitActivity, self.type, self.location, self.duration),
        ]:
            yield instruction

    def __repr__(self):
        return f"Act({self.type}, loc={self.location}, dur={self.duration})"


class Trip:
    def __init__(self, origin, destination, duration: Optional[int] = None):
        self.origin = origin
        self.destination = destination
        self.expected_duration = duration
        self.route = None

    def get_instructions(self):
        """Get the route instructions for the trip."""
        if self.route is None:
            raise ValueError("Route has not been planned yet.")
        if len(self.route) == 0:
            return
        for edge, expected_duration, minimum_duration in self.route:
            for instruction in [
                (InstructionType.EnterLink, None, edge, minimum_duration),
                (InstructionType.ExitLink, None, edge, minimum_duration),
            ]:
                yield instruction

    def __repr__(self):
        return f"Trip({self.origin}>{self.destination}, duration={self.expected_duration}, route={self.route})"


class EOS:
    def get_instructions(self):
        yield (InstructionType.EOS, None, None, 0)

    def __repr__(self):
        return "EOS()"


def load_xml(path: str):
    """Load a plans file into a dictionary of Plan objects.
    Input file is MATSim formatted XML:
    <?xml version="1.0" ?>
    <!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">
    <plans xml:lang="de-CH">
        <person id="1">
            <plan>
                    <act type="h" x="-15050" y="-150" link="1" end_time="06:00" />
                    <leg mode="car">
                            <route>2 7 12</route>
                    </leg>
                    <act type="w" x="4950" y="-150" link="20" dur="00:10" />
                    <leg mode="car">
                            <route> </route>
                    </leg>
                    <act type="w" x="4950" y="-150" link="20" dur="03:30" />
                    <leg mode="car">
                            <route>13 14 15 1</route>
                    </leg>
                    <act type="h" x="-15050" y="-150" link="1" />
            </plan>
        </person>
    """
    plans = {}
    tree = ET.parse(path)
    root = tree.getroot()
    for person in root.findall("person"):
        person_id = person.get("id")
        xml_plan = person.find("plan")
        plan = Plan()
        # loop through acts and legs in xml plan
        for component in xml_plan:
            if component.tag == "act":
                act_type = component.get("type")
                node = int(component.get("node"))

                if component.get("end_time"):
                    duration = string_to_seconds(component.get("end_time"))
                elif component.get("dur"):
                    duration = string_to_seconds(component.get("dur"))
                else:
                    duration = None

                if act_type == "h":
                    plan.add_activity(ActivityType.HOME, node, duration)
                elif act_type == "w":
                    plan.add_activity(ActivityType.WORK, node, duration)

            if component.tag == "leg":
                plan.add_trip(None, None, None)
        fixup_ods(plan)  # Ensure origin and destination are set
        plans[person_id] = plan
    return plans


def fixup_ods(plan: Plan):
    trip_idxs = []
    for i, component in enumerate(plan.components):
        if isinstance(component, Trip) and (
            component.origin is None or component.destination is None
        ):
            trip_idxs.append(i)
    for i in trip_idxs:
        plan.components[i].origin = plan.components[i - 1].location
        plan.components[i].destination = plan.components[i + 1].location
    return plan


def string_to_seconds(string: str) -> int:
    hms = string.split(":")
    if len(hms) == 2:
        return int(hms[0]) * 3600 + int(hms[1]) * 60
    elif len(hms) == 3:
        return int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    raise ValueError(f"Invalid time format: {string}")
