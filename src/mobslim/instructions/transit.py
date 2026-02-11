from typing import Dict, Hashable, List, Tuple

from mobslim import Schedules
from mobslim.entities.agents import EOS, SOS, InstructionType

type LineID = Hashable
type RouteID = Hashable
type DepartureID = Hashable
type Instruction = Tuple[InstructionType, str, Hashable, int]


def build_transit_instructions(
    schedules: Schedules,
) -> Dict[Tuple[LineID, RouteID, DepartureID], List[Instruction]]:
    """Build transit instructions from a schedule.

    Args:
        schedules (Schedules): The schedules to build instructions from.

    Returns:
        Dict[LineID, Dict[RouteID, Dict[DepartureID, List[Instruction]]]]: Lists of instructions hashed by line, route and departure.
    """

    instructions = {}

    # retrieve stops as a dict of links to stop ids
    links_to_stops = {v: k for k, v in schedules.stop_links.items()}

    for line_id, routes in schedules.lines.items():
        instructions[line_id] = {}
        for route_id, route_data in routes.items():
            instructions[line_id][route_id] = {}
            network_mode = route_data["transport_mode"]
            route_stops = route_data["route_stops"]
            route_links = route_data["route_links"]

            for departure in route_data["departures"]:
                departure_id = departure["id"]
                time = departure["departureTime"]
                veh_id = departure["vehicleRefId"]

                # Build instructions
                instructions = [SOS()]

                for link in route_links:
                    # check for stop at this link
                    stop_profile = route_stops.get(links_to_stops.get(link))

                    if stop_profile is None:
                        # Traverse link
                        instructions.append(
                            Instruction(
                                InstructionType.EnterLink,
                                network_mode,
                                link,
                                time=0,  # todo
                            )
                        )
                        instructions.append(
                            Instruction(
                                InstructionType.ExitLink,
                                network_mode,
                                link,
                                time=0,  # todo
                            )
                        )

                    else:
                        # Traverse link to stop
                        instructions.append(
                            Instruction(
                                InstructionType.EnterLink,
                                network_mode,
                                link,
                                time=0,  # todo
                            )
                        )
                        instructions.append(
                            Instruction(
                                InstructionType.ExitLink,
                                network_mode,
                                link,
                                time=0,  # todo
                            )
                        )

                        # enter stop
                        time = time + stop_profile["arrival_offset"]
                        instructions.append(
                            Instruction(
                                InstructionType.EnterFacility,
                                "stop",
                                stop_profile["ref_id"],
                                time,
                            )
                        )
                        # exit stop
                        time = time + stop_profile["departure_offset"]
                        instructions.append(
                            Instruction(
                                InstructionType.ExitFacility,
                                "stop",
                                stop_profile["ref_id"],
                                time,
                            )
                        )

                        # Re enter link after stop
                        instructions.append(
                            Instruction(
                                InstructionType.EnterLink,
                                network_mode,
                                link,
                                time=0,  # todo
                            )
                        )
                        instructions.append(
                            Instruction(
                                InstructionType.ExitLink,
                                network_mode,
                                link,
                                time=0,  # todo
                            )
                        )
                instructions.append(EOS())
                instructions[(line_id, route_id, departure_id)] = instructions
    return instructions
