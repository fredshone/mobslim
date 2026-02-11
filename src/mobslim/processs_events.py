from typing import Hashable

from mobslim.entities.agents import Activity, InstructionType, Plan, Trip
from mobslim.entities.networks import Networks


def events_to_plans(events: list) -> dict:
    """Parse events into agent plans"""
    plans = {}
    states = {}
    trip_starts = {}
    last_trip_mode = {}
    routes = {}  # edge, expected, minimum

    for time, idx, instruction in events:
        event = instruction[0]
        if event == InstructionType.SOS:
            plans[idx] = Plan()
            trip_starts[idx] = None

        elif event == InstructionType.EnterFacility:
            states[idx] = time
            # check for trip end
            if trip_starts[idx]:
                _, act, d, _ = instruction
                u, start_time = trip_starts[idx]
                nw_mode = last_trip_mode[idx]
                trip = Trip(u, d, network_mode=nw_mode)
                trip.route = routes[idx]
                plans[idx].add(trip)
                del trip_starts[idx]
                del routes[idx]

        elif event == InstructionType.ExitFacility:
            _, act, location, _ = instruction
            duration = time - states[idx]
            plans[idx].add(Activity(act, location, duration))
            # start a trip
            trip_starts[idx] = location, time
            # start route
            routes[idx] = []

        elif event == InstructionType.EnterLink:
            _, nw_mode, uv, minimum_duration = instruction
            states[idx] = time

        elif event == InstructionType.ExitLink:
            _, _, uv, minimum_duration = instruction
            duration = time - states[idx]
            routes[idx].append((uv, duration, minimum_duration))
            last_trip_mode[idx] = nw_mode

        elif event == InstructionType.EOS:
            plans[idx].finish()

    return plans


def trip_durations(events: list, network_mode: str = "road") -> list:
    """Calculate the lengths of trips based on events."""
    trip_monitor = {}
    durations = []
    for time, idx, instruction in events:
        event, _, _, _ = instruction
        if event == InstructionType.ExitFacility and idx not in trip_monitor:
            trip_monitor[idx] = time
        elif idx in trip_monitor and event == InstructionType.EnterLink:
            _, nw_mode, _, _ = instruction
            if not nw_mode == network_mode:
                del trip_monitor[idx]
        elif event == InstructionType.EnterFacility and idx in trip_monitor:
            duration = time - trip_monitor[idx]
            durations.append(duration)
            del trip_monitor[idx]
    return durations


def trip_lengths(
    networks: Networks, events: list, network_mode: str = "road"
) -> list:
    G = networks[network_mode]
    link_distances = {
        (u, v): data["length"] for (u, v, data) in G.edges(data=True)
    }
    trip_lengths = []
    trip_monitor = {}
    for _, idx, instruction in events:
        event, _, uv, _ = instruction
        if event == InstructionType.ExitFacility:
            trip_monitor[idx] = 0
        elif event == InstructionType.EnterLink:
            trip_monitor[idx] += link_distances[uv]
        elif event == InstructionType.EnterFacility and idx in trip_monitor:
            trip_lengths.append(trip_monitor[idx])
            del trip_monitor[idx]
    return trip_lengths


def av_link_durations(
    plans, networks: Networks, events: list, network_mode: str = "car"
) -> dict:
    """Calculate the average link durations based on events."""
    G = networks[network_mode]
    idx_monitor = {idx: None for idx in plans.keys()}
    link_durations = {link: [] for link in G.edges}
    for time, idx, instruction in events:
        event, _, uv, _ = instruction

        if event == InstructionType.EnterLink:
            idx_monitor[idx] = (time, uv)
        elif event == InstructionType.ExitLink:
            prev, link = idx_monitor[idx]
            duration = time - prev
            link_durations[link].append(duration)

    # Calculate average durations
    avg_durations = {
        link: sum(durations) / len(durations) if durations else None
        for link, durations in link_durations.items()
    }
    return avg_durations


def expected_link_durations(
    plans, networks: Networks, events: list, network_mode: str = "road"
) -> dict:
    """Calculate the expected link durations based on events."""
    G = networks[network_mode]
    idx_monitor = {idx: None for idx in plans.keys()}
    link_durations = {link: [] for link in G.edges}
    for time, idx, instruction in events:
        event, _, uv, _ = instruction

        if event == InstructionType.EnterLink:
            idx_monitor[idx] = (time, uv)
        elif event == InstructionType.ExitLink:
            prev, link = idx_monitor[idx]
            duration = time - prev
            link_durations[link].append(duration)

    # get minimyum durations per link
    min_durations = networks.minimum_durations(network_mode=network_mode)

    # Calculate expected durations
    expected_durations = {
        link: (
            sum(durations) / len(durations)
            if durations
            else min_durations[link]
        )
        for link, durations in link_durations.items()
    }

    return expected_durations


def av_link_speeds(
    plans, networks: Networks, events: list, network_mode: str = "road"
) -> dict:
    """Calculate the average link speeds based on events."""
    idx_monitor = {idx: None for idx in plans.keys()}
    link_ids = networks[network_mode].edges
    link_distances = {(u, v): networks.G[u][v]["length"] for (u, v) in link_ids}
    link_traverses = {link_id: [] for link_id in link_ids}
    for time, idx, instruction in events:
        event, _, uv, _ = instruction

        if event == InstructionType.EnterLink:
            idx_monitor[idx] = (time, uv)
        elif event == InstructionType.ExitLink:
            prev, link = idx_monitor[idx]
            duration = time - prev
            speed = link_distances[link] / duration
            link_traverses[link].append(speed)

    speeds = {}
    for link_id, traverses in link_traverses.items():
        if traverses:
            avg_duration = sum(traverses) / len(traverses)
            speeds[link_id] = link_distances[link_id] / avg_duration

    return speeds


def filter_agent(events: list, agent_id: Hashable) -> list:
    """Filter events for a specific agent."""
    return [event for event in events if event[1] == agent_id]
