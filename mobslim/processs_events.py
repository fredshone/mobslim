from typing import Hashable
from mobslim.agents import InstructionType


def trip_durations(events: list) -> list:
    """Calculate the lengths of trips based on events."""
    trip_monitor = {}
    durations = []
    for time, idx, instruction in events:
        event, _, _, _ = instruction
        if event == InstructionType.ExitActivity and idx not in trip_monitor:
            trip_monitor[idx] = time
        elif event == InstructionType.EnterActivity and idx in trip_monitor:
            duration = time - trip_monitor[idx]
            durations.append(duration)
            del trip_monitor[idx]
    return durations


def trip_lengths(network, events: list) -> list:
    link_distances = {(u, v): network.G[u][v]["length"] for (u, v) in network.G.edges}
    trip_lengths = []
    trip_monitor = {}
    for _, idx, instruction in events:
        event, _, uv, _ = instruction
        if event == InstructionType.ExitActivity:
            trip_monitor[idx] = 0
        elif event == InstructionType.EnterLink:
            trip_monitor[idx] += link_distances[uv]
        elif event == InstructionType.EnterActivity and idx in trip_monitor:
            trip_lengths.append(trip_monitor[idx])
            del trip_monitor[idx]
    return trip_lengths


def av_link_durations(plans, network, events: list) -> dict:
    """Calculate the average link durations based on events."""
    idx_monitor = {idx: None for idx in plans.keys()}
    link_durations = {link: [] for link in network.G.edges}
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


def av_link_speeds(plans, network, events: list) -> dict:
    """Calculate the average link speeds based on events."""
    idx_monitor = {idx: None for idx in plans.keys()}
    link_ids = network.G.edges
    link_distances = {(u, v): network.G[u][v]["length"] for (u, v) in link_ids}
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
