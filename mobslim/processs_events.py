from typing import Hashable
from mobslim.agents import Instruction


def trip_durations(events: list) -> list:
    """Calculate the lengths of trips based on events."""
    trip_monitor = {}
    durations = []
    for event, idx, time, uv in events:
        if event == Instruction.EnterLink and idx not in trip_monitor:
            trip_monitor[idx] = time
        elif event == Instruction.ExitLink and idx in trip_monitor:
            duration = time - trip_monitor[idx]
            durations.append(duration)
            del trip_monitor[idx]
    return durations


def trip_lengths(network, events: list) -> list:
    link_distances = {(u, v): network.G[u][v]["distance"] for (u, v) in network.G.edges}
    trip_lengths = []
    trip_monitor = {}
    for event, idx, _, uv in events:
        if event == Instruction.EnterLink and idx not in trip_monitor:
            trip_monitor[idx] = link_distances[uv]
        elif event == Instruction.EnterLink and idx in trip_monitor:
            trip_monitor[idx] += link_distances[uv]
        elif event == Instruction.ExitLink and idx in trip_monitor:
            trip_lengths.append(trip_monitor[idx])
            del trip_monitor[idx]
    return trip_lengths


def av_link_speeds(network, events: list) -> dict:
    """Calculate the average link speeds based on events."""
    link_ids = network.G.edges
    link_distances = {(u, v): network.G[u][v]["distance"] for (u, v) in link_ids}
    link_traverses = {link_id: [] for link_id in link_ids}
    link_monitor = {}
    for event, idx, time, uv in events:
        if event == Instruction.EnterLink and uv not in link_ids:
            link_monitor[uv] = time
        elif (
            event in (Instruction.EnterLink, Instruction.ExitLink)
            and uv in link_monitor
        ):
            duration = time - link_monitor[uv]
            if duration > 0:
                link_traverses[uv].append(duration)
            del link_monitor[uv]

    speeds = {}
    for link_id, traverses in link_traverses.items():
        if traverses:
            avg_duration = sum(traverses) / len(traverses)
            speeds[link_id] = link_distances[link_id] / avg_duration
        else:
            speeds[link_id] = 0

    return speeds


def filter_agent(events: list, agent_id: Hashable) -> list:
    """Filter events for a specific agent."""
    return [event for event in events if event[1] == agent_id]
