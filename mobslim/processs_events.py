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


def agent_routes(events: list) -> list:
    """Calculate the routes taken during trips based on events."""
    trip_monitor = {}
    routes = []
    for event, idx, _, uv in events:
        if event == Instruction.EnterLink and idx not in trip_monitor:
            trip_monitor[idx] = [uv]
        elif event == Instruction.EnterLink and idx in trip_monitor:
            trip_monitor[idx].append(uv)
        elif event == Instruction.ExitLink and idx in trip_monitor:
            routes.append(trip_monitor[idx])
            del trip_monitor[idx]
    return routes


def av_link_durations(plans, network, events: list) -> dict:
    """Calculate the average link durations based on events."""
    idx_monitor = {idx: None for idx in plans.keys()}
    link_durations = {link: [] for link in network.G.edges}
    for event, idx, time, uv in events:
        if idx_monitor[idx] is not None:
            prev, link = idx_monitor[idx]
            if event == Instruction.EnterLink:
                # traverse and update
                duration = time - prev
                link_durations[link].append(duration)
                idx_monitor[idx] = (time, uv)
            elif event == Instruction.ExitLink:
                # traverse and exit
                duration = time - prev
                link_durations[link].append(duration)
                idx_monitor[idx] = None
        elif event == Instruction.EnterLink and idx_monitor[idx] is None:
            # new trip
            idx_monitor[idx] = (time, uv)

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
    for event, idx, time, uv in events:

        if idx_monitor[idx] is not None:
            prev, link = idx_monitor[idx]
            if event == Instruction.EnterLink:
                # traverse and update
                duration = time - prev
                link_traverses[link].append(duration)
                idx_monitor[idx] = (time, uv)
            elif event == Instruction.ExitLink:
                # traverse and exit
                duration = time - prev
                link_traverses[link].append(duration)
                idx_monitor[idx] = None
        elif event == Instruction.EnterLink and idx_monitor[idx] is None:
            # new trip
            idx_monitor[idx] = (time, uv)

    speeds = {}
    for link_id, traverses in link_traverses.items():
        if traverses:
            avg_duration = sum(traverses) / len(traverses)
            speeds[link_id] = link_distances[link_id] / avg_duration

    return speeds


def filter_agent(events: list, agent_id: Hashable) -> list:
    """Filter events for a specific agent."""
    return [event for event in events if event[1] == agent_id]
