import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mobslim.agents import InstructionType


def build_traces(events, node_positions, step=1, start=0, limit=None):
    agent_traces = {}
    start_link_traverse = {}

    print("Building traces...")

    for time, agent_id, instruction in events:
        time = int(time)

        if limit is not None and time > limit:
            break

        if instruction[0] == InstructionType.SOS:
            agent_traces[agent_id] = []
        
        if time < start:
            continue

        elif instruction[0] == InstructionType.EnterActivity:
            loc = node_positions[instruction[2]]
            agent_traces[agent_id].append((time, loc))
        elif instruction[0] == InstructionType.EnterLink:
            start_link_traverse[agent_id] = time
        elif instruction[0] == InstructionType.ExitLink:
            u, v = instruction[2]
            start_time = start_link_traverse[agent_id]
            duration = time - start_time
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]
            dx, dy = x1 - x0, y1 - y0
            for t in range(0, duration, step):
                p = t/duration
                x = x0 + p * dx
                y = y0 + p * dy
                agent_traces[agent_id].append((start_time + t, (x,y)))
        
    return agent_traces, time


def plot_network(network, ax):
    locs = [(network.node_positions[u], network.node_positions[v]) for (u, v) in network.G.edges()]
    return [ax.plot([u[0],v[0]], [u[1], v[1]], color="grey") for (u,v) in locs]


def animate_traces(network, agent_traces, frames):

    print("Animating traces...")
    
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()

    _ = plot_network(network, ax)
    starts = {idx: trace.pop(0)[1] for idx, trace in agent_traces.items()}
    scatters = {idx: ax.scatter(loc[0], loc[1], color="red") for idx, loc in starts.items()}

    def animate(frame):
        current_time = frame
        for idx, locations in agent_traces.items():
            if not locations:
                continue
            time, pos = locations[0]
            if time <= current_time:
                _, pos = locations.pop(0)
                scatters[idx].set_offsets(pos)

        return scatters.values()

    ani = animation.FuncAnimation(
        fig, animate, frames=frames, blit=True
    )
    return ani


def animate_events(network, events, step=1, start=0, limit=None):
    agent_traces, finish_time = build_traces(events, network.node_positions, step=step, start=start, limit=limit)
    frames = range(start, finish_time + 1, step)
    ani = animate_traces(network, agent_traces, frames=frames)
    return ani