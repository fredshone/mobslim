"""Matplotlib animation of agents moving through the network (for notebooks)."""

from typing import Iterable, Optional

from mobslim.events import (
    ActivityEnd,
    ActivityStart,
    Event,
    LinkEnter,
    LinkLeave,
)
from mobslim.network import Network


def build_traces(network: Network, events: Iterable[Event], step: float = 1,
                 start: float = 0, limit: Optional[float] = None):
    """Per-person lists of ``(time, (x, y))`` positions, sampled every ``step`` s."""
    xy = network.node_xy
    traces: dict[int, list] = {}
    entered: dict[int, float] = {}
    last = start
    for e in events:
        if limit is not None and e.time > limit:
            break
        last = e.time
        trace = traces.setdefault(e.person, [])
        if isinstance(e, (ActivityStart, ActivityEnd)):
            if e.time >= start:
                trace.append((e.time, tuple(xy[e.node])))
        elif isinstance(e, LinkEnter):
            entered[e.person] = e.time
        elif isinstance(e, LinkLeave):
            t0 = entered.pop(e.person)
            (x0, y0) = xy[network.from_node[e.link]]
            (x1, y1) = xy[network.to_node[e.link]]
            duration = max(e.time - t0, 1e-9)
            t = t0
            while t < e.time:
                if t >= start:
                    p = (t - t0) / duration
                    trace.append((t, (x0 + p * (x1 - x0), y0 + p * (y1 - y0))))
                t += step
    return {k: v for k, v in traces.items() if v}, last


def animate_events(network: Network, events: Iterable[Event], step: float = 1,
                   start: float = 0, limit: Optional[float] = None, mode: Optional[str] = None):
    """Return a ``FuncAnimation`` of the given events (render with ``ani`` in Jupyter)."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    traces, finish = build_traces(network, events, step, start, limit)
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    xy = network.node_xy
    for link in range(network.n_links):
        if mode is not None and mode not in network.link_modes[link]:
            continue
        u, v = network.from_node[link], network.to_node[link]
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]], color="grey", lw=0.8)

    queues = {p: list(trace) for p, trace in traces.items()}
    dots = {p: ax.scatter(*trace[0][1], color="red", s=12) for p, trace in queues.items()}

    def update(now):
        for p, trace in queues.items():
            while trace and trace[0][0] <= now:
                _, pos = trace.pop(0)
                dots[p].set_offsets(pos)
        return list(dots.values())

    frames = [start + i * step for i in range(int((finish - start) // step) + 1)]
    return animation.FuncAnimation(fig, update, frames=frames, blit=True)
