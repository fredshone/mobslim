"""Network snapshots for the performance page: every car on the road at one moment.

For each performance case, simulates the scenario once with HermesSim and draws
the network with each car at its position at the busiest minute of the day
(most cars on the road). Positions come from the trajectories: a car enters
a link at its recorded time, travels at free speed, and waits at the link's
end until it enters the next one. Cars still inside their free-speed time are
drawn as moving, the rest as queued.

Writes ``docs/_static/showcase/net_<case>_{light,dark}.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from showcase.perf import CASES, OUT, prepare
from showcase.perf_report import STATIC, THEMES

QUEUE_SLACK = 2.0  # s past free-speed time before a car counts as queued
CELL = 7.5  # m of storage per vehicle, MATSim's default effective cell size


def snapshot(case) -> dict:
    """Simulate ``case`` with HermesSim; return the network and car positions."""
    from mobslim.config.matsim import MatsimConfig, mobsim_params
    from mobslim.events import EventsManager
    from mobslim.mobsim import HermesSim, HermesSimParams
    from verification.cases import MOBSIMS  # noqa: F401  (registers kernels)
    from verification.runner import mobslim_scenario

    prepare(case)
    cfg = MatsimConfig.load(OUT / case.name / "input" / "config.xml")
    kwargs, _ = mobsim_params(cfg)
    scenario = mobslim_scenario(cfg, cfg.input_path("plans", "inputPlansFile"), vehicles="network")
    tr = HermesSim(HermesSimParams(**kwargs)).run(scenario, EventsManager([]))
    net, p = scenario.network, tr.plans

    # each route link's entry and exit: the next link's entry, or the leg's arrival
    per_leg = np.diff(p.route_offsets)
    leg = np.repeat(np.arange(p.n_legs), per_leg)
    enter = np.asarray(tr.link_enter, dtype=float)
    exit_ = np.roll(enter, -1)
    last = p.route_offsets[1:][per_leg > 0] - 1
    exit_[last] = np.asarray(tr.leg_arrival)[leg[last]]
    links = p.route_links.astype(np.int64)
    ok = np.isfinite(enter) & np.isfinite(exit_)
    enter, exit_, links = enter[ok], exit_[ok], links[ok]

    # the busiest minute: most cars between entering and leaving a link
    minutes = np.arange(0, 30 * 3600, 60, dtype=float)
    on_road = (np.searchsorted(np.sort(enter), minutes, "right")
               - np.searchsorted(np.sort(exit_), minutes, "right"))
    t = float(minutes[np.argmax(on_road)])

    now = (enter <= t) & (t < exit_)
    lk, elapsed = links[now], t - enter[now]
    fstt = net.length[lk] / net.freespeed[lk]
    frac = np.clip(elapsed / np.maximum(fstt, 1e-9), 0.0, 1.0)
    # queued cars line up back from the link's end, in the order they reached
    # it, each taking the storage space of one vehicle
    q = elapsed > fstt
    order = np.lexsort((-elapsed, lk))  # per link, longest-waiting first
    order = order[q[order]]
    first = np.r_[True, lk[order][1:] != lk[order][:-1]]
    group = np.cumsum(first) - 1
    rank = np.arange(len(order)) - np.flatnonzero(first)[group]
    space = CELL / kwargs.get("storage_capacity_factor", 1.0) / net.lanes[lk[order]]
    frac[order] = np.clip(1.0 - rank * space / net.length[lk[order]], 0.0, 1.0)
    a, b = net.node_xy[net.from_node[lk]], net.node_xy[net.to_node[lk]]
    xy = a + frac[:, None] * (b - a)
    # shift each car to the right of its direction, so opposite directions separate
    d = b - a
    norm = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    extent = np.ptp(net.node_xy, axis=0).max()
    xy += np.column_stack([d[:, 1], -d[:, 0]]) / norm * extent * 0.0025
    return {"network": net, "time": t, "xy": xy, "queued": elapsed > fstt + QUEUE_SLACK}


def draw(case, snap: dict, mode: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    th = THEMES[mode]
    net, xy, queued = snap["network"], snap["xy"], snap["queued"]
    segs = np.stack([net.node_xy[net.from_node], net.node_xy[net.to_node]], axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.add_collection(LineCollection(segs, colors=th["grid"], linewidths=0.5, zorder=1))
    ms = 2.0 if len(xy) < 20_000 else 1.2
    ax.scatter(*xy[~queued].T, s=ms, c=th["mobslim"], linewidths=0, zorder=2, rasterized=True)
    ax.scatter(*xy[queued].T, s=ms, c=th["matsim"], linewidths=0, zorder=3, rasterized=True)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    dot = dict(marker="o", linestyle="", markersize=10)
    ax.legend(handles=[Line2D([], [], color=th["mobslim"], label="moving", **dot),
                       Line2D([], [], color=th["matsim"], label="queued", **dot)],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              labelcolor=th["text"], fontsize=13)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(patterns=()) -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        if patterns and not any(p in case.name for p in patterns):
            continue
        snap = snapshot(case)
        for mode in THEMES:
            draw(case, snap, mode, STATIC / f"net_{case.name}_{mode}.png")
        h, m = divmod(int(snap["time"]) // 60, 60)
        print(f"{case.name}: {len(snap['xy']):,} cars at {h:02d}:{m:02d}, "
              f"{int(snap['queued'].sum()):,} queued", flush=True)
