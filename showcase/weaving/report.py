"""Charts and tables for the Optimisation / Performance page, from
``showcase/results/weaving.json``.

Writes ``docs/_generated/weaving_*.md`` and
``docs/_static/showcase/weaving_{experiment}_{x}_{light,dark}.svg``
(``x``: ``time`` or ``iteration``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from showcase.weaving.experiment import BUDGET, RESULTS, RUNS

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "docs" / "_generated"
STATIC = ROOT / "docs" / "_static" / "showcase"
SMOOTH = 5  # iterations in the rolling mean
FINAL = 10  # last iterations that define the settled level

# categorical slots 1-5 of the reference palette, fixed order, per mode
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
THEMES = {
    "light": dict(text="#0b0b0b", muted="#52514e", grid="#e4e3df"),
    "dark": dict(text="#f0efec", muted="#c3c2b7", grid="#3a3a38"),
}
EXPERIMENTS = {
    "weaving": ("Controller", "HermesSim, 1 strand to the budget"),
    "budget": (f"p={BUDGET}, w=1", "PSim, p threads x w strands = the budget"),
}


def _runs(results: dict, experiment: str) -> list[tuple[str, dict]]:
    return [(r.name, results["runs"][r.slug]) for r in RUNS
            if r.experiment == experiment and r.slug in results["runs"]]


def _series(run: dict, x: str):
    rows = run["iterations"]
    xs = np.array([r["wall_s"] if x == "time" else r["iteration"] for r in rows], dtype=float)
    ys = np.array([r["score_performed"] for r in rows], dtype=float)
    return xs, ys


def _rolling(ys: np.ndarray) -> np.ndarray:
    k = min(SMOOTH, len(ys))
    c = np.cumsum(np.r_[0.0, ys])
    out = np.empty_like(ys)
    for i in range(len(ys)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def final_level(run: dict) -> float:
    """The level a run settles at: its mean over the last :data:`FINAL` iterations."""
    ys = _series(run, "time")[1]
    return float(ys[-FINAL:].mean())


def reaches(run: dict, level: float):
    """``(iteration, wall seconds)`` at which the run's rolling mean first
    reaches ``level``; ``(None, None)`` if it never does."""
    xs, ys = _series(run, "time")
    hit = np.flatnonzero(_rolling(ys) >= level)
    if not len(hit):
        return None, None
    i = int(hit[0])
    return run["iterations"][i]["iteration"], float(xs[i])


def at_time(run: dict, t: float) -> float:
    """The run's rolling mean at wall time ``t`` (its last value before it)."""
    xs, ys = _series(run, "time")
    return float(_rolling(ys)[max(int(np.searchsorted(xs, t, side="right")) - 1, 0)])


def table(results: dict, experiment: str) -> str:
    ref_name = EXPERIMENTS[experiment][0]
    rows = _runs(results, experiment)
    ref = dict(rows)[ref_name]
    target = final_level(ref)
    budget = ref["iterations"][-1]["wall_s"]
    ref_time = reaches(ref, target)[1]
    lines = [f"| run | s per iteration | final utility | reaches {target:.2f} at | "
             f"wall time | vs {ref_name} | utility after {budget:.0f} s |",
             "|---|---|---|---|---|---|---|"]
    for name, run in rows:
        it = run["iterations"]
        per_it = (it[-1]["wall_s"] - it[0]["wall_s"]) / max(len(it) - 1, 1)
        i, t = reaches(run, target)
        speed = "–" if t is None else ("reference" if name == ref_name else f"{ref_time / t:.2f}×")
        lines.append(f"| {name} | {per_it:.2f} | {final_level(run):.2f} | "
                     f"{'never' if i is None else f'iteration {i}'} | "
                     f"{'–' if t is None else f'{t:.0f} s'} | {speed} | "
                     f"{at_time(run, budget):.2f} |")
    lines.append("")
    lines.append(f"Utility is the mean score of everything simulated in an iteration (all strands), "
                 f"as a {SMOOTH}-iteration rolling mean. *Final utility* is the mean over the last "
                 f"{FINAL} iterations. *Reaches* is when a run first matches {ref_name}'s final "
                 f"utility; the last column is each run's utility once {ref_name} has finished "
                 f"({budget:.0f} s). Wall time starts after the one-off routing of the input "
                 "plans (the same for every run) and includes starting the worker processes.")
    return "\n".join(lines) + "\n"


def chart(results: dict, experiment: str, x: str, mode: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = THEMES[mode]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for (name, run), color in zip(_runs(results, experiment), SERIES[mode]):
        xs, ys = _series(run, x)
        ax.plot(xs, _rolling(ys), color=color, lw=2, label=name, solid_capstyle="round")
    ax.set_xlabel("wall-clock time (s)" if x == "time" else "iteration", color=th["muted"])
    ax.set_ylabel(f"utility performed\n({SMOOTH}-iteration mean)", color=th["muted"])
    ax.grid(axis="y", color=th["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.tick_params(colors=th["muted"], length=0)
    ax.set_xlim(left=0)
    leg = ax.legend(loc="upper right", frameon=False, labelcolor=th["text"], fontsize=8.5)
    for h in leg.legend_handles:
        h.set_linewidth(2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)


def machine_md(m: dict) -> str:
    return (f"{m['cpu']}: {m['physical_cores']} physical cores ({m['logical_cpus']} logical), "
            f"{m['memory_gb']} GB memory, Python {m['python']}.\n")


def main() -> None:
    results = json.loads(RESULTS.read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "weaving_machine.md").write_text(machine_md(results["machine"]))
    for experiment in EXPERIMENTS:
        if not _runs(results, experiment):
            continue
        (GEN / f"weaving_{experiment}.md").write_text(table(results, experiment))
        for x in ("time", "iteration"):
            for mode in ("light", "dark"):
                chart(results, experiment, x, mode, STATIC / f"weaving_{experiment}_{x}_{mode}.svg")
    print(f"wrote {GEN}/weaving_*.md and {STATIC}/weaving_*.svg")


if __name__ == "__main__":
    main()
