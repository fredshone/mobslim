"""Charts and tables for the Optimisation / Performance page, from
``showcase/results/weaving.json``.

Writes ``docs/_generated/weaving_{machine,summary}.md`` and
``docs/_static/showcase/weaving_{experiment}_time_{light,dark}.svg``.
Every comparison is in wall-clock time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from showcase.weaving.experiment import RESULTS, RUNS, TIME_LIMIT

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "docs" / "_generated"
STATIC = ROOT / "docs" / "_static" / "showcase"
SMOOTH = 5  # iterations in the rolling mean

# categorical slots 1-5 of the reference palette, fixed order, per mode
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
THEMES = {
    "light": dict(text="#0b0b0b", muted="#52514e", grid="#e4e3df"),
    "dark": dict(text="#f0efec", muted="#c3c2b7", grid="#3a3a38"),
}
EXPERIMENTS = ("weaving", "async")  # one chart each
# runs of other experiments shown with an experiment's own (the reference,
# and the synchronous loop with the most strands)
SHARED = {"async": ["controller", f"{max(r.strands for r in RUNS if r.experiment == 'weaving')}-strands"]}


def _runs(results: dict, experiment: str) -> list[tuple[str, dict]]:
    shared = SHARED.get(experiment, [])
    return [(r.name, results["runs"][r.slug]) for r in RUNS
            if (r.experiment == experiment or r.slug in shared) and r.slug in results["runs"]]


def _series(run: dict):
    rows = run["iterations"]
    xs = np.array([r["wall_s"] for r in rows], dtype=float)
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


def at_time(run: dict, t: float) -> float:
    """The run's rolling mean at wall time ``t`` (its last value before it)."""
    xs, ys = _series(run)
    return float(_rolling(ys)[max(int(np.searchsorted(xs, t, side="right")) - 1, 0)])


def table(results: dict) -> str:
    """One row per run: iterations done, and utility at each minute up to
    the time limit."""
    runs = [(r.name, results["runs"][r.slug]) for r in RUNS if r.slug in results["runs"]]
    times = [60.0 * m for m in range(1, int(TIME_LIMIT // 60) + 1)]
    lines = ["| run | iterations | " + " | ".join(f"utility at {t / 60:.0f} min" for t in times) + " |",
             "|---|---|" + "---|" * len(times)]
    for name, run in runs:
        lines.append(f"| {name} | {len(run['iterations'])} | "
                     + " | ".join(f"{at_time(run, t):.2f}" for t in times) + " |")
    lines.append("")
    lines.append(f"Utility is the mean score of everything simulated in an iteration (all "
                 f"strands), as a {SMOOTH}-iteration rolling mean.")
    return "\n".join(lines) + "\n"


def chart(results: dict, experiment: str, mode: str, path: Path) -> None:
    """Utility against wall time, up to the time limit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = THEMES[mode]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    runs = _runs(results, experiment)
    end = TIME_LIMIT
    shown = []
    for (name, run), color in zip(runs, SERIES[mode]):
        xs, ys = _series(run)
        ys = _rolling(ys)
        ax.plot(xs, ys, color=color, lw=2, label=name, solid_capstyle="round")
        shown.append(ys[xs <= end])
    ax.set_xlabel("wall-clock time (s)", color=th["muted"])
    ax.set_ylabel(f"utility performed\n({SMOOTH}-iteration mean)", color=th["muted"])
    ax.grid(axis="y", color=th["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.tick_params(colors=th["muted"], length=0)
    ax.set_xlim(0, end)
    lo, hi = float(min(y.min() for y in shown)), float(max(y.max() for y in shown))
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    leg = ax.legend(loc="lower right", frameon=False, labelcolor=th["text"], fontsize=8.5)
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
    (GEN / "weaving_summary.md").write_text(table(results))
    for experiment in EXPERIMENTS:
        if not _runs(results, experiment):
            continue
        for mode in ("light", "dark"):
            chart(results, experiment, mode, STATIC / f"weaving_{experiment}_time_{mode}.svg")
    print(f"wrote {GEN}/weaving_*.md and {STATIC}/weaving_*.svg")


if __name__ == "__main__":
    main()
