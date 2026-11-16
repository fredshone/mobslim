"""Charts and tables for the Optimisation / Performance page, from
``showcase/results/weaving.json``.

Writes ``docs/_generated/weaving_{machine,summary}.md`` and
``docs/_static/showcase/weaving_{experiment}_time_{light,dark}.svg``.
Every comparison is in wall-clock time. Each chart has two panels: when
each run's mobsims ran in the first :data:`CADENCE` seconds (one bar per
mobsim run, one lane per process running them at once), and utility in the
first :data:`SHOWN` seconds, as in the table.
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
CADENCE = 10.0  # s of wall-clock time in the mobsim-runs panel
SHOWN = 60.0  # s of wall-clock time in the utility panel and the table (of TIME_LIMIT)
TABLE_STEP = 15.0  # s between the table's utility columns

# categorical slots 1-8 of the reference palette, fixed order, per mode: the
# Controller first, then one per strand count, the same in every chart
# (AsyncWeave runs take their strand count's colour, dotted)
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
REFERENCE = "matsim-hermes"  # drawn as a dashed reference line, in the muted ink
THEMES = {
    "light": dict(text="#0b0b0b", muted="#52514e", grid="#e4e3df"),
    "dark": dict(text="#f0efec", muted="#c3c2b7", grid="#3a3a38"),
}
EXPERIMENTS = ("weaving", "async")  # one chart each
# runs of other experiments shown with an experiment's own (the reference,
# and the synchronous loop with the most strands)
SHARED = {"async": ["controller", REFERENCE,
                    f"{max(r.strands for r in RUNS if r.experiment == 'weaving')}-strands"]}


def _styles(mode: str) -> dict:
    """slug -> (color, linestyle) for every run."""
    counts = sorted({r.strands for r in RUNS if r.loop != "matsim"})  # 0: the Controller
    if len(counts) > len(SERIES[mode]):
        raise ValueError(f"{len(counts)} strand counts, {len(SERIES[mode])} colours: "
                         "lower WEAVING_BUDGET")
    color = dict(zip(counts, SERIES[mode]))
    styles = {r.slug: (color[r.strands], (0, (1, 1.2)) if r.loop == "async" else "-")
              for r in RUNS if r.loop != "matsim"}
    styles[REFERENCE] = (THEMES[mode]["muted"], (0, (4, 2)))
    return styles


def _runs(results: dict, experiment: str) -> list[tuple[str, str, dict]]:
    """(slug, name, record) of the runs in ``experiment``'s chart."""
    shared = SHARED.get(experiment, [])
    return [(r.slug, r.name, results["runs"][r.slug]) for r in RUNS
            if (r.experiment == experiment or r.slug in shared) and r.slug in results["runs"]]


def _lanes(sims: list) -> list[tuple[float, float, int]]:
    """``(start, end, lane)`` per mobsim run: each in the lowest lane free
    when it starts, so a run uses as many lanes as it ran mobsims at once."""
    free_at, out = [], []
    for start, end, *_ in sorted(sims):
        lane = next((i for i, t in enumerate(free_at) if t <= start), len(free_at))
        if lane == len(free_at):
            free_at.append(end)
        free_at[lane] = end
        out.append((start, end, lane))
    return out


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
    """One row per run: iterations done in the first :data:`SHOWN` seconds,
    and utility every :data:`TABLE_STEP` seconds up to then."""
    runs = [(r.name, results["runs"][r.slug]) for r in RUNS if r.slug in results["runs"]]
    shown = min(SHOWN, TIME_LIMIT)
    times = [TABLE_STEP * k for k in range(1, int(shown // TABLE_STEP) + 1)]
    lines = [f"| run | iterations in {shown:g} s | " + " | ".join(f"utility at {t:g} s" for t in times)
             + " |", "|---|---|" + "---|" * len(times)]
    for name, run in runs:
        done = sum(r["wall_s"] <= shown for r in run["iterations"])
        lines.append(f"| {name} | {done} | "
                     + " | ".join(f"{at_time(run, t):.2f}" for t in times) + " |")
    lines.append("")
    lines.append(f"Utility is the mean score of everything simulated in an iteration (all "
                 f"strands), as a {SMOOTH}-iteration rolling mean.")
    return "\n".join(lines) + "\n"


def chart(results: dict, experiment: str, mode: str, path: Path) -> None:
    """When each run's mobsims ran, above utility against wall time, up to
    the time limit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = THEMES[mode]
    styles = _styles(mode)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "svg.fonttype": "none"})
    runs = _runs(results, experiment)
    fig, (top, ax) = plt.subplots(2, 1, figsize=(7.2, 3.9 + 0.32 * len(runs)),
                                  gridspec_kw={"height_ratios": [0.32 * len(runs), 3.6]})
    fig.patch.set_alpha(0)
    end = min(SHOWN, TIME_LIMIT)

    # cadence: one row per run, top to bottom in legend order
    band = 0.8  # of each row's height
    for row, (slug, name, run) in enumerate(runs):
        color, _ = styles[slug]
        bars = [b for b in _lanes(run.get("sims", [])) if b[0] < CADENCE]
        lanes = 1 + max((b[2] for b in bars), default=0)
        h = band / lanes
        y0 = row - band / 2
        for lane in range(lanes):
            spans = [(a, min(b, CADENCE) - a) for a, b, ln in bars if ln == lane]
            top.broken_barh(spans, (y0 + lane * h + h * 0.1, h * 0.8), facecolors=color,
                            edgecolor="none")
    top.set_yticks(range(len(runs)))
    top.set_yticklabels([name for _, name, _ in runs], color=th["text"], fontsize=8)
    top.set_ylim(len(runs) - 0.5, -0.5)
    top.set_xlim(0, CADENCE)
    top.xaxis.set_major_formatter(lambda v, _: f"{v:g} s")
    top.set_title(f"mobsim runs in the first {CADENCE:g} s (one bar each; one lane per process)",
                  loc="left",
                  color=th["muted"], fontsize=8.5)

    # utility
    shown = []
    for slug, name, run in runs:
        color, ls = styles[slug]
        xs, ys = _series(run)
        ys = _rolling(ys)
        ax.plot(xs, ys, color=color, ls=ls, lw=2, label=name, solid_capstyle="round")
        shown.append(ys[xs <= end])
    ax.set_xlabel("wall-clock time (s)", color=th["muted"])
    ax.set_ylabel(f"utility performed\n({SMOOTH}-iteration mean)", color=th["muted"])
    ax.grid(axis="y", color=th["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for a in (top, ax):
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(th["grid"])
        a.tick_params(colors=th["muted"], length=0)
        a.set_facecolor("none")
    ax.set_xlim(0, end)
    lo, hi = float(min(y.min() for y in shown)), float(max(y.max() for y in shown))
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    leg = ax.legend(loc="lower right", frameon=False, labelcolor=th["text"], fontsize=8.5)
    for h in leg.legend_handles:
        h.set_linewidth(2)
    fig.tight_layout(h_pad=0.6)
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
