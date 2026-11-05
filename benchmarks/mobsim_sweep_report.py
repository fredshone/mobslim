"""Charts for the mobsim sweep: ``uv run python -m benchmarks.mobsim_sweep_report``.

Reads ``benchmarks/results/mobsim_sweep.json`` (from ``benchmarks.mobsim_sweep``)
and writes ``time.png`` and ``memory.png`` next to it.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order, one hue per variant (8 at most). agentsim-poll+events
# is left out of the charts (it tracks agentsim-poll); the report's tables have it
COLORS = {
    "qsim": "#2a78d6",
    "qsim+events": "#eb6834",
    "agentsim-poll": "#1baf7a",
    "agentsim-chain": "#eda100",
    "agentsim-poll-r5": "#e87ba4",
    "hermes": "#008300",
    "hermes+events": "#6250d6",
    "hermes-py": "#e34948",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def chart(rows, key, ylabel, path):
    sizes = sorted({r["size"] for r in rows})
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8), sharey=True)
    for ax, size in zip(axes, sizes):
        cell = [r for r in rows if r["size"] == size and key in r]
        caps = sorted({r["capacity"] for r in cell}, reverse=True)
        for variant, color in COLORS.items():
            pts = sorted((r["capacity"], r[key]) for r in cell if r["variant"] == variant)
            if not pts:
                continue
            ax.plot([caps.index(c) for c, _ in pts], [v for _, v in pts], color=color,
                    lw=2, marker="o", ms=5, label=variant)
        r0 = cell[0]
        ax.set_title(f"grid {size}: {r0['links']} links, {r0['persons']} persons",
                     fontsize=10, color=INK, loc="left")
        delay = {r["capacity"]: r["delay_ratio"] for r in cell if r["variant"] == "qsim"}
        ax.set_xticks(range(len(caps)),
                      [f"{c:g}\n×{delay.get(c, float('nan')):.1f}" for c in caps])
        ax.set_xlabel("capacity factor / travel time vs free speed", fontsize=9, color=MUTED)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel(ylabel, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/mobsim_sweep.json")
    rows = json.loads(src.read_text())
    chart(rows, "time_s", "mobsim run time [s] (log)", src.with_name("time.png"))
    chart(rows, "peak_rss_mb", "peak RSS during run [MB] (log)", src.with_name("memory.png"))


if __name__ == "__main__":
    main()
