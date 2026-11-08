"""Tables and charts for the performance page, from ``showcase/results/perf.json``.

Writes ``docs/_generated/perf_*.md`` (included by ``docs/simulation/performance.md``) and
``docs/_static/showcase/perf_{time,memory}_{light,dark}.svg``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from showcase.dsim import DSIM_MATSIM
from showcase.perf import CASES, RESULTS

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "docs" / "_generated"
STATIC = ROOT / "docs" / "_static" / "showcase"
REF = "MATSim Hermes"  # the baseline every speed-up is measured against

# categorical slots 1 (MobSlim) and 2 (MATSim) of the reference palette, per mode
THEMES = {
    "light": dict(mobslim="#2a78d6", matsim="#eb6834", text="#0b0b0b", muted="#52514e",
                  grid="#e4e3df", surface="none"),
    "dark": dict(mobslim="#3987e5", matsim="#d95926", text="#f0efec", muted="#c3c2b7",
                 grid="#3a3a38", surface="none"),
}


def _t(s):
    if s is not None and s < 0.001:
        return f"{s * 1e6:.0f} µs"
    if s is None:
        return "–"
    if s < 0.01:
        return f"{s * 1000:.1f} ms"
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 100:
        return f"{s:.2f} s" if s < 10 else f"{s:.1f} s"
    return f"{s:.0f} s"


def _mb(m):
    if m is None:
        return "–"
    m = max(m, 0.0)  # a mobsim that frees memory as it runs shows as 0, not -0
    return f"{m / 1024:.2f} GB" if m >= 1024 else f"{m:.0f} MB"


def _speedup(ref, t):
    if not ref or not t:
        return "–"
    x = ref / t
    return f"**{x:.0f}×** faster" if x >= 10 else (f"{x:.1f}× faster" if x >= 1.05 else
                                                   (f"{1 / x:.1f}× slower" if x < 0.95 else "same"))


def _ordered(runs: dict) -> list[tuple[str, dict]]:
    return [(k, v) for k, v in runs.items() if "error" not in v]


def case_table(name: str, case: dict) -> str:
    runs = case["runs"]
    ref = runs.get(REF, {}).get("time_s")
    lines = ["| Simulation | Mobsim time | vs MATSim Hermes | Peak memory (whole process) |",
             "|---|---|---|---|"]
    for label, r in runs.items():
        if "error" in r:
            lines.append(f"| {label} | failed | | |")
            continue
        t = r["time_s"] + (r.get("post_s") or 0)
        lines.append(f"| {label} | {_t(t)} | {_speedup(ref, t) if label != REF else 'reference'} "
                     f"| {_mb(r['peak_rss_mb'])} |")
    return "\n".join(lines) + "\n"


def headline(results: dict) -> str:
    lines = ["| Scenario | MATSim Hermes | Fastest MobSlim | Speed-up "
             "| Peak memory: MATSim Hermes → that MobSlim |", "|---|---|---|---|---|"]
    for c in CASES:
        case = results["cases"].get(c.name)
        if not case:
            continue
        runs = case["runs"]
        ref = runs.get(REF, {})
        ms = [(k, v["time_s"] + (v.get("post_s") or 0), v) for k, v in runs.items()
              if v.get("family") == "mobslim" and "error" not in v]
        if not ms or "time_s" not in ref:
            continue
        label, t, best = min(ms, key=lambda x: x[1])
        lines.append(f"| {c.name} | {_t(ref['time_s'])} "
                     f"| {_t(t)} ({label.removeprefix('MobSlim ')}) "
                     f"| **{ref['time_s'] / t:.1f}×** "
                     f"| {_mb(ref['peak_rss_mb'])} → {_mb(best['peak_rss_mb'])} |")
    return "\n".join(lines) + "\n"


def chart(results: dict, metric: str, mode: str, path: Path) -> None:
    """One full-width column chart per scenario, stacked, on one shared y range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

    th = THEMES[mode]
    cases = [c for c in CASES if c.name in results["cases"]]
    labels = list(next(iter(results["cases"].values()))["runs"])
    n = len(cases)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(n, 1, figsize=(9.6, 3.9 * n), squeeze=False)

    def value(r):
        if "error" in r or not r:
            return math.nan
        return r["time_s"] + (r.get("post_s") or 0) if metric == "time" else r["peak_rss_mb"]

    every = [value(r) for c in cases for r in results["cases"][c.name]["runs"].values()]
    lo, hi = min(v for v in every if v == v), max(v for v in every if v == v)
    for ax, c in zip(axes.flat, cases):
        runs = results["cases"][c.name]["runs"]
        vals = [value(runs.get(lab, {})) for lab in labels]
        colors = [th["matsim"] if runs.get(lab, {}).get("family") == "matsim" else th["mobslim"]
                  for lab in labels]
        names = [lab.replace("MobSlim ", "").replace("MATSim ", "MATSim\n").replace(", ", "\n")
                 for lab in labels]
        x = list(range(len(vals)))
        ax.bar(x, vals, color=colors, width=0.7, edgecolor="none")
        for xi, v in zip(x, vals):
            if v == v:
                ax.text(xi, v * 1.12 if metric == "time" else v + hi * 0.02,
                        _t(v) if metric == "time" else _mb(v),
                        ha="center", va="bottom", color=th["muted"], fontsize=8)
        if metric == "time":
            ax.set_yscale("log")
            ax.set_ylim(lo / 1.6, hi * 4)
            ax.yaxis.set_major_locator(FixedLocator(
                [10.0 ** k for k in range(-4, 4) if lo / 1.6 <= 10.0 ** k <= hi * 4]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v * 1000:g} ms" if v < 1 else f"{v:g} s"))
        else:
            top = hi * 1.15
            ax.set_ylim(0, top)
            if top > 1500:  # round GB ticks (1 GB = 1024 MB, as in the labels)
                step = next(g for g in (0.5, 1, 2, 4) if top / 1024 / g <= 5)
                ax.yaxis.set_major_locator(FixedLocator(
                    [k * step * 1024 for k in range(int(top / 1024 / step) + 1)]))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, _: "0" if v == 0 else f"{v / 1024:g} GB"))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, _: "0" if v == 0 else f"{v:g} MB"))
        ax.set_xticks(x)
        ax.set_xticklabels(names, color=th["text"], fontsize=8)
        ax.set_title(c.name, color=th["text"], fontsize=11, loc="left", fontweight="bold")
        ax.tick_params(axis="y", colors=th["muted"], labelsize=8)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=th["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor("none")
    fig.legend(handles=[Patch(color=th["matsim"], label="Java MATSim"),
                        Patch(color=th["mobslim"], label="MobSlim")],
               loc="upper right", ncol=2, frameon=False, labelcolor=th["text"], fontsize=10)
    fig.text(0.01, 0.995, ("Mobsim time per iteration (log scale, lower is faster)" if metric == "time"
                           else "Peak memory of the whole process (lower is smaller)"),
             color=th["text"], fontsize=10, va="top")
    fig.patch.set_alpha(0)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def machine_md(m: dict) -> str:
    return (f"*Measured {m['date']} on {m['cpu']} ({m['cpus']} logical CPUs, {m['memory_gb']} GB), "
            f"{m['os'].split('-with')[0]}; Python {m['python']}; {m['java']}, MATSim {m['matsim']} "
            f"(DSim: weekly build {DSIM_MATSIM}) with `-Xmx{m['xmx']}`; MobSlim commit `{m['commit']}`"
            f"{' (with uncommitted changes)' if m['dirty'] else ''}. Best of {m['repeats']} "
            "iterations or runs.*\n")


def main() -> None:
    if not RESULTS.exists():
        print(f"no results at {RESULTS}; run `python -m showcase perf` first")
        return
    results = json.loads(RESULTS.read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    (GEN / "perf_machine.md").write_text(machine_md(results["machine"]))
    (GEN / "perf_headline.md").write_text(headline(results))
    for c in CASES:
        if c.name in results["cases"]:
            (GEN / f"perf_{c.name}.md").write_text(case_table(c.name, results["cases"][c.name]))
    for metric in ("time", "memory"):
        for mode in THEMES:
            chart(results, metric, mode, STATIC / f"perf_{metric}_{mode}.svg")
    print(headline(results))


if __name__ == "__main__":
    main()
