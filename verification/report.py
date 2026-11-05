"""Markdown report from the ``result.json`` files of the cases that were run."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from verification import compare
from verification.matsim import MATSIM_VERSION

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"

_MARK = {"green": "🟢", "amber": "🟡", "red": "🔴", "": ""}

# (metric, column header, format)
L2_COLUMNS = [
    ("geh_lt5_share", "GEH<5", "{:.1%}"),
    ("volume_r2", "vol R²", "{:.4f}"),
    ("volume_abs_diff_rel", "vol Σ|Δ|", "{:.1%}"),
    ("link_time_mae_s", "link tt MAE", "{:.2f} s"),
    ("leg_tt_mean_rel_diff", "trip tt mean Δ", "{:+.2%}"),
    ("leg_tt_quantile_dist_s", "trip tt dist", "{:.1f} s"),
    ("leg_tt_abs_diff_p95_s", "trip |Δ| p95", "{:.0f} s"),
    ("veh_hours_rel_diff", "veh-h Δ", "{:+.2%}"),
    ("veh_km_rel_diff", "veh-km Δ", "{:+.2%}"),
    ("stuck_test", "stuck", "{:d}"),
]

LEGEND = """\
Metrics compare each mobslim mobsim (rows: `reference → test`) with a MATSim
mobsim on the same plans and routes (one iteration, no replanning).
`qsim → matsim_hermes` compares MATSim's two mobsims with each other: a yardstick
for how much two faithful queue simulations differ. All are computed from the
events of both sides by the same code (`verification/events_io.py`).

* **GEH<5**: share of (link, hour) volumes with GEH < 5. **vol R²**, **vol Σ|Δ|**:
  hourly link volumes, R² and total absolute difference over total volume.
* **link tt MAE**: mean absolute difference of mean link traversal time per
  (link, 15-min entry bin).
* **trip tt mean Δ**: relative difference of mean door-to-door trip time (a pt
  trip is several legs in MATSim, one in mobslim). **trip tt dist**: distance
  between the two trip time distributions (mean quantile gap).
* **trip |Δ| p95**: 95th percentile of per-trip time differences (trips
  matched by person and index). Large values with small distribution gaps mean
  agents queue in a different order, not that traffic moves differently.
* Transit (pt-tutorial): **pt trip Δ**; **stop arr/dep |Δ|**: mean difference of
  stop arrival and departure times per (departure, stop); **boardings Σ|Δ|**:
  boardings per (departure, stop), total absolute difference over total.
* **veh-h Δ**, **veh-km Δ**: network vehicle-hours and vehicle-km. **stuck**:
  mobslim agents stuck (MATSim's count is in the case header).

Bands (🟢 🟡 🔴) are first guesses, set in `verification/compare.py::BANDS`.
"""

KNOWN_DIFFERENCES = """\
Known differences that affect these numbers (see `VERIFICATION_PLAN.md` §3):

* **D4 departure buffer**: MATSim puts a departing car into the departure
  link's buffer, where it waits for that link's flow capacity; mobslim starts
  it at the link's to-node. Many agents leaving at the same time are released
  over several seconds in MATSim and at once in mobslim.
* **D5 time steps**: MATSim moves vehicles in 1 s steps; mobslim QSim and
  AgentSim are continuous, HermesSim uses whole seconds.
* **D8 transit**: mobslim simulates trips, not vehicles (no `Umlauf`), and in
  `schedule` mode runs them on the timetable rather than in traffic.
* **MATSim Hermes and transit**: on pt-tutorial, MATSim's Hermes runs a
  vehicle's departures route by route, not in time order (3→1 trips start up
  to 17 h late), so it is not used as a transit reference.
* **Tie order**: agents with equal departure times enter queues in a
  different order, so individual legs can differ by minutes while the
  aggregates agree.
"""


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
    except OSError:
        return "?"


def _cell(metric: str, fmt: str, value) -> str:
    if value is None:
        return ""
    text = fmt.format(int(value)) if fmt == "{:d}" else fmt.format(value)
    mark = _MARK[compare.band(metric, value)]
    return f"{text} {mark}".strip()


L2_TRANSIT_COLUMNS = [
    ("trip_tt_mean_rel_diff_pt", "pt trip Δ", "{:+.2%}"),
    ("stop_arrival_abs_diff_mean_s", "stop arr |Δ|", "{:.1f} s"),
    ("stop_departure_abs_diff_mean_s", "stop dep |Δ|", "{:.1f} s"),
    ("boardings_abs_diff_rel", "boardings Σ|Δ|", "{:.1%}"),
]

L1_COLUMNS = [
    ("persons_compared", "persons", "{:d}"),
    ("score_mean_ref", "mean (MATSim)", "{:.4f}"),
    ("score_mean_diff", "mean Δ", "{:+.2e}"),
    ("score_abs_diff_max", "max |Δ|", "{:.2e}"),
    ("share_within_1e-6", "|Δ|≤1e-6", "{:.1%}"),
    ("stuck", "stuck", "{:d}"),
]

L1_LEGEND = """\
Per-person scores of the same executed day, scored by both sides with the
case's MATSim config (translated by `mobslim.config.matsim`).

* **MATSim → mobslim**: MATSim's iteration-0 events, scored by mobslim's
  `CharyparNagelScorer` (`verification/experienced.py` builds the experienced
  plans), against MATSim's executed score in `output_plans`.
* **mobslim → MATSim**: mobslim's QSim run on the same plans, scored by
  mobslim's vectorised `score_trajectories`, against MATSim's
  `EventsToScore` on the same events (`verification/matsim_events.py`
  rewrites them into MATSim's trip structure; `java/ScoreEvents`).
"""


def _row(name, metrics, columns, extra=""):
    cells = [_cell(k, fmt, metrics.get(k)) for k, _, fmt in columns]
    return f"| {name} | " + " | ".join(cells) + f" |{extra}"


def _header(first, columns, extra=""):
    head = f"| {first} | " + " | ".join(h for _, h, _ in columns) + f" |{extra}"
    return [head, "|---" * (len(columns) + 1 + (1 if extra else 0)) + "|"]


def render(results: list[dict]) -> str:
    dirty = " (dirty)" if _git("status", "--porcelain") else ""
    lines = [
        "# MATSim verification report",
        "",
        f"* Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"* mobslim: `{_git('rev-parse', '--short', 'HEAD')}`{dirty}",
        f"* MATSim: {MATSIM_VERSION}",
        "",
    ]
    l1 = [r for r in results if r["layer"] == "l1"]
    if l1:
        lines += ["## Layer 1: scoring", "", L1_LEGEND]
        lines += _header("case / direction", L1_COLUMNS)
        for r in l1:
            lines.append(_row(f"{r['case']}: MATSim → mobslim", r["metrics"], L1_COLUMNS))
            if "reverse_metrics" in r:
                lines.append(_row(f"{r['case']}: mobslim → MATSim", r["reverse_metrics"], L1_COLUMNS))
        lines.append("")
        for r in l1:
            if r["unsupported"]:
                lines.append(f"* {r['case']}: config params mobslim does not honour: "
                             + ", ".join(f"`{u}`" for u in r["unsupported"]))
            for w in (r.get("worst") or [])[:3]:
                lines.append(f"* {r['case']}: person {w['person']}: MATSim {w['matsim']:.6f}, "
                             f"mobslim {w['mobslim']:.6f}")
        lines.append("")
    l2 = [r for r in results if r["layer"] == "l2"]
    if l2:
        lines += ["## Layer 2: mobsims on fixed plans", "", LEGEND, KNOWN_DIFFERENCES]
        for r in l2:
            comps = r["comparisons"]
            first = comps[0]["metrics"]
            transit = any("boardings_ref" in c["metrics"] for c in comps)
            cols = L2_COLUMNS + (L2_TRANSIT_COLUMNS if transit else [])
            lines += [
                f"### {r['case']}",
                "",
                f"{r['description']}. MATSim QSim: {first['legs_ref']} trips, "
                f"{first['traversals_ref']} car link traversals, {first['stuck_ref']} stuck, "
                f"mean trip time {first['leg_tt_mean_ref_s']:.0f} s"
                + (f", {first['boardings_ref']} boardings" if transit else "")
                + f", run {r['matsim_runtime_s']} s.",
                "",
            ]
            if r["unsupported"]:
                lines += ["Config params mobslim does not honour: "
                          + ", ".join(f"`{u}`" for u in r["unsupported"]), ""]
            lines += _header("MATSim → test", cols, " runtime |")
            for c in comps:
                lines.append(_row(f"{c['reference']} → {c['test']}", c["metrics"], cols,
                                  f" {c['runtime_s']} s |"))
            lines.append("")
    l3 = [r for r in results if r["layer"] == "l3"]
    if l3:
        lines += _render_l3(l3)
    return "\n".join(lines)


L3_LEGEND = """\
Both sides run the case's MATSim config for the whole loop, each with several
seeds (mobslim's strategies, scoring and mobsim are translated from the
config). Values are for the last iteration. The **MATSim band** is MATSim's
spread over its seeds: **z** is mobslim's mean minus MATSim's mean, in MATSim
standard deviations. 🟢 inside MATSim's min-max range, within 0.1% of its
mean, or |z| ≤ 2; 🟡 |z| ≤ 4; 🔴 beyond. Mode shares are per trip, a pt trip
routed as walk-only counting as `walk` (MATSim's convention). Link volumes: hourly volumes of the last iteration, mobslim seed k
against MATSim seed k, next to MATSim seed k against MATSim seed k+1 (the
floor: how much MATSim differs from itself).
"""

# colours: the reference palette's categorical slots 1 and 2 (dataviz skill)
_MATSIM_COLOR, _MOBSLIM_COLOR = "#2a78d6", "#eb6834"
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _l3_band(v: dict) -> str:
    z, t, m = v.get("z"), v["mobslim_mean"], v["matsim_mean"]
    if v["matsim_min"] - 1e-12 <= t <= v["matsim_max"] + 1e-12:
        return "green"
    if m and abs(t - m) <= 1e-3 * abs(m):  # within 0.1%: a tiny MATSim spread is no target
        return "green"
    if z is None or z != z:
        return "red" if abs(t - v["matsim_mean"]) > 1e-9 else "green"
    return "green" if abs(z) <= 2 else "amber" if abs(z) <= 4 else "red"


def _fmt(x: float) -> str:
    return f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"


def plot_l3(r: dict, path: Path) -> Path:
    """Per-iteration curves: MATSim's seed band and mean, mobslim's mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, t = r["curves"]["matsim"], r["curves"]["mobslim"]
    panels = [("score_executed", "Average executed score")]
    panels += [(k, f"Mode share: {k[6:]}") for k in m["mean"]
               if k.startswith("share_") and k != "share_car" and k in t["mean"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    for ax, (key, title) in zip(axes[0], panels):
        ax.set_facecolor(_SURFACE)
        it_m, it_t = m["mean"]["iteration"], t["mean"]["iteration"]
        ax.fill_between(it_m, m["min"][key], m["max"][key], color=_MATSIM_COLOR, alpha=0.18,
                        linewidth=0, label="MATSim seed range")
        ax.plot(it_m, m["mean"][key], color=_MATSIM_COLOR, linewidth=2, label="MATSim mean")
        ax.plot(it_t, t["mean"][key], color=_MOBSLIM_COLOR, linewidth=2, label="mobslim mean")
        for x, y, name, c in ((it_m[-1], m["mean"][key][-1], "MATSim", _MATSIM_COLOR),
                              (it_t[-1], t["mean"][key][-1], "mobslim", _MOBSLIM_COLOR)):
            ax.annotate(name, (x, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=_INK)
            ax.plot([x], [y], "o", color=c, markersize=4)
        ax.set_title(title, fontsize=10, color=_INK, loc="left")
        ax.set_xlabel("iteration", fontsize=8, color=_MUTED)
        ax.tick_params(colors=_MUTED, labelsize=8, length=0)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.margins(x=0.08)
    axes[0][0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=_SURFACE)
    plt.close(fig)
    return path


def _render_l3(results: list[dict]) -> list[str]:
    lines = ["## Layer 3: the full loop", "", L3_LEGEND]
    for r in results:
        png = plot_l3(r, REPORTS / f"{r['case']}.png")
        lines += [f"### {r['case']}", "",
                  f"{r['description']}. {r['seeds']} seeds per side, {r['iterations']} iterations; "
                  f"run time per seed: MATSim {sum(r['runtimes']['matsim']) / r['seeds']:.0f} s, "
                  f"mobslim {sum(r['runtimes']['mobslim']) / r['seeds']:.0f} s.", ""]
        if r["unsupported"]:
            lines += ["Config params mobslim does not honour: "
                      + ", ".join(f"`{u}`" for u in r["unsupported"]), ""]
        lines += [f"![{r['case']}]({png.name})", "",
                  "| last iteration | MATSim mean ± sd [min, max] | mobslim mean ± sd | Δ | z |",
                  "|---|---|---|---|---|"]
        for k, v in r["final"].items():
            if k == "trips" or (v["matsim_sd"] == 0 and v["mobslim_sd"] == 0
                                and v["matsim_mean"] == v["mobslim_mean"]):
                continue
            mark = _MARK[_l3_band(v)]
            z = "" if v["z"] != v["z"] else f"{v['z']:+.1f}"
            lines.append(
                f"| {k} | {_fmt(v['matsim_mean'])} ± {_fmt(v['matsim_sd'])} "
                f"[{_fmt(v['matsim_min'])}, {_fmt(v['matsim_max'])}] | "
                f"{_fmt(v['mobslim_mean'])} ± {_fmt(v['mobslim_sd'])} | "
                f"{v['mobslim_mean'] - v['matsim_mean']:+.4f} | {z} {mark} |")
        vol = r["volumes"]
        lines += ["", "| link volumes (last iteration) | GEH<5 | R² | Σ|Δ| / total |", "|---|---|---|---|"]
        for name, label in (("matsim_vs_matsim", "MATSim vs MATSim (other seed)"),
                            ("mobslim_vs_matsim", "mobslim vs MATSim")):
            if name in vol:
                v = vol[name]
                lines.append(f"| {label} | {v['geh_lt5_share']:.1%} | {v['volume_r2']:.4f} | "
                             f"{v['volume_abs_diff_rel']:.1%} |")
        lines.append("")
    return lines


def write(results: list[dict], path: Path | None = None) -> Path:
    path = path or REPORTS / "latest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(results))
    return path


def load_results(output_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(output_dir.glob("*/result.json"))]
