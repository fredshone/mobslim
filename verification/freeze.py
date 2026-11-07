"""Freeze MATSim references into ``tests/fixtures/matsim_reference/`` for pytest.

``python -m verification freeze`` copies, per frozen case, MATSim's
iteration-0 events and plans plus the inputs mobslim reads (config, network,
transit) into a self-contained directory, and records the metrics mobslim
reaches now (``expected.json``). ``tests/test_matsim_reference.py`` then
reruns layer 1 (MATSim's events scored by mobslim) and layer 2 (every mobslim
mobsim on MATSim's plans) without Java and fails if a metric gets worse than
frozen by more than its slack.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mobslim.config.matsim import MatsimConfig
from verification import cases, runner

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "matsim_reference"

# frozen case name -> (layer-2 case it takes MATSim's run from, layer-1 case or None)
FROZEN = {
    "equil-100": ("equil-100.l2", "equil-100.l1"),
    "equil-2000-cap0.5": ("equil-2000-cap0.5.l2", "equil-2000-cap0.5.l1"),
    "pt-tutorial": ("pt-tutorial.l2", "pt-tutorial.l1"),
}

# metric -> (absolute slack, higher is better); "worse" is judged on abs() otherwise
GUARDED = {
    "geh_lt5_share": (0.01, True),
    "volume_r2": (0.01, True),
    "volume_abs_diff_rel": (0.01, False),
    "link_time_mae_s": (0.5, False),
    "leg_tt_mean_rel_diff": (0.002, False),
    "leg_tt_quantile_dist_s": (2.0, False),
    "veh_hours_rel_diff": (0.005, False),
    "veh_km_rel_diff": (0.001, False),
    "stop_arrival_abs_diff_mean_s": (0.5, False),
    "stop_departure_abs_diff_mean_s": (0.5, False),
    "boardings_abs_diff_rel": (0.01, False),
    "trip_tt_mean_rel_diff_pt": (0.005, False),
}

GUARDED_MOBSIMS = ("qsim", "hermes", "parallel", "shard")

_INPUTS = [("network", "inputNetworkFile", "network.xml"),
           ("transit", "transitScheduleFile", "transitSchedule.xml"),
           ("transit", "vehiclesFile", "transitVehicles.xml")]


def freeze(names=None) -> list[Path]:
    written = []
    for name, (l2_name, l1_name) in FROZEN.items():
        if names and not any(n in name for n in names):
            continue
        l2 = next(c for c in cases.CASES if c.name == l2_name)
        result = runner.run_l2(l2)  # cached MATSim run; fresh mobslim metrics
        matsim_dir = runner.OUTPUT / l2.name / "matsim_qsim"
        out = FIXTURES / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        cfg = runner.case_config(l2)
        for module, param, filename in _INPUTS:
            src = cfg.input_path(module, param)
            if src is not None:
                shutil.copy(src, out / filename)
                cfg.set(module, param, filename)
        cfg.set("plans", "inputPlansFile", "output_plans.xml.gz")
        for f in ("output_events.xml.gz", "output_plans.xml.gz"):
            shutil.copy(matsim_dir / f, out / f)
        _write_relative(cfg, out / "config.xml")

        expected = {"case": name, "transit_walk": l2.transit_walk, "vehicles": list(l2.vehicles),
                    "l1": l1_name is not None, "l2": {}}
        for c in result["comparisons"]:  # the kernel sims (AgentSim: tested against the QSim)
            if c["reference"] == "qsim" and c["test"].split("/")[0] in GUARDED_MOBSIMS:
                expected["l2"][c["test"]] = {k: c["metrics"][k] for k in GUARDED if k in c["metrics"]}
        (out / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
        written.append(out)
    return written


def _write_relative(cfg: MatsimConfig, path: Path) -> None:
    """Like ``MatsimConfig.write`` but keeping the (relative) input paths."""
    import xml.etree.ElementTree as ET

    from mobslim.config.matsim import DOCTYPE

    root = cfg.copy().root
    ET.indent(root)
    path.write_text(f'<?xml version="1.0" ?>\n{DOCTYPE}\n{ET.tostring(root, encoding="unicode")}\n')
