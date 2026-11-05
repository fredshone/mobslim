"""Build and run the Java MATSim side (pinned to ``MATSIM_VERSION``).

Runs are cached: an output directory is reused when its ``manifest.json``
records the same key (MATSim version, config text, overrides, runner source).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from mobslim.config.matsim import MatsimConfig

MATSIM_VERSION = "2025.0"  # keep in sync with java/pom.xml
JAVA_DIR = Path(__file__).parent / "java"
_SOURCES = sorted((JAVA_DIR / "src").rglob("*.java")) + [JAVA_DIR / "pom.xml"]


class MatsimError(RuntimeError):
    pass


def _source_hash() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def classpath() -> str:
    """Compile the runner if its sources changed; return the Java classpath."""
    target = JAVA_DIR / "target"
    stamp = target / "source.hash"
    cp_file = target / "classpath.txt"
    if not (stamp.exists() and cp_file.exists() and stamp.read_text() == _source_hash()):
        if shutil.which("mvn") is None:
            raise MatsimError("Maven (mvn) is needed to build the MATSim runner.")
        proc = subprocess.run(
            ["mvn", "-q", "package", "dependency:build-classpath",
             f"-Dmdep.outputFile={cp_file}"],
            cwd=JAVA_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            raise MatsimError(f"Maven build failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
        stamp.write_text(_source_hash())
    return f"{target / 'classes'}:{cp_file.read_text().strip()}"


def run_matsim(config: MatsimConfig, output_dir, overrides: dict | None = None,
               force: bool = False) -> Path:
    """Run MATSim on ``config`` into ``output_dir`` (cached). Returns ``output_dir``."""
    output_dir = Path(output_dir).resolve()
    overrides = {k: str(v) for k, v in sorted((overrides or {}).items())}
    config_path = config.write(output_dir.parent / f"{output_dir.name}.config.xml")
    key = hashlib.sha256(json.dumps(
        [MATSIM_VERSION, config_path.read_text(), overrides, _source_hash()]).encode()).hexdigest()[:16]
    manifest = output_dir / "manifest.json"
    if not force and manifest.exists() and json.loads(manifest.read_text()).get("key") == key:
        return output_dir

    cmd = ["java", "-Xmx4g", "-cp", classpath(), "mobslim.verification.RunCase",
           "--config", str(config_path), "--output", str(output_dir)]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    t0 = time.perf_counter()
    log = output_dir.parent / f"{output_dir.name}.stdout.log"
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not (output_dir / "output_events.xml.gz").exists():
        raise MatsimError(f"MATSim run failed (exit {proc.returncode}); see {log}")
    manifest.write_text(json.dumps({
        "key": key,
        "matsim_version": MATSIM_VERSION,
        "config": str(config_path),
        "overrides": overrides,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }, indent=2) + "\n")
    return output_dir


def score_events(config: MatsimConfig, plans, events, output) -> dict[str, float]:
    """Score ``events`` with MATSim's Charypar-Nagel scoring (``ScoreEvents``)."""
    output = Path(output).resolve()
    config_path = config.write(output.with_suffix(".config.xml"))
    log = output.with_suffix(".log")
    with open(log, "w") as f:
        proc = subprocess.run(
            ["java", "-Xmx4g", "-cp", classpath(), "mobslim.verification.ScoreEvents",
             "--config", str(config_path), "--plans", str(Path(plans).resolve()),
             "--events", str(Path(events).resolve()), "--output", str(output)],
            stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise MatsimError(f"ScoreEvents failed (exit {proc.returncode}); see {log}")
    scores = {}
    for line in output.read_text().splitlines()[1:]:
        pid, _, score = line.partition(",")
        scores[pid] = float(score) if score else float("nan")
    return scores
