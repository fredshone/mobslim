"""MATSim's DSim, its parallel mobsim, which is only in MATSim's weekly builds.

QSim and Hermes run on the MATSim 2025.0 release, as in ``verification/``.
DSim runs from :data:`DSIM_MATSIM` (``java/dsim/pom.xml``) through
``java/dsim/DSimRun.java``, MATSim's own single-JVM recipe
(``RunDistributedSim``). It copies the ``qsim`` module into ``dsim`` (FIFO
links, queue dynamics, stuck time, start and end time), so DSim simulates
the same traffic model as the QSim.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).parent / "java" / "dsim"
CLASSES = Path(__file__).parent / "java" / "classes-dsim"
SOURCE = HERE / "DSimRun.java"
DSIM_MATSIM = "2027.0-2026w45"  # keep in sync with java/dsim/pom.xml
# the communication libraries DSim loads need this (MATSim's distributed-simulation pom)
JVM_FLAGS = ["--add-opens", "java.base/jdk.internal.misc=ALL-UNNAMED"]


def classpath() -> str:
    """Resolve the weekly build (Maven) and compile ``DSimRun`` if needed."""
    cp_file = HERE / "classpath.txt"
    pom = HERE / "pom.xml"
    if not cp_file.exists() or cp_file.stat().st_mtime < pom.stat().st_mtime:
        subprocess.run(["mvn", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={cp_file}"],
                       cwd=HERE, check=True, capture_output=True)
    cp = cp_file.read_text().strip()
    cls = CLASSES / "DSimRun.class"
    if not cls.exists() or cls.stat().st_mtime < SOURCE.stat().st_mtime:
        # Java 21 class files: Guice in this build cannot read newer ones
        subprocess.run(["javac", "--release", "21", "-nowarn", "-d", str(CLASSES), "-cp", cp,
                        str(SOURCE)], check=True, capture_output=True)
    return f"{CLASSES}:{cp}"


def command(config: Path, output: Path, threads: int, timings: Path | None = None,
            sets: dict | None = None, xmx: str = "4g") -> list[str]:
    """The ``java`` command line for one DSim run (events written unless ``timings``)."""
    cmd = ["java", f"-Xmx{xmx}", *JVM_FLAGS, "-cp", classpath(), "DSimRun",
           "--config", str(config), "--output", str(output),
           "--set", f"dsim.threads={threads}", "--set", "controller.compressionType=gzip"]
    if timings is not None:
        cmd += ["--timings", str(timings)]
    for k, v in (sets or {}).items():
        cmd += ["--set", f"{k}={v}"]
    return cmd


def run(config: Path, output: Path, threads: int = 1, force: bool = False) -> Path:
    """Run DSim on ``config`` writing events into ``output`` (skipped if they exist)."""
    events = output / "output_events.xml.gz"
    if events.exists() and not force:
        return output
    log = output.parent / f"{output.name}.stdout.log"
    with open(log, "w") as f:
        proc = subprocess.run(command(config, output.resolve(), threads), stdout=f,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not events.exists():
        raise RuntimeError(f"DSim failed (exit {proc.returncode}); see {log}")
    return output
