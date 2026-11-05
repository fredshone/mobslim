"""Timing, memory, reporting, saving and comparing benchmark results.

Nothing here knows about MobSlim. A benchmark is a ``setup(workload)``
function that does the untimed preparation and returns the zero-argument
callable to time.

Memory is measured with :mod:`tracemalloc` in one extra call made *after*
timing, because tracing slows Python code down several times. It counts
Python and numpy allocations (numpy reports to tracemalloc), not memory
held by other C extensions.
"""

import gc
import json
import platform
import statistics
import subprocess
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Fast calls are repeated in an inner loop so one sample lasts at least this
# long; otherwise timer resolution and call overhead dominate.
MIN_SAMPLE_TIME = 0.005


class SkipBenchmark(Exception):
    """Raised by a benchmark's setup when it cannot run here (e.g. an
    optional compiled extension is not built)."""


@dataclass(frozen=True)
class Benchmark:
    name: str  # "<group>.<what>", e.g. "mobsim.qsim"
    description: str
    setup: Callable[[Any], Callable[[], Any]]
    # Turns the timed callable's return value into a few numbers that pin
    # down behaviour (event counts, scores, ...). A change in these numbers
    # means the design change also changed the results.
    info: Optional[Callable[[Any], dict]] = None

    @property
    def group(self) -> str:
        return self.name.split(".", 1)[0]


@dataclass
class Result:
    name: str
    rounds: int  # number of samples
    number: int  # calls per sample
    min: float  # seconds per call
    median: float
    mean: float
    stdev: float
    info: dict = field(default_factory=dict)
    # bytes, from one traced call (None: memory was not measured)
    peak: Optional[int] = None  # most memory in use during the call
    retained: Optional[int] = None  # still allocated after it, return value included

    @classmethod
    def from_samples(cls, name, samples, number, info, memory=(None, None)):
        return cls(
            name, len(samples), number, min(samples),
            statistics.median(samples), statistics.fmean(samples),
            statistics.stdev(samples) if len(samples) > 1 else 0.0, info,
            *memory,
        )


def _calibrate(fn: Callable[[], Any]) -> tuple[int, Any]:
    """Run once (warm-up) and pick how many calls one sample needs."""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    number = 1 if dt >= MIN_SAMPLE_TIME else int(MIN_SAMPLE_TIME / max(dt, 1e-7)) + 1
    return number, out


def measure_memory(fn: Callable[[], Any]) -> tuple[int, int]:
    """``(peak, retained)`` bytes allocated by one call of ``fn``.

    ``peak`` is the high-water mark above what was allocated before the call;
    ``retained`` is what is still allocated afterwards while the return value
    is alive (the result itself plus anything the call cached). Both count
    only allocations made during the call.
    """
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()
        out = fn()
        after, peak = tracemalloc.get_traced_memory()
        del out
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return max(peak - before, 0), max(after - before, 0)


def measure(bench: Benchmark, workload, budget: float = 1.0,
            min_rounds: int = 3, max_rounds: int = 1000,
            memory: bool = True) -> Result:
    """Time one benchmark, then (optionally) measure its memory.

    Collects samples until ``budget`` seconds are spent and at least
    ``min_rounds`` samples exist. Setup and warm-up are not counted. Memory
    comes from one more, traced call, so lazily built caches that the
    warm-up filled are not counted again.
    """
    fn = bench.setup(workload)
    gc.collect()
    number, out = _calibrate(fn)
    info = bench.info(out) if bench.info else {}
    samples: list[float] = []
    start = time.perf_counter()
    while len(samples) < max_rounds:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        samples.append((time.perf_counter() - t0) / number)
        if len(samples) >= min_rounds and time.perf_counter() - start >= budget:
            break
    mem = measure_memory(fn) if memory else (None, None)
    return Result.from_samples(bench.name, samples, number, info, mem)


def allocation_sites(fn: Callable[[], Any], top: int = 25,
                     key: str = "lineno") -> tuple[int, int, list]:
    """Run ``fn`` once under tracemalloc and return ``(peak, retained, stats)``.

    ``stats`` are the ``top`` :class:`tracemalloc.StatisticDiff` entries
    (grouped by ``key``: ``"lineno"``, ``"filename"`` or ``"traceback"``)
    for memory still allocated after the call, largest first.
    """
    gc.collect()
    tracemalloc.start(25 if key == "traceback" else 1)
    try:
        before = tracemalloc.take_snapshot()
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        out = fn()
        after_bytes, peak = tracemalloc.get_traced_memory()
        after = tracemalloc.take_snapshot()
        del out
    finally:
        tracemalloc.stop()
    ignore = [tracemalloc.Filter(False, tracemalloc.__file__),
              tracemalloc.Filter(False, "<frozen importlib._bootstrap*>")]
    stats = after.filter_traces(ignore).compare_to(before.filter_traces(ignore), key)
    stats = [st for st in stats if st.size_diff > 0][:top]
    return max(peak - base, 0), max(after_bytes - base, 0), stats


# ---------------------------------------------------------------- reporting


def fmt_time(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:7.2f} {unit:2}"
    return f"{seconds / 1e-9:7.0f} ns"


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:7.2f} {unit}"
    return f"{n:7d} B "


def fmt_info(info: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in info.items())


def report(results: list[Result], baseline: Optional[dict] = None,
           threshold: float = 0.1) -> str:
    """A text table. With a baseline, adds the change in ``min`` time and,
    when memory was measured, in ``peak`` memory."""
    width = max([len(r.name) for r in results] + [9])
    memory = any(r.peak is not None for r in results)
    head = f"{'benchmark':<{width}}  {'min':>10}  {'median':>10}  {'±stdev':>9}  {'rounds':>6}"
    if baseline is not None:
        head += f"  {'vs base':>10}"
    if memory:
        head += f"  {'peak mem':>10}  {'retained':>10}"
        if baseline is not None:
            head += f"  {'mem vs base':>11}"
    lines = [head, "-" * len(head)]
    base = (baseline or {}).get("results", {})
    for r in results:
        rel = r.stdev / r.median if r.median else 0.0
        old = base.get(r.name)
        line = (f"{r.name:<{width}}  {fmt_time(r.min):>10}  "
                f"{fmt_time(r.median):>10}  {rel:>8.1%}  {r.rounds:>6}")
        if baseline is not None:
            line += f"  {_change(r.min, old and old.get('min'), threshold):>10}"
        if memory:
            line += f"  {fmt_bytes(r.peak):>10}  {fmt_bytes(r.retained):>10}"
            if baseline is not None:
                line += f"  {_change(r.peak, old and old.get('peak'), threshold):>11}"
        lines.append(line)
    lines.extend(_info_lines(results, base if baseline is not None else None))
    return "\n".join(lines)


def report_stages(stages: dict[str, float]) -> str:
    total = sum(stages.values())
    width = max(len(k) for k in stages)
    lines = [
        f"  {k:<{width}}  {fmt_time(v):>10}  {v / total:6.1%}"
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1])
    ]
    lines.append(f"  {'total':<{width}}  {fmt_time(total):>10}")
    return "\n".join(lines)


def _change(new: Optional[float], old: Optional[float], threshold: float) -> str:
    """``new / old`` with ▼ (less) / ▲ (more) beyond ``threshold``."""
    if new is None:
        return "-"
    if old is None:
        return "new"
    if old == 0:
        return "   same  " if new == 0 else "  from 0 ▲"
    ratio = new / old
    tag = "  " if abs(ratio - 1) < threshold else (" ▲" if ratio > 1 else " ▼")
    return f"{ratio:6.2f}x{tag}"


def _info_lines(results, base: Optional[dict]) -> list[str]:
    lines = ["", "results (should only change when behaviour changes):"]
    for r in results:
        if not r.info:
            continue
        lines.append(f"  {r.name}: {fmt_info(r.info)}")
        old = (base or {}).get(r.name, {}).get("info")
        if old is not None and old != r.info:
            lines.append(f"  {'':{len(r.name)}}  CHANGED, was: {fmt_info(old)}")
    return lines


# ---------------------------------------------------------- save and load


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def metadata(scenario: str, budget: float) -> dict:
    return {
        "scenario": scenario,
        "budget": budget,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()} {platform.node()}",
        "max_rss": max_rss(),
    }


def max_rss() -> Optional[int]:
    """Peak resident memory of this process so far, in bytes (Unix only)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if platform.system() == "Darwin" else rss * 1024  # Linux: kB


def results_path(name: str) -> Path:
    path = Path(name)
    return path if path.suffix == ".json" else RESULTS_DIR / f"{name}.json"


def save(path: Path, meta: dict, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "results": {r.name: asdict(r) for r in results}}
    path.write_text(json.dumps(data, indent=2) + "\n")


def load(path: Path) -> dict:
    return json.loads(path.read_text())
