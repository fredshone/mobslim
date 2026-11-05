"""Command line for the benchmark suite. See ``benchmarks/README.md``.

    uv run python -m benchmarks                    # run everything
    uv run python -m benchmarks -k mobsim -k run   # only matching benchmarks
    uv run python -m benchmarks --save base        # save as a baseline
    uv run python -m benchmarks --compare base     # compare with a baseline
    uv run python -m benchmarks --profile mobsim.qsim
    uv run python -m benchmarks --profile-memory routing.dijkstra
    uv run python -m benchmarks --no-memory        # timings only (faster)
"""

import argparse
import cProfile
import linecache
import pstats
import sys
import time
from pathlib import Path

from benchmarks import harness
from benchmarks.scenarios import SCENARIOS
from benchmarks.suite import BENCHMARKS, Workload, stage_breakdown


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-k", dest="patterns", action="append", default=[],
                   help="run benchmarks whose name contains this text (repeatable)")
    p.add_argument("-s", "--scenario", default="equil", choices=sorted(SCENARIOS))
    p.add_argument("--budget", type=float, default=0.5,
                   help="seconds of timing per benchmark (default 0.5)")
    p.add_argument("--min-rounds", type=int, default=3)
    p.add_argument("--save", metavar="NAME",
                   help="save results to benchmarks/results/NAME.json (or a .json path)")
    p.add_argument("--compare", metavar="NAME", help="compare with saved results")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="relative change marked ▲/▼ in a comparison (default 0.1)")
    p.add_argument("--profile", metavar="BENCHMARK",
                   help="profile one benchmark with cProfile instead of timing")
    p.add_argument("--profile-memory", metavar="BENCHMARK",
                   help="show where one benchmark allocates memory (tracemalloc)")
    p.add_argument("--memory-by", choices=("lineno", "filename", "traceback"),
                   default="lineno", help="grouping for --profile-memory (default lineno)")
    p.add_argument("--no-memory", dest="memory", action="store_false",
                   help="skip the traced memory call per benchmark")
    p.add_argument("--top", type=int, default=25,
                   help="rows shown by --profile / --profile-memory")
    p.add_argument("--list", action="store_true", help="list benchmarks and exit")
    return p.parse_args(argv)


def progress(text: str) -> None:
    """One overwritten status line, only on a terminal."""
    if sys.stdout.isatty():
        print(f"\r\033[K{text}", end="", flush=True)


def select(patterns: list[str]) -> list:
    benches = [b for name, b in BENCHMARKS.items()
               if not patterns or any(p in name for p in patterns)]
    if not benches:
        sys.exit(f"No benchmark matches {patterns}. Use --list.")
    return benches


def profile(name: str, workload: Workload, top: int) -> None:
    if name not in BENCHMARKS:
        sys.exit(f"Unknown benchmark {name!r}. Use --list.")
    fn = BENCHMARKS[name].setup(workload)
    fn()  # warm-up, so one-off caches do not show up
    profiler = cProfile.Profile()
    profiler.enable()
    deadline = time.perf_counter() + 1.0
    calls = 0
    while calls == 0 or time.perf_counter() < deadline:
        fn()
        calls += 1
    profiler.disable()
    print(f"{name}: {calls} calls\n")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(top)


def profile_memory(name: str, workload: Workload, top: int, key: str) -> None:
    if name not in BENCHMARKS:
        sys.exit(f"Unknown benchmark {name!r}. Use --list.")
    fn = BENCHMARKS[name].setup(workload)
    fn()  # warm-up, so one-off caches do not show up
    peak, retained, stats = harness.allocation_sites(fn, top, key)
    print(f"{name}: peak {harness.fmt_bytes(peak).strip()}, "
          f"retained {harness.fmt_bytes(retained).strip()}\n")
    print(f"top {len(stats)} allocation sites still alive after the call (by {key}):")
    harness_dir = str(Path(__file__).resolve().parent)
    for st in stats:
        # most recent frame first, without the benchmark machinery around it
        frames = [f for f in reversed(st.traceback)
                  if not f.filename.startswith((harness_dir, "<frozen"))] or [st.traceback[0]]
        if key != "traceback":
            frames = frames[:1]
        first, *rest = frames[:8]
        print(f"  {harness.fmt_bytes(st.size_diff)}  {st.count_diff:>8} blocks  "
              f"{first.filename}:{first.lineno}")
        if key == "traceback":
            code = linecache.getline(first.filename, first.lineno).strip()
            if code:
                print(f"  {'':>10}  {'':>8}         {code}")
            for f in rest:
                print(f"  {'':>10}  {'':>8}         called from {f.filename}:{f.lineno}")
    print("\nPeak counts temporaries too; the list shows only what survives the call.")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.list:
        for b in BENCHMARKS.values():
            print(f"{b.name:<32} {b.description.splitlines()[0]}")
        return

    spec = SCENARIOS[args.scenario]
    workload = Workload(spec)
    if args.profile:
        profile(args.profile, workload, args.top)
        return
    if args.profile_memory:
        profile_memory(args.profile_memory, workload, args.top, args.memory_by)
        return

    baseline = harness.load(harness.results_path(args.compare)) if args.compare else None
    if baseline and baseline["meta"]["scenario"] != spec.name:
        print(f"warning: baseline is for scenario {baseline['meta']['scenario']!r}")

    print(f"scenario: {spec.name} ({spec.description})")
    start = time.perf_counter()
    results = []
    benches = select(args.patterns)
    for bench in benches:
        progress(f"  {bench.name} ...")
        try:
            results.append(harness.measure(bench, workload, args.budget, args.min_rounds,
                                           memory=args.memory))
        except harness.SkipBenchmark as skip:
            progress("")
            print(f"skipped {bench.name}: {skip}")
    stages = None
    if any(b.name == "run.controller" for b in benches):
        progress("  stage breakdown ...")
        stages = stage_breakdown(workload)
    progress("")
    print(f"ran {len(results)} benchmarks in {time.perf_counter() - start:.1f} s\n")
    print(harness.report(results, baseline, args.threshold))
    if stages:
        print(f"\nrun.controller by stage ({spec.iterations} iterations, best of 3):")
        print(harness.report_stages(stages))
    rss = harness.max_rss()
    if rss is not None:
        print(f"\nprocess peak RSS: {harness.fmt_bytes(rss).strip()}")

    if args.save:
        path = harness.results_path(args.save)
        harness.save(path, harness.metadata(spec.name, args.budget), results)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
