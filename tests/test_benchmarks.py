"""Smoke tests: every benchmark still runs against the current API.

Timing is not checked here; run ``uv run python -m benchmarks`` for that.
"""

import pytest

from benchmarks import harness
from benchmarks.__main__ import main
from benchmarks.scenarios import SCENARIOS
from benchmarks.suite import BENCHMARKS, Workload, stage_breakdown


@pytest.fixture(scope="module")
def workload():
    return Workload(SCENARIOS["equil"])


@pytest.fixture(scope="module")
def transit_workload():
    return Workload(SCENARIOS["pt_tutorial"])


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_benchmark_runs(workload, transit_workload, name):
    """On equil; the transit benchmarks on pt_tutorial (equil has no schedule)."""
    bench = BENCHMARKS[name]
    try:
        fn = bench.setup(transit_workload if name.startswith("transit.") else workload)
    except harness.SkipBenchmark as skip:
        pytest.skip(str(skip))
    out = fn()
    if bench.info:
        assert isinstance(bench.info(out), dict)


def test_run_matches_controller_results(workload):
    run = BENCHMARKS["run.controller"]
    info = run.info(run.setup(workload)())
    assert info["iterations"] == SCENARIOS["equil"].iterations
    assert info["stuck"] == 0


def test_stage_breakdown_covers_the_run(workload):
    stages = stage_breakdown(workload, repeats=1)
    assert {"mobsim + event handlers", "replanning", "scoring"} <= stages.keys()
    assert all(v >= 0 for v in stages.values())


def test_measure_save_compare(workload, tmp_path):
    bench = BENCHMARKS["scoring.charypar_nagel"]
    result = harness.measure(bench, workload, budget=0.0, min_rounds=2)
    assert result.rounds >= 2 and result.min <= result.median

    path = tmp_path / "base.json"
    harness.save(path, harness.metadata("equil", 0.0), [result])
    text = harness.report([result], harness.load(path))
    assert "vs base" in text
    assert "CHANGED" not in text


def test_cli(tmp_path, capsys):
    main(["-k", "load.network", "--budget", "0", "--save", str(tmp_path / "r.json")])
    main(["-k", "load.network", "--budget", "0", "--compare", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "load.network" in out and "vs base" in out


def test_measure_memory_counts_peak_and_retained():
    def allocate():
        _scratch = bytearray(4 << 20)  # 4 MB, freed before returning
        return bytearray(1 << 20)  # 1 MB, kept by the caller

    peak, retained = harness.measure_memory(allocate)
    assert peak >= 5 << 20
    assert (1 << 20) <= retained < (2 << 20)


def test_memory_is_reported_saved_and_compared(workload, tmp_path):
    bench = BENCHMARKS["scoring.charypar_nagel"]
    result = harness.measure(bench, workload, budget=0.0, min_rounds=1)
    assert result.peak is not None and result.peak >= result.retained >= 0
    path = tmp_path / "base.json"
    harness.save(path, harness.metadata("equil", 0.0), [result])
    text = harness.report([result], harness.load(path))
    assert "peak mem" in text and "mem vs base" in text

    timing_only = harness.measure(bench, workload, budget=0.0, min_rounds=1, memory=False)
    assert timing_only.peak is None
    assert "peak mem" not in harness.report([timing_only])


def test_cli_profile_memory(capsys):
    main(["--profile-memory", "scoring.charypar_nagel", "--top", "3"])
    out = capsys.readouterr().out
    assert "peak" in out and "allocation sites" in out
