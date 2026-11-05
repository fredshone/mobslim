import csv
import gzip

import numpy as np
import pytest

from mobslim import (
    Controller,
    ControllerConfig,
    ControllerListener,
    EventRecorder,
    Scenario,
    StrategyManager,
    grid_network,
)
from mobslim.events import LinkEnter
from mobslim.replanning import ProcessPoolRunner, SerialRunner, VectorRunner
from mobslim.scoring import apply_scores
from tests.conftest import commuters


def test_equil_learns(equil_scenario):
    result = Controller(equil_scenario, config=ControllerConfig(iterations=15)).run()
    m = result.metrics
    assert len(m) == 15
    assert m[-1]["score_best"] > m[0]["score_executed"]
    assert m[-1]["car_travel_time"] < m[0]["car_travel_time"]
    assert all(row["stuck"] == 0 for row in m)
    assert all(len(p.plans) <= 5 for p in result.scenario.population)
    assert not np.isnan(result.scenario.population.selected_scores()).any()


def test_grid_spreads_traffic():
    net = grid_network(size=4, capacity=360)
    scenario = Scenario(net, commuters(30, net.node_index((0, 0)), net.node_index((4, 4))))
    strategies = StrategyManager.default()
    result = Controller(scenario, strategies=strategies, config=ControllerConfig(iterations=10)).run()
    assert result.metrics[-1]["car_travel_time"] < result.metrics[0]["car_travel_time"]


def test_runs_are_reproducible(equil_scenario):
    a = Controller(equil_scenario, config=ControllerConfig(iterations=4, seed=1)).run()
    b = Controller(equil_scenario, config=ControllerConfig(iterations=4, seed=1)).run()
    assert a.metrics == b.metrics


def test_parallel_replanning_is_identical(equil_scenario):
    cfg = ControllerConfig(iterations=3, seed=7)
    serial = Controller(equil_scenario, runner=SerialRunner(), config=cfg).run()
    parallel = Controller(equil_scenario, runner=ProcessPoolRunner(workers=2), config=cfg).run()
    assert serial.metrics == parallel.metrics


def test_vectorised_replanning_is_the_default_and_deterministic(equil_scenario):
    cfg = ControllerConfig(iterations=4, seed=7)
    controller = Controller(equil_scenario, config=cfg)
    assert isinstance(controller.runner, VectorRunner)
    a, b = controller.run(), Controller(equil_scenario, config=cfg).run()
    assert a.metrics == b.metrics
    other = Controller(equil_scenario, config=ControllerConfig(iterations=4, seed=8)).run()
    assert other.metrics != a.metrics


def test_listener_hook_order(equil_scenario):
    calls = []

    class Spy(ControllerListener):
        def __getattribute__(self, name):
            if name.startswith("on_"):
                return lambda arg: calls.append(name)
            return super().__getattribute__(name)

    Controller(equil_scenario, listeners=[Spy()], config=ControllerConfig(iterations=2)).run()
    assert calls == [
        "on_startup",
        "on_iteration_start", "on_before_mobsim", "on_after_mobsim", "on_scoring", "on_iteration_end",
        "on_iteration_start", "on_after_replanning", "on_before_mobsim", "on_after_mobsim",
        "on_scoring", "on_iteration_end",
        "on_shutdown",
    ]


def test_event_recorder_keeps_last_iteration(equil_scenario):
    recorder = EventRecorder()
    Controller(equil_scenario, listeners=[recorder], config=ControllerConfig(iterations=3)).run()
    assert list(recorder.events) == [2]
    assert any(isinstance(e, LinkEnter) for e in recorder.last)


def test_output_files(equil_scenario, tmp_path):
    cfg = ControllerConfig(iterations=3, output_dir=tmp_path, write_events_interval=2)
    Controller(equil_scenario, config=cfg).run()
    rows = list(csv.DictReader(open(tmp_path / "metrics.csv")))
    assert [r["iteration"] for r in rows] == ["0", "1", "2"]
    written = sorted(p.parent.name for p in tmp_path.glob("ITERS/*/events.xml.gz"))
    assert written == ["it.0", "it.2"]
    assert "</events>" in gzip.open(tmp_path / "ITERS" / "it.2" / "events.xml.gz", "rt").read()


def test_unknown_mode_is_rejected(equil_scenario):
    net = equil_scenario.network
    scenario = Scenario(net, commuters(1, 0, 1, mode="hoverboard"))
    with pytest.raises(ValueError, match="hoverboard"):
        Controller(scenario, config=ControllerConfig(iterations=1)).run()


def test_apply_scores_learning_rate(equil_scenario):
    pop = equil_scenario.population
    once = apply_scores(pop, [10.0] * len(pop), learning_rate=0.5)
    twice = apply_scores(once, [20.0] * len(pop), learning_rate=0.5)
    assert once[0].selected_score == 10.0
    assert twice[0].selected_score == 15.0
    assert pop[0].selected_score is None  # pure
