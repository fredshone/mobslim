"""The ``mobslim`` command: ``run`` on both config kinds, with overrides."""

from pathlib import Path

from mobslim.cli import main

SCENARIOS = Path(__file__).parent.parent / "scenarios"


def test_run_mobslim_config(tmp_path, capsys):
    assert main(["run", str(SCENARIOS / "equil" / "mobslim.yaml"), "-n", "2",
                 "-o", str(tmp_path), "-q"]) == 0
    assert "done: 2 iterations" in capsys.readouterr().out
    assert (tmp_path / "metrics.csv").exists()
    assert "iterations: 2" in (tmp_path / "output_config.yaml").read_text()


def test_run_matsim_config(tmp_path, capsys):
    assert main(["run", str(SCENARIOS / "pt-simple" / "config.xml"), "-n", "1",
                 "-o", str(tmp_path), "-q"]) == 0
    assert "done: 1 iteration," in capsys.readouterr().out
    assert (tmp_path / "metrics.csv").exists()


def test_check(capsys):
    assert main(["check", str(SCENARIOS / "equil" / "mobslim.yaml")]) == 0
    assert "OK" in capsys.readouterr().out
