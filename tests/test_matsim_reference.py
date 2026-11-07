"""mobslim against frozen Java MATSim 2025.0 runs (no Java needed).

``tests/fixtures/matsim_reference/<case>/`` holds MATSim's iteration-0 events
and plans and the inputs mobslim reads; ``python -m verification freeze``
writes them (see ``verification/freeze.py`` and ``verification/README.md``).

* Layer 1: mobslim's scorer, on MATSim's events, gives MATSim's scores exactly.
* Layer 2: the QSim, HermesSim, PSim and DSim, on MATSim's plans and routes, are no further
  from MATSim than when frozen (link volumes, link and trip times, transit).
"""

import json

import pytest

from mobslim import read_network_xml
from mobslim.config.matsim import MatsimConfig, scoring_params
from mobslim.scoring import CharyparNagelScorer
from verification import compare, events_io, experienced, runner
from verification.cases import MOBSIMS
from verification.freeze import FIXTURES, GUARDED

CASES = sorted(p.name for p in FIXTURES.glob("*") if (p / "expected.json").exists())
if not CASES:  # pragma: no cover
    pytest.skip("no frozen MATSim references (python -m verification freeze)", allow_module_level=True)


def load(name):
    d = FIXTURES / name
    cfg = MatsimConfig.load(d / "config.xml")
    expected = json.loads((d / "expected.json").read_text())
    network = read_network_xml(cfg.input_path("network", "inputNetworkFile"))
    link_length = {str(i): float(x) for i, x in zip(network.link_ids, network.length)}
    events = events_io.read_events(d / "output_events.xml.gz")
    return d, cfg, expected, link_length, events


@pytest.mark.parametrize("name", CASES)
def test_scores_match_matsim(name):
    d, cfg, expected, link_length, events = load(name)
    if not expected["l1"]:
        pytest.skip("no layer-1 reference")
    params, unsupported = scoring_params(cfg)
    assert unsupported == []
    matsim, _ = runner.matsim_scores(d / "output_plans.xml.gz")
    modes = (cfg.get("transit", "transitModes") or "pt").split(",")
    plans = experienced.experienced_plans(events, link_length, modes)
    ours = experienced.score_plans(CharyparNagelScorer(params), plans)
    assert set(ours) == set(matsim)
    worst = max(abs(ours[p] - matsim[p]) for p in matsim)
    assert worst < 1e-9


def _runs(name):
    expected = json.loads((FIXTURES / name / "expected.json").read_text())
    return [(name, label) for label in expected["l2"]]


@pytest.mark.parametrize("name,label", [r for n in CASES for r in _runs(n)])
def test_mobsim_no_further_from_matsim(name, label, tmp_path):
    d, cfg, expected, link_length, events = load(name)
    mobsim, _, vehicles = label.partition("/")
    if mobsim not in MOBSIMS:
        pytest.skip(f"{mobsim} not available")
    scenario = runner.mobslim_scenario(cfg, d / "output_plans.xml.gz", vehicles or "network",
                                       transit_walk=expected["transit_walk"])
    path = tmp_path / "events.xml.gz"
    runner.run_mobslim(cfg, scenario, mobsim, path)
    metrics = compare.compare_runs(events, events_io.read_events(path), link_length)
    worse = []
    for key, frozen in expected["l2"][label].items():
        slack, higher = GUARDED[key]
        now = metrics[key]
        if (frozen - now if higher else abs(now) - abs(frozen)) > slack:
            worse.append(f"{key}: {now:.6g} (frozen {frozen:.6g}, slack {slack})")
    assert not worse, "; ".join(worse)
