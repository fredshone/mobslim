"""Run a case on both sides and compare: layer 1 (scoring), 2 (mobsims), 3 (loop)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mobslim import (
    Scenario,
    read_network_xml,
    read_population_xml,
    route_scenario,
)
from mobslim.config.matsim import (
    MatsimConfig,
    mobsim_params,
    scoring_params,
    set_scoring_param,
)
from mobslim.events import EventsManager, EventsXMLWriter
from verification import compare, events_io
from verification.cases import MOBSIMS, Case
from verification.matsim import MATSIM_VERSION, run_matsim, score_events

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"  # git-ignored raw outputs


def case_config(case: Case) -> MatsimConfig:
    cfg = MatsimConfig.load(case.config_path)
    for (module, param), value in case.params.items():
        cfg.set(module, param, value)
    for (set_type, key), value in case.scoring_sets.items():
        set_scoring_param(cfg, set_type, key, value)
    return cfg


def mobslim_scenario(cfg: MatsimConfig, plans_path: Path, vehicles: str = "network",
                     keep_pt_routes: bool = True, transit_walk=None) -> Scenario:
    """The case's scenario on mobslim's side: MATSim's network, ``plans_path``
    with its car routes kept, and transit if the config uses it, with
    MATSim's pt routes unless ``keep_pt_routes=False`` (then mobslim's RAPTOR
    routes them). Legs without a route are routed by mobslim."""
    network = read_network_xml(cfg.input_path("network", "inputNetworkFile"))
    population = read_population_xml(plans_path, network, keep_routes=True)
    transit = None
    if cfg.get("transit", "useTransit") == "true":
        from mobslim.transit import TransitConfig, read_transit
        from verification.pt_routes import apply_matsim_pt_routes

        modes = tuple((cfg.get("transit", "transitModes") or "pt").split(","))
        from mobslim.routing.teleport import TeleportedModeParams

        tc = dict(passenger_modes=modes, vehicles=vehicles)
        if transit_walk is not None:
            tc["walk"] = TeleportedModeParams(*transit_walk)
        transit = read_transit(cfg.input_path("transit", "transitScheduleFile"),
                               cfg.input_path("transit", "vehiclesFile"), network, TransitConfig(**tc))
        if keep_pt_routes:
            population = apply_matsim_pt_routes(population, plans_path, transit.schedule, modes)
    return route_scenario(Scenario(network, population, transit=transit))


def run_mobslim(cfg: MatsimConfig, scenario: Scenario, mobsim: str, events_path: Path):
    """One mobsim run with events written to ``events_path``; returns
    (trajectories, runtime in s)."""
    kwargs, _ = mobsim_params(cfg)
    sim = MOBSIMS[mobsim](**kwargs)
    events = EventsManager([EventsXMLWriter(events_path, scenario.network, scenario.population,
                                            scenario.transit, vehicle_per_trip=True)])
    t0 = time.perf_counter()
    events.reset(0)
    tr = sim.run(scenario, events)
    events.finish()
    return tr, round(time.perf_counter() - t0, 2)


def _hermes_config(cfg: MatsimConfig) -> MatsimConfig:
    """``cfg`` run on MATSim's Hermes, with the qsim's capacities and stuck time."""
    h = cfg.copy().set("controller", "mobsim", "hermes")
    for param in ("flowCapacityFactor", "storageCapacityFactor", "stuckTime", "endTime"):
        v = cfg.get("qsim", param)
        if v is not None and not (param == "endTime" and v in ("00:00:00", "")):
            h.set("hermes", param, int(float(v)) if param == "stuckTime" else v)
    if cfg.get("transit", "useTransit") == "true":
        h.set("hermes", "mainMode", "car")
    return h


def run_l2(case: Case, mobsims=None, force_matsim: bool = False) -> dict:
    """MATSim (each of ``case.references``) for one iteration; then every
    mobslim mobsim on MATSim's iteration-0 plans and routes."""
    out = OUTPUT / case.name
    cfg = case_config(case)
    _, unsupported = mobsim_params(cfg)
    network = read_network_xml(cfg.input_path("network", "inputNetworkFile"))
    link_length = {str(i): float(x) for i, x in zip(network.link_ids, network.length)}

    refs, ref_events = {}, {}
    for ref in case.references:
        ref_cfg = cfg if ref == "qsim" else _hermes_config(cfg)
        refs[ref] = run_matsim(ref_cfg, out / f"matsim_{ref}", force=force_matsim)
        ref_events[ref] = events_io.read_events(refs[ref] / "output_events.xml.gz")
    base = case.references[0]
    plans_path = refs[base] / "output_plans.xml.gz"

    comparisons = []
    for ref in case.references[1:]:  # MATSim's own mobsims against each other
        comparisons.append({"reference": base, "test": f"matsim_{ref}", "runtime_s": json.loads(
            (refs[ref] / "manifest.json").read_text()).get("runtime_s"),
            "metrics": compare.compare_runs(ref_events[base], ref_events[ref], link_length)})
    has_transit = cfg.get("transit", "useTransit") == "true"
    for vehicles in (case.vehicles if has_transit else ("network",)):
        scenario = mobslim_scenario(cfg, plans_path, vehicles, transit_walk=case.transit_walk)
        for name in mobsims or MOBSIMS:
            label = f"{name}/{vehicles}" if has_transit else name
            events_path = out / f"mobslim_{label.replace('/', '_')}.events.xml.gz"
            _, runtime = run_mobslim(cfg, scenario, name, events_path)
            test_events = events_io.read_events(events_path)
            for ref in case.references:
                comparisons.append({"reference": ref, "test": label, "runtime_s": runtime,
                                    "metrics": compare.compare_runs(ref_events[ref], test_events,
                                                                    link_length)})
    result = {
        "case": case.name,
        "layer": case.layer,
        "description": case.description,
        "matsim_version": MATSIM_VERSION,
        "matsim_runtime_s": json.loads((refs[base] / "manifest.json").read_text()).get("runtime_s"),
        "unsupported": unsupported,
        "comparisons": comparisons,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ------------------------------------------------------------------ layer 1


def matsim_scores(plans_path: Path) -> tuple[dict, dict]:
    """Selected-plan score and MATSim's ``scoreExplanation`` (if written) per person."""
    import gzip
    import xml.etree.ElementTree as ET

    scores, explanations = {}, {}
    with gzip.open(plans_path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "person":
                continue
            for plan in el.findall("plan"):
                if plan.get("selected") == "yes":
                    scores[el.get("id")] = float(plan.get("score"))
                    for a in plan.iter("attribute"):
                        if a.get("name") == "scoreExplanation":
                            explanations[el.get("id")] = (a.text or "").strip()
            el.clear()
    return scores, explanations


def run_l1(case: Case, force_matsim: bool = False) -> dict:
    """Score MATSim's iteration-0 events with mobslim; compare per person."""
    import numpy as np

    from mobslim.scoring import CharyparNagelScorer
    from verification import experienced, matsim_events

    out = OUTPUT / case.name
    cfg = case_config(case)
    params, unsupported = scoring_params(cfg)
    matsim_dir = run_matsim(cfg, out / "matsim", force=force_matsim)
    manifest = json.loads((matsim_dir / "manifest.json").read_text())
    ref, explanations = matsim_scores(matsim_dir / "output_plans.xml.gz")

    network = read_network_xml(cfg.input_path("network", "inputNetworkFile"))
    link_length = {str(i): float(x) for i, x in zip(network.link_ids, network.length)}
    ev = events_io.read_events(matsim_dir / "output_events.xml.gz")
    transit_modes = (cfg.get("transit", "transitModes") or "pt").split(",")
    plans = experienced.experienced_plans(ev, link_length, transit_modes)
    scorer = CharyparNagelScorer(params)
    test = experienced.score_plans(scorer, plans)
    metrics, worst = compare.compare_scores(ref, test, {p for p, pl in plans.items() if pl.stuck})

    # the other way round: mobslim's run, scored by mobslim (vectorised) and by MATSim
    plans_path = matsim_dir / "output_plans.xml.gz"
    scenario = mobslim_scenario(cfg, plans_path, transit_walk=case.transit_walk)
    tr, _ = run_mobslim(cfg, scenario, "qsim", out / "mobslim_qsim.events.xml.gz")
    ids = [str(i) for i in scenario.population.ids]
    own = dict(zip(ids, scorer.score_trajectories(tr).tolist()))
    adapted = matsim_events.to_matsim(out / "mobslim_qsim.events.xml.gz", plans_path,
                                      out / "mobslim_qsim.matsim_events.xml.gz",
                                      transit_modes=transit_modes)
    by_matsim = score_events(cfg, plans_path, adapted, out / "mobslim_qsim.matsim_scores.csv")
    rev_metrics, rev_worst = compare.compare_scores(
        by_matsim, own, {i for i, s in zip(ids, np.asarray(tr.stuck)) if s})
    result = {
        "case": case.name, "layer": "l1", "description": case.description,
        "matsim_version": MATSIM_VERSION, "matsim_runtime_s": manifest.get("runtime_s"),
        "unsupported": unsupported,
        "metrics": metrics,
        "worst": [{"person": pid, "matsim": ref.get(pid), "mobslim": test.get(pid),
                   "explanation": explanations.get(pid)} for pid in worst],
        "reverse_metrics": rev_metrics,
        "reverse_worst": [{"person": pid, "matsim": by_matsim.get(pid), "mobslim": own.get(pid)}
                          for pid in rev_worst],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ------------------------------------------------------------------ layer 3


def read_counts(path) -> dict:
    """MATSim counts (``counts_v1``): {(link, hour index 0..23): volume}."""
    import xml.etree.ElementTree as ET

    out = {}
    for c in ET.parse(path).getroot().iter("count"):
        for v in c.iter("volume"):
            out[(c.get("loc_id"), int(v.get("h")) - 1)] = float(v.get("val"))
    return out


def _counts_fit(trav, counts: dict) -> dict:
    """Simulated hourly volumes on the counted links against the counts."""
    vol = compare.hourly_volumes(trav)
    sim = np.array([vol.get(k, 0.0) for k in counts])
    obs = np.array(list(counts.values()))
    g = compare.geh(sim, obs)
    return {"counts_geh_lt5_share": float((g < 5).mean()),
            "counts_abs_diff_rel": float(np.abs(sim - obs).sum() / obs.sum())}


def _last_iteration_events(path: Path):
    from mobslim.listeners import ControllerListener

    class Writer(ControllerListener):
        def on_before_mobsim(self, state):
            if state.is_last:
                state.events.add_handler(EventsXMLWriter(
                    path, state.scenario.network, state.scenario.population,
                    state.scenario.transit, vehicle_per_trip=True))
    return Writer()


def _matsim_series(run_dir: Path):
    import pandas as pd

    s = pd.read_csv(run_dir / "scorestats.csv", sep=";").set_index("iteration")
    s = s.rename(columns={"avg_executed": "score_executed", "avg_best": "score_best",
                          "avg_worst": "score_worst", "avg_average": "score_average"})
    return s[["score_executed", "score_best", "score_worst", "score_average"]]


def _mobslim_series(metrics: list[dict]):
    import pandas as pd

    # (mode shares: from the last iteration's trips only, where both sides
    # classify walk-only transit trips the same way)
    df = pd.DataFrame(metrics).set_index("iteration")
    return df[["score_executed", "score_best", "score_worst", "score_average"]]


def run_l3(case: Case, force_matsim: bool = False) -> dict:
    """Full loop, ``case.seeds`` runs per side; compares per-iteration score and
    mode-share curves and final link volumes against MATSim's seed spread."""
    import pandas as pd

    from mobslim import Controller, ControllerConfig
    from mobslim.config.matsim import strategy_manager
    from mobslim.scoring import CharyparNagelScorer

    out = OUTPUT / case.name
    cfg = case_config(case)
    params, unsupported = scoring_params(cfg)
    manager, unsupported_rp = strategy_manager(cfg)
    kwargs, unsupported_q = mobsim_params(cfg)
    unsupported = unsupported + unsupported_rp + unsupported_q
    last = int(cfg.get("controller", "lastIteration"))
    lr = float(cfg.get("scoring", "learningRate", "1.0"))
    base_seed = int(cfg.get("global", "randomSeed", "4711"))
    counts_path = cfg.input_path("counts", "inputCountsFile")
    if counts_path is None and (case.config_path.parent / "counts100.xml").exists() \
            and "plans100" in (cfg.get("plans", "inputPlansFile") or ""):
        counts_path = case.config_path.parent / "counts100.xml"
    counts = read_counts(counts_path) if counts_path else None

    series = {"matsim": [], "mobslim": []}
    final = {"matsim": [], "mobslim": []}  # last-iteration link traversals and trips
    runtimes = {"matsim": [], "mobslim": []}
    for k in range(case.seeds):
        seed = base_seed + k
        mcfg = cfg.copy().set("global", "randomSeed", seed)
        run_dir = run_matsim(mcfg, out / f"matsim_seed{seed}", force=force_matsim)
        runtimes["matsim"].append(json.loads((run_dir / "manifest.json").read_text())["runtime_s"])
        series["matsim"].append(_matsim_series(run_dir))
        ev = events_io.read_events(run_dir / "output_events.xml.gz")
        final["matsim"].append((events_io.link_traversals(ev), events_io.trips(ev), ev))

        scenario = mobslim_scenario(cfg, cfg.input_path("plans", "inputPlansFile"),
                                    keep_pt_routes=False)
        events_path = out / f"mobslim_seed{seed}.events.xml.gz"
        controller = Controller(
            scenario, mobsim=MOBSIMS["qsim"](**kwargs), scorer=CharyparNagelScorer(params),
            strategies=manager, listeners=[_last_iteration_events(events_path)],
            config=ControllerConfig(iterations=last + 1, seed=seed, learning_rate=lr))
        t0 = time.perf_counter()
        res = controller.run()
        runtimes["mobslim"].append(round(time.perf_counter() - t0, 1))
        series["mobslim"].append(_mobslim_series(res.metrics))
        ev = events_io.read_events(events_path)
        final["mobslim"].append((events_io.link_traversals(ev), events_io.trips(ev), ev))

    # per-iteration curves: mean, sd, min, max over seeds
    curves = {}
    for side, runs in series.items():
        stacked = pd.concat(runs, keys=range(len(runs)), names=["seed", "iteration"])
        g = stacked.groupby(level="iteration")
        curves[side] = {stat: getattr(g, stat)().reset_index().to_dict(orient="list")
                        for stat in ("mean", "std", "min", "max")}

    # final iteration: distribution over seeds, and link volumes pairwise
    def final_stats(side):
        rows = []
        for trav, trips, ev in final[side]:
            tt = trips["arrival"] - trips["departure"]
            row = {"trip_tt_mean_s": float(tt.mean()), "trips": int(len(trips))}
            for mode, share in trips["mode"].value_counts(normalize=True).items():
                row[f"trip_share_{mode}"] = float(share)
            if counts:
                row.update(_counts_fit(trav, counts))
            rows.append(row)
        return pd.DataFrame(rows).fillna(0.0)

    finals = {side: pd.concat([final_stats(side),
                               pd.concat([r.iloc[-1:] for r in series[side]]).reset_index(drop=True)],
                              axis=1) for side in series}
    summary = {}
    for metric in sorted(set(finals["matsim"].columns) | set(finals["mobslim"].columns)):
        m = finals["matsim"].get(metric, pd.Series(dtype=float)).fillna(0.0)
        t = finals["mobslim"].get(metric, pd.Series(dtype=float)).fillna(0.0)
        if m.empty or t.empty:
            continue
        sd = float(m.std(ddof=1)) if len(m) > 1 else float("nan")
        summary[metric] = {
            "matsim_mean": float(m.mean()), "matsim_sd": sd,
            "matsim_min": float(m.min()), "matsim_max": float(m.max()),
            "mobslim_mean": float(t.mean()),
            "mobslim_sd": float(t.std(ddof=1)) if len(t) > 1 else float("nan"),
            "z": float((t.mean() - m.mean()) / sd) if sd and sd > 0 else float("nan"),
        }
    pairs = {"matsim_vs_matsim": [], "mobslim_vs_matsim": []}
    n = case.seeds
    for i in range(n):
        if n > 1:
            pairs["matsim_vs_matsim"].append(compare.compare_volumes(
                final["matsim"][i][0], final["matsim"][(i + 1) % n][0]))
        pairs["mobslim_vs_matsim"].append(compare.compare_volumes(
            final["matsim"][i][0], final["mobslim"][i][0]))
    volumes = {name: {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}
               for name, ps in pairs.items() if ps}
    result = {
        "case": case.name, "layer": "l3", "description": case.description,
        "matsim_version": MATSIM_VERSION, "seeds": case.seeds, "iterations": last + 1,
        "unsupported": unsupported, "runtimes": runtimes,
        "curves": curves, "final": summary, "volumes": volumes,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
