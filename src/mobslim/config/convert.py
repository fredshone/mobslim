"""MATSim ``config.xml`` -> mobslim config (YAML).

Uses the translators of :mod:`mobslim.config.matsim`, so a converted config
runs exactly as ``Controller.from_matsim_config`` would. On top of them:

* each ``scoringParameters`` set with a ``subpopulation`` becomes rules
  ``when: subpopulation == '<name>'`` on the parameters where it differs
  from the default set;
* ``strategysettings`` with a ``subpopulation`` become a strategy set of
  that name, chosen by the same rule;
* ``changeMode.ignoreCarAvailability=false`` becomes
  ``persons.car_availability`` rules on the ``carAvail`` and ``hasLicense``
  attributes (for those the population has).

What mobslim cannot honour is returned as ``unsupported`` (and written as a
comment block at the top of the YAML).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from mobslim.config.matsim import (
    MatsimConfig,
    _hermes_params,
    _params,
    _qsim_params,
    controller_params,
    scoring_params_by_subpopulation,
    teleported_modes,
)
from mobslim.config.schema import SCORING_ACTIVITY, SCORING_MODE
from mobslim.timeutils import parse_time

_STRATEGY_NAMES = {
    "BestScore": "best_score", "KeepLastSelected": "keep_selected",
    "ChangeExpBeta": "change_exp_beta", "SelectExpBeta": "select_exp_beta",
    "SelectRandom": "select_random", "ReRoute": "reroute",
    "TimeAllocationMutator": "time_allocation_mutator",
    "TimeAllocationMutator_ReRoute": "time_allocation_mutator",
    "ChangeTripMode": "change_trip_mode", "ChangeLegMode": "change_trip_mode",
}
MODE_FIELDS_ORDER = tuple(SCORING_MODE.fields)
ACT_FIELDS_ORDER = tuple(f for f in SCORING_ACTIVITY.fields if f != "zero_utility")
_SCALARS = {  # config key -> ScoringParams attribute
    "performing": "performing", "late_arrival": "late_arrival",
    "early_departure": "early_departure", "waiting": "waiting",
    "waiting_pt": "marginal_utility_of_waiting_pt",
    "utility_of_line_switch": "utility_of_line_switch",
    "marginal_utility_of_money": "marginal_utility_of_money", "stuck_penalty": "stuck_penalty",
}


def _when(name: str) -> str:
    return f"subpopulation == {name!r}"


def _value(default, by_subpop: dict, path: str, unsupported: list):
    """A plain value, or {default, rules} if subpopulations differ."""
    rules = []
    for name, v in by_subpop.items():
        if v == default:
            continue
        if v is None:
            unsupported.append(f"{path}: subpopulation {name} unsets it (rules cannot set null)")
            continue
        rules.append({"when": _when(name), "set": v})
    return {"default": default, "rules": rules} if rules else default


def _scoring(cfg: MatsimConfig, unsupported: list) -> dict:
    params, by_subpop, u = scoring_params_by_subpopulation(cfg)
    unsupported += u
    learning_rate = float(cfg.get("scoring", "learningRate") or cfg.get("planCalcScore", "learningRate")
                          or "1.0")
    out: dict = {"learning_rate": learning_rate}
    for key, attr in _SCALARS.items():
        out[key] = _value(getattr(params, attr), {n: getattr(p, attr) for n, p in by_subpop.items()},
                          f"scoring.{key}", unsupported)
    modes = {}
    for m in dict.fromkeys([*params.modes, *(m for p in by_subpop.values() for m in p.modes)]):
        modes[m] = {f: _value(getattr(params.mode(m), f),
                              {n: getattr(p.mode(m), f) for n, p in by_subpop.items()},
                              f"scoring.modes.{m}.{f}", unsupported)
                    for f in MODE_FIELDS_ORDER}
    out["modes"] = modes
    acts = {}
    types = dict.fromkeys([*params.activities, *(t for p in by_subpop.values() for t in p.activities)])
    for t in types:
        owners = {n: p.activities[t] for n, p in by_subpop.items() if t in p.activities}
        base = params.activities.get(t) or next(iter(owners.values()))
        entry = {}
        for f in ACT_FIELDS_ORDER:
            entry[f] = _value(getattr(base, f), {n: getattr(a, f) for n, a in owners.items()},
                              f"scoring.activities.{t}.{f}", unsupported)
        if any(a.zero_utility != base.zero_utility for a in owners.values()):
            unsupported.append(f"scoring.activities.{t}.zero_utility differs by subpopulation")
        entry["zero_utility"] = base.zero_utility
        acts[t] = entry
    out["activities"] = acts
    return out


def _replanning(cfg: MatsimConfig, unsupported: list) -> tuple[dict, bool]:
    """(replanning section, whether some change_trip_mode respects car availability)."""
    module = cfg._module("replanning")
    if module is None:
        module = cfg._module("strategy")
    top = _params(module) if module is not None else {}
    beta = float(cfg.get("scoring", "brainExpBeta", "1.0"))
    tam = cfg.get("timeAllocationMutator", "mutationRange", "1800.0")
    tam_range = float(parse_time(tam)) if ":" in tam else float(tam)
    modes = (cfg.get("changeMode", "modes") or "car,pt").split(",")
    ignore_car = cfg.get("changeMode", "ignoreCarAvailability", "true") != "false"
    if cfg.get("timeAllocationMutator", "mutationAffectsDuration", "true") != "true":
        unsupported.append("timeAllocationMutator.mutationAffectsDuration=false")
    sets: dict[str, list] = {}
    for ps in (module.findall("parameterset") if module is not None else ()):
        if ps.get("type") != "strategysettings":
            continue
        p = _params(ps)
        name = p.get("strategyName")
        if name not in _STRATEGY_NAMES:
            unsupported.append(f"replanning: strategy {name}")
            continue
        if p.get("disableAfterIteration") not in (None, "", "-1"):
            unsupported.append(f"replanning: {name}.disableAfterIteration")
        entry = {"name": _STRATEGY_NAMES[name], "weight": float(p.get("weight", "0"))}
        if entry["name"] in ("change_exp_beta", "select_exp_beta"):
            entry["beta"] = beta
        elif entry["name"] == "time_allocation_mutator":
            entry["mutation_range"] = tam_range
        elif entry["name"] == "change_trip_mode":
            entry["modes"] = modes
            entry["ignore_car_availability"] = ignore_car
        sub = p.get("subpopulation") or "default"
        sets.setdefault(sub, []).append(entry)
    for k in ("planSelectorForRemoval",):
        if top.get(k) not in (None, "WorstPlanSelector"):
            unsupported.append(f"replanning.{k}={top[k]}")
    out: dict = {}
    memory = int(top.get("maxAgentPlanMemorySize", "5"))
    if memory == 0:
        unsupported.append("replanning.maxAgentPlanMemorySize=0 (unlimited); using 1000")
        memory = 1000
    out["max_plans"] = memory
    frac = top.get("fractionOfIterationsToDisableInnovation")
    out["disable_innovation_after"] = None if frac in (None, "", "Infinity") else float(frac)
    if not sets:
        unsupported.append("replanning: no strategies (MATSim fails); using mobslim's default set")
    else:
        default = "default" if "default" in sets else next(iter(sets))
        if default != "default":
            unsupported.append(f"replanning: no strategies for persons without a subpopulation; "
                               f"they use {default}'s")
        rules = [{"when": _when(n), "set": n} for n in sets if n != default]
        out["strategy_set"] = {"default": default, "rules": rules} if rules else default
        out["strategy_sets"] = sets
    respects = any(s["name"] == "change_trip_mode" and not s["ignore_car_availability"]
                   for ss in sets.values() for s in ss)
    return out, respects


def matsim_to_mobslim(cfg: MatsimConfig, attribute_names=None) -> tuple[dict, list[str]]:
    """A MATSim config as a mobslim config dict (``(config, unsupported)``).

    ``attribute_names``: the population's person attributes, to translate
    car availability (read from the plans if needed and not given)."""
    unsupported: list[str] = []
    out: dict = {"version": 1}

    controller, u = controller_params(cfg)
    unsupported += u
    out["run"] = {"iterations": controller.iterations, "seed": controller.seed,
                  "output_dir": str(Path(controller.output_dir or "output").resolve()),  # MATSim: from the cwd
                  "write_events_interval": controller.write_events_interval}

    def path(module, param):
        if cfg.get(module, param) in (None, "", "null"):
            return None
        return str(cfg.input_path(module, param))

    for module, param in (("network", "timeVariantNetwork"), ("network", "changeEventsInputFile"),
                          ("households", "inputFile"), ("vehicles", "vehiclesFile")):
        v = cfg.get(module, param)
        if v not in (None, "", "null", "false"):
            unsupported.append(f"{module}.{param}={v}")
    out["inputs"] = {"network": path("network", "inputNetworkFile"),
                     "plans": path("plans", "inputPlansFile"),
                     "person_attributes": path("plans", "inputPersonAttributesFile")}
    if cfg.get("transit", "useTransit") == "true":
        out["transit"] = {"schedule": path("transit", "transitScheduleFile"),
                          "vehicles": path("transit", "vehiclesFile"),
                          "passenger_modes": (cfg.get("transit", "transitModes") or "pt").split(","),
                          "vehicles_on": "network"}
    modes, u = teleported_modes(cfg)
    unsupported += u
    out["modes"] = {m: {"max_speed": p.speed, "beeline_distance_factor": p.beeline_distance_factor}
                    for m, p in modes.items()}

    name = cfg.get("controller", "mobsim", "qsim")
    if name == "hermes":
        kwargs, u = _hermes_params(cfg)
    else:
        kwargs, u = _qsim_params(cfg)
        if name != "qsim":
            unsupported.append(f"controller.mobsim={name} (using qsim)")
            name = "qsim"
    unsupported += u
    out["mobsim"] = {"type": name, **kwargs}
    if controller.travel_time_bin_size != 900.0:
        out["travel_time"] = {"bin_size": controller.travel_time_bin_size}

    out["scoring"] = _scoring(cfg, unsupported)
    out["replanning"], respects_car = _replanning(cfg, unsupported)

    if respects_car:
        if attribute_names is None:
            from mobslim.config.matsim import read_scenario

            scenario, _ = read_scenario(cfg)
            attribute_names = scenario.population.attributes.names
        rules = []
        if "carAvail" in attribute_names:
            rules.append({"when": "carAvail == 'never'", "set": False})
        if "hasLicense" in attribute_names:
            rules.append({"when": "hasLicense == 'no'", "set": False})
        if rules:
            out["persons"] = {"car_availability": {"default": True, "rules": rules}}
    return out, list(dict.fromkeys(unsupported))


def to_yaml(config: dict, unsupported=(), source: Optional[str] = None) -> str:
    """The converted config as YAML, with the unsupported params as a comment."""
    head = [f"# mobslim config translated from {source}" if source else "# mobslim config"]
    if unsupported:
        head.append("# MATSim params mobslim cannot honour (ignored):")
        head += [f"#   {u}" for u in unsupported]
    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False, width=100)
    return "\n".join(head) + "\n" + body

