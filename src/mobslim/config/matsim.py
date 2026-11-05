"""MATSim ``config.xml``: load, edit and write one, and translate it for mobslim.

``MatsimConfig`` loads a config (input paths resolve against its directory,
as in MATSim), lets callers change params, and writes it back with absolute
input paths. The translators each turn one part of it into the objects of
mobslim's Python API and return the params mobslim cannot honour:

* ``read_scenario``: ``network``, ``plans``, ``transit``, ``routing``
  (teleported modes) -> ``Scenario``
* ``mobsim``: ``controller.mobsim`` + ``qsim`` / ``hermes`` -> a mobsim
* ``scoring_params_by_subpopulation``: ``scoring`` -> ``ScoringParams``
  (default and per subpopulation)
* ``strategy_manager``: ``replanning`` (+ ``timeAllocationMutator``,
  ``changeMode``) -> ``StrategyManager``
* ``controller_config``: ``controller``, ``global``, ``travelTimeCalculator``
  -> ``ControllerConfig``

``controller_kwargs`` combines them into keyword arguments for
``Controller``; ``Controller.from_matsim_config`` is the entry point.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from mobslim.timeutils import parse_time


class UnsupportedConfigError(ValueError):
    """A MATSim config uses params mobslim cannot honour (``unsupported``)."""

    def __init__(self, unsupported: list[str]):
        self.unsupported = list(unsupported)
        super().__init__("MATSim config params mobslim cannot honour:\n  " + "\n  ".join(unsupported))


DOCTYPE = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'


class MatsimConfig:
    def __init__(self, root: ET.Element, base_dir: Path):
        self.root = root
        self.base_dir = base_dir  # where relative input paths are resolved

    @classmethod
    def load(cls, path) -> "MatsimConfig":
        path = Path(path).resolve()
        return cls(ET.parse(path).getroot(), path.parent)

    def copy(self) -> "MatsimConfig":
        return MatsimConfig(copy.deepcopy(self.root), self.base_dir)

    def _module(self, name: str, create: bool = False) -> Optional[ET.Element]:
        for m in self.root.findall("module"):
            if m.get("name") == name:
                return m
        if create:
            return ET.SubElement(self.root, "module", name=name)
        return None

    def get(self, module: str, param: str, default: Optional[str] = None) -> Optional[str]:
        m = self._module(module)
        if m is not None:
            for p in m.findall("param"):
                if p.get("name") == param:
                    return p.get("value")
        return default

    def set(self, module: str, param: str, value) -> "MatsimConfig":
        m = self._module(module, create=True)
        for p in m.findall("param"):
            if p.get("name") == param:
                p.set("value", str(value))
                return self
        ET.SubElement(m, "param", name=param, value=str(value))
        return self

    def input_path(self, module: str, param: str) -> Optional[Path]:
        value = self.get(module, param)
        return self._resolve(value) if value else None

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def write(self, path) -> Path:
        """Write with every input file path made absolute."""
        out = self.copy()
        for m in out.root.findall("module"):
            for p in m.iter("param"):
                name, value = p.get("name", ""), p.get("value", "")
                if name.lower().endswith("file") and value and "://" not in value:
                    p.set("value", str(self._resolve(value)))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(out.root)
        body = ET.tostring(out.root, encoding="unicode")
        path.write_text(f'<?xml version="1.0" ?>\n{DOCTYPE}\n{body}\n')
        return path


# qsim params whose MATSim default mobslim matches; any other value is unsupported.
_QSIM_DEFAULTS = {
    "removeStuckVehicles": "false",
    "trafficDynamics": "queue",
    "linkDynamics": "FIFO",
    "timeStepSize": "00:00:01",
    "flowCapacityFactor": None,  # translated
    "storageCapacityFactor": None,  # translated
    "stuckTime": None,  # translated
    "endTime": None,  # translated
    "startTime": None,  # translated (mobslim starts at the first activity end)
    "snapshotperiod": None,  # output only
}


def mobsim_params(cfg: MatsimConfig) -> tuple[dict, list[str]]:
    """``qsim`` module -> (mobslim mobsim kwargs, unsupported params).

    Also lists ``controller.mobsim`` if it is not ``qsim``: these kwargs are
    what mobslim's mobsims take to match MATSim's QSim.
    """
    kwargs, unsupported = _qsim_params(cfg)
    if cfg.get("controller", "mobsim", "qsim") != "qsim":
        unsupported.append(f"controller.mobsim={cfg.get('controller', 'mobsim')} (reference side)")
    return kwargs, unsupported


def _qsim_params(cfg: MatsimConfig) -> tuple[dict, list[str]]:
    kwargs: dict = {}
    unsupported: list[str] = []
    if (v := cfg.get("qsim", "flowCapacityFactor")) is not None:
        kwargs["flow_capacity_factor"] = float(v)
    if (v := cfg.get("qsim", "storageCapacityFactor")) is not None:
        kwargs["storage_capacity_factor"] = float(v)
    # MATSim forces a vehicle on when ``now - last_moved > stuckTime``, checked
    # once per time step: the first step that passes is stuckTime + 1 step
    # (QNodeImpl). mobslim is continuous, so it gets that effective time.
    step = float(parse_time(cfg.get("qsim", "timeStepSize", "00:00:01")))
    kwargs["stuck_time"] = float(cfg.get("qsim", "stuckTime", "10")) + step
    end = cfg.get("qsim", "endTime")
    if end and parse_time(end) > 0:  # 00:00:00 = run until everyone is done
        kwargs["end_time"] = float(parse_time(end))
    start = cfg.get("qsim", "startTime")
    if start and parse_time(start) > 0:
        unsupported.append(f"qsim.startTime={start}")
    qsim = cfg._module("qsim")
    for p in qsim.findall("param") if qsim is not None else ():
        name, value = p.get("name"), p.get("value")
        if name not in _QSIM_DEFAULTS:
            unsupported.append(f"qsim.{name}={value}")
        elif _QSIM_DEFAULTS[name] is not None and value != _QSIM_DEFAULTS[name]:
            unsupported.append(f"qsim.{name}={value}")
    return kwargs, unsupported


# hermes params mobslim's HermesSim translates; any other is unsupported.
_HERMES_TRANSLATED = {"flowCapacityFactor", "storageCapacityFactor", "stuckTime", "endTime"}


def _hermes_params(cfg: MatsimConfig) -> tuple[dict, list[str]]:
    """``hermes`` module -> (HermesSim kwargs, unsupported), with the same
    stuck-time rule as the qsim (MATSim's defaults: capacities 1, stuck 10 s)."""
    kwargs: dict = {}
    unsupported: list[str] = []
    if (v := cfg.get("hermes", "flowCapacityFactor")) is not None:
        kwargs["flow_capacity_factor"] = float(v)
    if (v := cfg.get("hermes", "storageCapacityFactor")) is not None:
        kwargs["storage_capacity_factor"] = float(v)
    kwargs["stuck_time"] = float(cfg.get("hermes", "stuckTime", "10")) + 1.0
    end = cfg.get("hermes", "endTime")
    if end and parse_time(end) > 0:
        kwargs["end_time"] = float(parse_time(end))
    hermes = cfg._module("hermes")
    for p in hermes.findall("param") if hermes is not None else ():
        if p.get("name") not in _HERMES_TRANSLATED:
            unsupported.append(f"hermes.{p.get('name')}={p.get('value')}")
    return kwargs, unsupported


def mobsim(cfg: MatsimConfig):
    """``controller.mobsim`` (+ its module) -> (mobslim mobsim, unsupported).

    ``qsim`` gives a ``QSim``, ``hermes`` a ``HermesSim`` (Rust kernel if
    built); anything else falls back to the ``QSim`` and is listed.
    """
    from mobslim.mobsim import HermesSim, HermesSimConfig, QSim, QSimConfig

    name = cfg.get("controller", "mobsim", "qsim")
    if name == "hermes":
        kwargs, unsupported = _hermes_params(cfg)
        return HermesSim(HermesSimConfig(**kwargs)), unsupported
    kwargs, unsupported = _qsim_params(cfg)
    if name != "qsim":
        unsupported.append(f"controller.mobsim={name} (using qsim)")
    return QSim(QSimConfig(**kwargs)), unsupported


# ---------------------------------------------------------------- scoring

_MATSIM_DEFAULT_MODES = ("car", "pt", "walk", "bike", "ride", "other")
# scoring params that don't affect a score (loop-level or output-only)
_SCORING_IGNORED = {"learningRate", "brainExpBeta", "pathSizeLogitBeta", "writeExperiencedPlans",
                    "explainScores", "writeScoreExplanations", "memorizingExperiencedPlans",
                    "fractionOfIterationsToStartScoreMSA", "usesDeprecatedSyntax"}
_SCORING_TOP = {"lateArrival", "earlyDeparture", "performing", "waiting", "waitingPt",
                "utilityOfLineSwitch", "marginalUtilityOfMoney", "subpopulation"}
_MODE_KEYS = {"mode", "constant", "marginalUtilityOfTraveling_util_hr", "marginalUtilityOfDistance_util_m",
              "monetaryDistanceRate", "dailyMonetaryConstant", "dailyUtilityConstant"}
_ACT_KEYS = {"activityType", "priority", "typicalDuration", "minimalDuration", "openingTime",
             "closingTime", "latestStartTime", "earliestEndTime", "typicalDurationScoreComputation",
             "scoringThisActivityAtAll"}


def _params(el) -> dict:
    return {p.get("name"): p.get("value") for p in el.findall("param")}


def _opt_time(v) -> Optional[float]:
    return float(parse_time(v)) if v not in (None, "", "undefined") else None


def _subpopulation(ps) -> Optional[str]:
    v = _params(ps).get("subpopulation")
    return None if v in (None, "", "null") else v


def scoring_params_by_subpopulation(cfg: MatsimConfig):
    """``scoring`` module -> (default ``ScoringParams``, ``{subpopulation:
    ScoringParams}``, unsupported params).

    Each ``scoringParameters`` set with a ``subpopulation`` scores the
    persons whose ``subpopulation`` attribute names it (see
    ``CharyparNagelScorer(subpopulations=...)``); the set without one scores
    everyone else. If every set names a subpopulation, the first is also
    the default. MATSim's defaults are filled in where the config is silent:
    its six default modes (traveling -6 util/h), ``waitingPt`` = pt's
    traveling, ``relative`` zero-utility durations, and ``abortedPlanScore``
    (the stuck penalty) derived from the betas as MATSim does.
    """
    unsupported: list[str] = []
    module = cfg._module("scoring")
    if module is None:
        module = cfg._module("planCalcScore")
    top = _params(module) if module is not None else {}
    sets = [ps for ps in (module.findall("parameterset") if module is not None else ())
            if ps.get("type") == "scoringParameters"]
    if not sets:
        return _scoring_set(top, module, unsupported), {}, unsupported
    default = next((ps for ps in sets if _subpopulation(ps) is None), sets[0])
    by_subpop = {}
    for ps in sets:
        name = _subpopulation(ps)
        if name is not None:
            by_subpop[name] = _scoring_set(top, ps, unsupported, f"scoring[{name}]")
    params = (_scoring_set(top, default, unsupported)
              if _subpopulation(default) is None else by_subpop[_subpopulation(default)])
    return params, by_subpop, list(dict.fromkeys(unsupported))


def scoring_params(cfg: MatsimConfig):
    """``scoring`` module -> (default ``ScoringParams``, unsupported params).

    Only the default set; see :func:`scoring_params_by_subpopulation` for
    the per-subpopulation sets (listed as unsupported here).
    """
    params, by_subpop, unsupported = scoring_params_by_subpopulation(cfg)
    unsupported += [f"scoring: scoringParameters for subpopulation {name}" for name in by_subpop]
    return params, unsupported


def _scoring_set(top: dict, sp, unsupported: list, label: str = "scoring"):
    """One ``scoringParameters`` set (falling back to the module's own params)."""
    from mobslim.scoring import ActivityParams, ModeParams, ScoringParams

    values = {**top, **(_params(sp) if sp is not None else {})}
    for k in values:
        if k not in _SCORING_TOP | _SCORING_IGNORED:
            unsupported.append(f"{label}.{k}={values[k]}")
    if values.get("usingOldScoringBelowZeroUtilityDuration") == "true":
        unsupported.append(f"{label}.usingOldScoringBelowZeroUtilityDuration=true")

    modes = {m: {} for m in _MATSIM_DEFAULT_MODES}
    acts = {}
    for ps in (sp.findall("parameterset") if sp is not None else ()):
        p = _params(ps)
        if ps.get("type") == "modeParams":
            modes[p["mode"]] = p
            unsupported += [f"{label}.modeParams[{p['mode']}].{k}" for k in p if k not in _MODE_KEYS]
        elif ps.get("type") == "activityParams":
            acts[p["activityType"]] = p
            unsupported += [f"{label}.activityParams[{p['activityType']}].{k}"
                            for k in p if k not in _ACT_KEYS]

    def f(d, k, default):
        v = d.get(k)
        return float(v) if v not in (None, "") else default

    mode_params = {
        m: ModeParams(
            constant=f(p, "constant", 0.0),
            marginal_utility_of_traveling=f(p, "marginalUtilityOfTraveling_util_hr", -6.0),
            marginal_utility_of_distance=f(p, "marginalUtilityOfDistance_util_m", 0.0),
            monetary_distance_rate=f(p, "monetaryDistanceRate", 0.0),
            daily_constant=f(p, "dailyUtilityConstant", 0.0),
            daily_monetary_constant=f(p, "dailyMonetaryConstant", 0.0),
        ) for m, p in modes.items()}
    act_params = {}
    for t, p in acts.items():
        if t.endswith(" interaction"):
            continue  # stage activities: mobslim has none (a pt trip is one leg)
        if p.get("scoringThisActivityAtAll") == "false":
            unsupported.append(f"{label}.activityParams[{t}].scoringThisActivityAtAll=false")
        typical = _opt_time(p.get("typicalDuration"))
        if typical is None:
            unsupported.append(f"{label}.activityParams[{t}]: no typicalDuration")
            continue
        act_params[t] = ActivityParams(
            typical_duration=typical,
            priority=f(p, "priority", 1.0),
            zero_utility=p.get("typicalDurationScoreComputation") or "relative",
            opening_time=_opt_time(p.get("openingTime")),
            closing_time=_opt_time(p.get("closingTime")),
            latest_start_time=_opt_time(p.get("latestStartTime")),
            earliest_end_time=_opt_time(p.get("earliestEndTime")),
            minimal_duration=_opt_time(p.get("minimalDuration")),
        )

    performing = f(values, "performing", 6.0)
    late = f(values, "lateArrival", -18.0)
    early = f(values, "earlyDeparture", 0.0)
    waiting = f(values, "waiting", 0.0)
    worst_travel = min([0.0] + [m.marginal_utility_of_traveling for m in mode_params.values()])
    aborted = min(min(late, early), min(worst_travel - performing, waiting - performing)) * 24.0
    return ScoringParams(
        performing=performing,
        activities=act_params,
        modes=mode_params,
        default_activity=None,  # MATSim fails on unknown activity types too
        stuck_penalty=aborted,
        marginal_utility_of_waiting_pt=f(values, "waitingPt",
                                         mode_params["pt"].marginal_utility_of_traveling),
        utility_of_line_switch=f(values, "utilityOfLineSwitch", -1.0),
        late_arrival=late,
        early_departure=early,
        waiting=waiting,
        marginal_utility_of_money=f(values, "marginalUtilityOfMoney", 1.0),
    )


def set_scoring_param(cfg: MatsimConfig, set_type: str, key: str, value) -> MatsimConfig:
    """Set ``key`` in every ``set_type`` parameterset of the scoring module
    (e.g. ``activityParams``/``typicalDurationScoreComputation``)."""
    module = cfg._module("scoring")
    for ps in module.iter("parameterset"):
        if ps.get("type") == set_type:
            for p in ps.findall("param"):
                if p.get("name") == key:
                    p.set("value", str(value))
                    break
            else:
                ET.SubElement(ps, "param", name=key, value=str(value))
    return cfg


# -------------------------------------------------------------- replanning


def strategy_manager(cfg: MatsimConfig):
    """``replanning`` module -> (mobslim ``StrategyManager``, unsupported).

    As MATSim builds them: selectors choose among a person's plans;
    innovative strategies copy a random plan (``RandomPlanSelector``) and
    change it, rerouting afterwards.
    """
    from mobslim.replanning import (
        BestScoreSelector,
        ChangeExpBetaSelector,
        ChangeTripMode,
        ExpBetaSelector,
        KeepSelected,
        PlanStrategy,
        RandomPlanSelector,
        ReRoute,
        StrategyManager,
        TimeAllocationMutator,
    )

    unsupported: list[str] = []
    module = cfg._module("replanning")
    if module is None:
        module = cfg._module("strategy")
    top = _params(module) if module is not None else {}
    beta = float(cfg.get("scoring", "brainExpBeta", "1.0"))
    tam = cfg.get("timeAllocationMutator", "mutationRange", "1800.0")
    tam_range = float(parse_time(tam)) if ":" in tam else float(tam)  # seconds
    modes = (cfg.get("changeMode", "modes") or "car,pt").split(",")
    ignore_car = cfg.get("changeMode", "ignoreCarAvailability", "true") != "false"

    def innovate(*modules):
        return PlanStrategy(RandomPlanSelector(), [*modules, ReRoute()])

    builders = {
        "BestScore": lambda: PlanStrategy(BestScoreSelector()),
        "KeepLastSelected": lambda: PlanStrategy(KeepSelected()),
        "ChangeExpBeta": lambda: PlanStrategy(ChangeExpBetaSelector(beta)),
        "SelectExpBeta": lambda: PlanStrategy(ExpBetaSelector(beta)),
        "SelectRandom": lambda: PlanStrategy(RandomPlanSelector()),
        "ReRoute": lambda: PlanStrategy(RandomPlanSelector(), [ReRoute()]),
        "TimeAllocationMutator": lambda: innovate(TimeAllocationMutator(tam_range)),
        "TimeAllocationMutator_ReRoute": lambda: innovate(TimeAllocationMutator(tam_range)),
        "ChangeTripMode": lambda: innovate(ChangeTripMode(modes, ignore_car)),
        "ChangeLegMode": lambda: innovate(ChangeTripMode(modes, ignore_car)),
    }
    chosen = []
    for ps in (module.findall("parameterset") if module is not None else ()):
        if ps.get("type") != "strategysettings":
            continue
        p = _params(ps)
        name, weight = p.get("strategyName"), float(p.get("weight", "0"))
        if p.get("subpopulation"):
            unsupported.append(f"replanning: strategy {name} for subpopulation {p['subpopulation']}")
            continue
        if name not in builders:
            unsupported.append(f"replanning: strategy {name}")
            continue
        if p.get("disableAfterIteration") not in (None, "", "-1"):
            unsupported.append(f"replanning: {name}.disableAfterIteration")
        chosen.append((weight, builders[name]()))
    if cfg.get("timeAllocationMutator", "mutationAffectsDuration", "true") != "true":
        unsupported.append("timeAllocationMutator.mutationAffectsDuration=false")
    for k in ("planSelectorForRemoval",):
        if top.get(k) not in (None, "WorstPlanSelector"):
            unsupported.append(f"replanning.{k}={top[k]}")

    first = int(cfg.get("controller", "firstIteration", "0"))
    last = int(cfg.get("controller", "lastIteration", "10"))
    frac = top.get("fractionOfIterationsToDisableInnovation")
    disable_at = None
    if frac not in (None, "", "Infinity"):
        disable_at = first + int(float(frac) * (last - first))
    memory = int(top.get("maxAgentPlanMemorySize", "5"))
    if memory == 0:
        unsupported.append("replanning.maxAgentPlanMemorySize=0 (unlimited); using 1000")
        memory = 1000
    return StrategyManager(chosen, max_plans=memory, disable_innovation_at=disable_at), unsupported


# ---------------------------------------------------------------- scenario

_ROUTING_MODE_KEYS = {"mode", "teleportedModeSpeed", "beelineDistanceFactor"}


def teleported_modes(cfg: MatsimConfig):
    """``routing`` teleportedModeParameters -> ({mode: TeleportedModeParams},
    unsupported). Starts from mobslim's defaults (MATSim's walk and bike)
    unless ``clearDefaultTeleportedModeParams`` is true."""
    from mobslim.routing.teleport import (
        DEFAULT_TELEPORTED_MODES,
        TeleportedModeParams,
    )

    unsupported: list[str] = []
    module = cfg._module("routing")
    if module is None:
        module = cfg._module("planscalcroute")
    clear = cfg.get("routing", "clearDefaultTeleportedModeParams", "false") == "true"
    modes = {} if clear else dict(DEFAULT_TELEPORTED_MODES)
    for ps in (module.findall("parameterset") if module is not None else ()):
        if ps.get("type") != "teleportedModeParameters":
            continue
        p = _params(ps)
        mode = p.get("mode")
        if p.get("teleportedModeSpeed") in (None, "", "null"):
            unsupported.append(f"routing.teleportedModeParameters[{mode}]: no teleportedModeSpeed")
            continue
        modes[mode] = TeleportedModeParams(float(p["teleportedModeSpeed"]),
                                           float(p.get("beelineDistanceFactor") or 1.3))
        unsupported += [f"routing.teleportedModeParameters[{mode}].{k}={v}" for k, v in p.items()
                        if k not in _ROUTING_MODE_KEYS and v not in (None, "", "null")]
    return modes, unsupported


def read_scenario(cfg: MatsimConfig):
    """``network``, ``plans``, ``transit`` and ``routing`` -> (``Scenario``,
    unsupported).

    Routes in the plans file are kept for single-leg network trips, as MATSim
    keeps them; other legs are routed by the controller before the first
    iteration. Transit vehicles run on the network, as in MATSim's QSim.
    """
    from mobslim.network import read_network_xml
    from mobslim.population import read_population_xml
    from mobslim.scenario import Scenario

    unsupported: list[str] = []
    network_path = cfg.input_path("network", "inputNetworkFile")
    plans_path = cfg.input_path("plans", "inputPlansFile")
    if network_path is None or plans_path is None:
        raise ValueError("MATSim config needs network.inputNetworkFile and plans.inputPlansFile.")
    for module, param in (("network", "timeVariantNetwork"), ("network", "changeEventsInputFile"),
                          ("households", "inputFile"), ("vehicles", "vehiclesFile")):
        v = cfg.get(module, param)
        if v not in (None, "", "null", "false"):
            unsupported.append(f"{module}.{param}={v}")
    network = read_network_xml(network_path)
    attributes_path = (cfg.input_path("plans", "inputPersonAttributesFile")
                       if cfg.get("plans", "inputPersonAttributesFile") not in (None, "", "null")
                       else None)
    population = read_population_xml(plans_path, network, keep_routes=True,
                                     attributes_file=attributes_path)
    modes, unsupported_tp = teleported_modes(cfg)
    unsupported += unsupported_tp
    transit = None
    if cfg.get("transit", "useTransit") == "true":
        from mobslim.transit import TransitConfig, read_transit

        passenger_modes = tuple((cfg.get("transit", "transitModes") or "pt").split(","))
        transit = read_transit(cfg.input_path("transit", "transitScheduleFile"),
                               cfg.input_path("transit", "vehiclesFile"), network,
                               TransitConfig(passenger_modes=passenger_modes, vehicles="network"))
    return Scenario(network, population, teleported_modes=modes, transit=transit), unsupported


# -------------------------------------------------------------- controller


def controller_config(cfg: MatsimConfig):
    """``controller``, ``global``, ``scoring.learningRate`` and
    ``travelTimeCalculator`` -> (``ControllerConfig``, unsupported).

    ``outputDirectory`` is relative to the working directory, as in MATSim.
    """
    from mobslim.controller import ControllerConfig

    unsupported: list[str] = []
    first = int(cfg.get("controller", "firstIteration", "0"))
    last = int(cfg.get("controller", "lastIteration", "10"))
    if first != 0:
        unsupported.append(f"controller.firstIteration={first} (iterations are numbered from 0)")
    kwargs = dict(
        iterations=last - first + 1,
        seed=int(cfg.get("global", "randomSeed", "4711")),
        learning_rate=float(cfg.get("scoring", "learningRate")
                            or cfg.get("planCalcScore", "learningRate") or "1.0"),
        output_dir=cfg.get("controller", "outputDirectory"),
        write_events_interval=int(cfg.get("controller", "writeEventsInterval", "10")),
    )
    if (v := cfg.get("travelTimeCalculator", "traveltimeBinSize")) is not None:
        kwargs["travel_time_bin_size"] = float(parse_time(v)) if ":" in v else float(v)
    return ControllerConfig(**kwargs), unsupported


def controller_kwargs(cfg: MatsimConfig) -> tuple[dict, list[str]]:
    """The whole config -> (keyword arguments for ``Controller``, unsupported).

    Keys: ``scenario``, ``mobsim``, ``scorer``, ``strategies``, ``config``.
    """
    from mobslim.scoring import CharyparNagelScorer

    scenario, u_scenario = read_scenario(cfg)
    sim, u_mobsim = mobsim(cfg)
    params, by_subpop, u_scoring = scoring_params_by_subpopulation(cfg)
    manager, u_replanning = strategy_manager(cfg)
    config, u_controller = controller_config(cfg)
    scorer = CharyparNagelScorer(params, subpopulations=by_subpop)
    kwargs = dict(scenario=scenario, mobsim=sim, scorer=scorer,
                  strategies=manager, config=config)
    return kwargs, u_scenario + u_mobsim + u_scoring + u_replanning + u_controller
