"""The mobslim config: a YAML run configuration with person filters.

``MobslimConfig.load(path)`` reads a YAML file (see
:mod:`mobslim.config.schema` for every parameter), fills in defaults
(MATSim's, each logged), and checks types and ranges. Parameters marked
*per person* may be given as a default plus rules on person attributes; the
last matching rule wins (see :mod:`mobslim.config.conditions`)::

    scoring:
      modes:
        walk:
          marginal_utility_of_traveling:
            default: -10
            rules:
              - when: age > 50
                set: -15
              - when: age > 70
                set: -30

``build_scenario()`` reads the inputs and runs :meth:`MobslimConfig.precheck`
(attributes, activity types, modes, strategy sets), raising one
:class:`ConfigError` that lists every problem. ``controller_kwargs()`` turns
the config into keyword arguments for :class:`~mobslim.controller.Controller`
(``Controller.from_mobslim_config`` is the entry point), including a
listener that writes ``output_config.yaml``: the complete config with every
default, which reproduces the run when loaded.

A run is configured by a mobslim config **or** a MATSim config, never both;
``python -m mobslim.config from-matsim`` translates the latter.
"""

from __future__ import annotations

import datetime
import difflib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from mobslim.config.conditions import (
    Condition,
    ConditionError,
    check_condition,
    evaluate,
    parse_condition,
)
from mobslim.config.schema import (
    REQUIRED,
    SCHEMA,
    STRATEGIES,
    MapOf,
    Param,
    Section,
    StrategyList,
)
from mobslim.listeners import ControllerListener
from mobslim.timeutils import parse_time

logger = logging.getLogger("mobslim.config")

CAR_AVAILABILITY = "mobslim:car_availability"  # derived person attribute
_MISSING = object()


class ConfigError(ValueError):
    """A mobslim config with problems (``problems``: one message each)."""

    def __init__(self, problems: list[str], source: str = "mobslim config"):
        self.problems = list(problems)
        super().__init__(f"{source}: {len(self.problems)} problem(s):\n  " + "\n  ".join(self.problems))


@dataclass(frozen=True)
class Rule:
    when: Condition
    value: Any
    raw_when: Any  # as written, for the output config


@dataclass(frozen=True)
class Filtered:
    """A per-person value: ``default``, then ``rules`` in order (last match wins)."""

    default: Any
    rules: tuple

    def evaluate(self, attributes, dtype=float) -> np.ndarray:
        """One value per person."""
        out = np.empty(attributes.n_persons, dtype=dtype)
        out[:] = self.default
        for rule in self.rules:
            out[evaluate(rule.when, attributes)] = rule.value
        return out


def plain(value):
    """The default of a :class:`Filtered`, else the value itself."""
    return value.default if isinstance(value, Filtered) else value


def has_rules(value) -> bool:
    return isinstance(value, Filtered) and bool(value.rules)


# ------------------------------------------------------------------ YAML

class _Loader(yaml.SafeLoader):
    """YAML 1.1 without the surprises: only true/false are booleans (not
    yes/no/on/off), so ``{hasLicense: no}`` compares with the string "no"."""


_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
                              list("tTfF"))


def read_yaml(text: str):
    return yaml.load(text, Loader=_Loader)


def _scalar_yaml(v) -> str:
    if isinstance(v, float) and math.isinf(v):
        return ".inf" if v > 0 else "-.inf"
    s = yaml.safe_dump(v, default_flow_style=True, width=10_000, allow_unicode=True).strip()
    return s[:-3].strip() if s.endswith("...") else s


# ------------------------------------------------------------------ resolving

class _Resolver:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.problems: list[str] = []
        self.defaults: dict[str, Any] = {}  # path -> value filled in

    def error(self, path: str, msg: str) -> None:
        self.problems.append(f"{path}: {msg}" if path else msg)

    def resolve(self, spec, raw, path: str, fallback=_MISSING):
        if isinstance(spec, Param):
            return self.param(spec, raw, path, fallback)
        if isinstance(spec, Section):
            return self.section(spec, raw, path, fallback)
        if isinstance(spec, MapOf):
            return self.map_of(spec, raw, path)
        return self.strategies(raw, path, fallback)

    # -- params
    def param(self, p: Param, raw, path: str, fallback):
        if raw is _MISSING:
            value = fallback if fallback is not _MISSING else p.default
            if value is REQUIRED:
                self.error(path, f"required ({p.doc})")
                return None
            value = self.coerce(p, value, path)
            self.defaults[path] = value
            return value
        if isinstance(raw, dict) and p.type != "strlist":
            if not p.filterable:
                self.error(path, "cannot vary by person (a plain value is needed here)")
                return None
            return self.filtered(p, raw, path, fallback)
        return self.coerce(p, raw, path)

    def filtered(self, p: Param, raw: dict, path: str, fallback) -> Filtered:
        unknown = set(raw) - {"default", "rules"}
        if unknown:
            self.error(path, f"a per-person value has keys 'default' and 'rules', not "
                             f"{', '.join(map(repr, sorted(unknown, key=str)))}")
        default = self.param(p, raw.get("default", _MISSING), f"{path}.default", fallback)
        rules = []
        raw_rules = raw.get("rules", [])
        if not isinstance(raw_rules, list):
            self.error(f"{path}.rules", "expected a list of {when, set}")
            raw_rules = []
        for i, r in enumerate(raw_rules):
            rp = f"{path}.rules[{i}]"
            if not isinstance(r, dict) or set(r) != {"when", "set"}:
                self.error(rp, "a rule is a mapping with exactly 'when' and 'set'")
                continue
            try:
                cond = parse_condition(r["when"])
            except ConditionError as e:
                self.error(f"{rp}.when", str(e))
                continue
            if r["set"] is None:
                self.error(f"{rp}.set", "a rule cannot set null")
                continue
            value = self.coerce(p, r["set"], f"{rp}.set")
            rules.append(Rule(cond, value, r["when"]))
        return Filtered(default, tuple(rules))

    def coerce(self, p: Param, v, path: str):
        if v is None:
            if not p.nullable:
                self.error(path, "may not be null")
            return None
        t = p.type
        original = v
        bad = False
        if t == "float":
            bad = isinstance(v, bool) or not isinstance(v, (int, float))
            v = None if bad else float(v)
        elif t == "int":
            bad = isinstance(v, bool) or not isinstance(v, int)
        elif t == "bool":
            bad = not isinstance(v, bool)
        elif t == "str":
            bad = not isinstance(v, str)
        elif t == "time":
            if isinstance(v, str):
                try:
                    v = float(parse_time(v))
                except ValueError:
                    self.error(path, f"expected seconds or HH:MM:SS, got {v!r}")
                    return None
            elif isinstance(v, bool) or not isinstance(v, (int, float)):
                bad = True
            else:
                v = float(v)
        elif t == "path":
            bad = not isinstance(v, str)
            if not bad:
                v = str(Path(v) if Path(v).is_absolute() else (self.base_dir / v).resolve())
        elif t == "strlist":
            bad = not isinstance(v, list) or not all(isinstance(x, str) for x in v)
            if not bad:
                v = list(v)
        if bad:
            kind = {"float": "a number", "int": "an integer", "bool": "true or false",
                    "str": "a string", "path": "a path", "strlist": "a list of strings",
                    "time": "seconds or HH:MM:SS"}[t]
            self.error(path, f"expected {kind}, got {original!r}")
            return None
        if p.choices and v not in p.choices:
            self.error(path, f"expected one of {', '.join(map(str, p.choices))}, got {v!r}")
        if p.min is not None and isinstance(v, (int, float)) and (
                v < p.min or (p.min_exclusive and v == p.min)):
            self.error(path, f"expected a value {'>' if p.min_exclusive else '>='} {p.min}, got {v!r}")
        return v

    # -- containers
    def _unknown(self, raw: dict, valid, path: str, what="parameter") -> None:
        for k in raw:
            if k not in valid:
                close = difflib.get_close_matches(str(k), list(valid), n=1)
                hint = f"; did you mean {close[0]!r}?" if close else \
                    f" (valid: {', '.join(valid)})"
                self.error(f"{path}.{k}" if path else str(k), f"unknown {what}{hint}")

    def section(self, s: Section, raw, path: str, fallback):
        if raw is _MISSING or raw is None:
            raw = {}
        if not isinstance(raw, dict):
            self.error(path, f"expected a mapping, got {raw!r}")
            raw = {}
        self._unknown(raw, list(s.fields), path)
        fb = fallback if isinstance(fallback, dict) else {}
        return {k: self.resolve(spec, raw.get(k, _MISSING), f"{path}.{k}" if path else k,
                                fb.get(k, _MISSING))
                for k, spec in s.fields.items()}

    def map_of(self, m: MapOf, raw, path: str) -> dict:
        if raw is _MISSING or raw is None:
            raw = {}
        if not isinstance(raw, dict):
            self.error(path, f"expected a mapping of {m.key} -> settings, got {raw!r}")
            raw = {}
        out = {}
        for key, default in m.defaults.items():
            if key not in raw:
                out[key] = self.resolve(m.item, _MISSING, f"{path}.{key}", default)
                if isinstance(m.item, StrategyList):
                    self.defaults[f"{path}.{key}"] = out[key]
        for key, value in raw.items():
            if not isinstance(key, str):
                self.error(path, f"{m.key} names must be strings, got {key!r}")
                continue
            if isinstance(m.item, StrategyList):
                out[key] = self.resolve(m.item, value, f"{path}.{key}")
            else:
                out[key] = self.resolve(m.item, value, f"{path}.{key}", m.defaults.get(key, _MISSING))
        return out

    def strategies(self, raw, path: str, fallback) -> list:
        record = raw is _MISSING
        if raw is _MISSING:
            raw = fallback if fallback is not _MISSING else []
        if not isinstance(raw, list) or not raw:
            self.error(path, "expected a non-empty list of strategies")
            return []
        out = []
        for i, item in enumerate(raw):
            ip = f"{path}[{i}]"
            if not isinstance(item, dict) or "name" not in item:
                self.error(ip, "a strategy is a mapping with a 'name'")
                continue
            name = item["name"]
            if name not in STRATEGIES:
                close = difflib.get_close_matches(str(name), list(STRATEGIES), n=1)
                hint = f"; did you mean {close[0]!r}?" if close else ""
                self.error(f"{ip}.name", f"unknown strategy {name!r}{hint} "
                                         f"(strategies: {', '.join(STRATEGIES)})")
                continue
            if record:
                value = self.section(STRATEGIES[name], _MISSING, ip, item)
            else:
                value = self.section(STRATEGIES[name], item, ip, _MISSING)
            if name == "change_trip_mode" and value.get("modes") is not None and len(value["modes"]) < 2:
                self.error(f"{ip}.modes", "needs at least two modes")
            out.append(value)
        return out


# -------------------------------------------------------------- the config

class MobslimConfig:
    """A resolved mobslim config (see the module docstring)."""

    def __init__(self, values: dict, defaults: dict, source: Optional[Path] = None,
                 base_dir: Optional[Path] = None):
        self.values = values
        self.defaults = defaults  # path -> default value filled in
        self.source = source
        self.base_dir = base_dir or Path.cwd()

    # -- loading
    @classmethod
    def load(cls, path) -> "MobslimConfig":
        path = Path(path).resolve()
        try:
            raw = read_yaml(path.read_text())
        except yaml.YAMLError as e:
            raise ConfigError([f"not valid YAML: {e}"], str(path)) from None
        return cls.from_dict(raw, path.parent, source=path)

    @classmethod
    def from_dict(cls, raw, base_dir=None, source: Optional[Path] = None) -> "MobslimConfig":
        base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
        r = _Resolver(base_dir)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError([f"expected a mapping at the top level, got {type(raw).__name__}"],
                              str(source or "mobslim config"))
        values = r.resolve(SCHEMA, raw, "")
        cfg = cls(values, r.defaults, source, base_dir)
        r.problems += cfg._check_static()
        if r.problems:
            raise ConfigError(r.problems, str(source or "mobslim config"))
        return cfg

    def get(self, path: str):
        """The value at a dotted path, e.g. ``"scoring.modes.car.constant"``."""
        v = self.values
        for part in path.split("."):
            v = v[part]
        return v

    def _check_static(self) -> list[str]:
        """Checks that need no scenario."""
        problems = []
        v = self.values
        for key in ("network", "plans", "person_attributes"):
            p = v["inputs"][key]
            if p is not None and not Path(p).exists():
                problems.append(f"inputs.{key}: file not found: {p}")
        t = v["transit"]
        if (t["schedule"] is None) != (t["vehicles"] is None):
            problems.append("transit: give both schedule and vehicles, or neither")
        for key in ("schedule", "vehicles"):
            if t[key] is not None and not Path(t[key]).exists():
                problems.append(f"transit.{key}: file not found: {t[key]}")
        frac = v["replanning"]["disable_innovation_after"]
        if frac is not None and frac > 1:
            problems.append(f"replanning.disable_innovation_after: a fraction in [0, 1], got {frac}")
        sets = v["replanning"]["strategy_sets"]
        chosen = v["replanning"]["strategy_set"]
        names = [plain(chosen)] + [r.value for r in (chosen.rules if isinstance(chosen, Filtered) else ())]
        for name in dict.fromkeys(names):
            if name is not None and name not in sets:
                problems.append(f"replanning.strategy_set: no strategy set {name!r} "
                                f"(sets: {', '.join(sets)})")
        innovation_stops = frac is not None
        for name, strategies in sets.items():
            weights = [(s["weight"] or 0.0) for s in strategies]
            if strategies and sum(weights) <= 0:
                problems.append(f"replanning.strategy_sets.{name}: no strategy has a positive weight")
            elif innovation_stops and strategies and not any(
                    w > 0 and s["name"] in _SELECTORS for s, w in zip(strategies, weights)):
                problems.append(f"replanning.strategy_sets.{name}: innovation stops "
                                "(disable_innovation_after) but no plan selector has a positive weight")
        for name, mode in v["modes"].items():
            if name in v["transit"]["passenger_modes"]:
                problems.append(f"modes.{name}: a transit mode cannot be teleported")
        return problems

    # -- logging and output
    def log_defaults(self) -> None:
        """Log every default filled in (INFO), then a summary line."""
        for path, value in self.defaults.items():
            shown = "[...]" if isinstance(value, list) and value and isinstance(value[0], dict) else value
            logger.info("default: %s = %s", path, shown)
        logger.info("mobslim config %s: %d defaults filled in (see output_config.yaml)",
                    self.source or "", len(self.defaults))

    def derived(self) -> dict[str, tuple[Any, str]]:
        """Values computed from others because the config left them null:
        ``{path: (value, explanation)}``."""
        out = {}
        s = self.values["scoring"]
        if s["waiting_pt"] is None:
            out["scoring.waiting_pt"] = (s["modes"]["pt"]["marginal_utility_of_traveling"],
                                         "= scoring.modes.pt.marginal_utility_of_traveling")
        if s["stuck_penalty"] is None:
            out["scoring.stuck_penalty"] = (_stuck_penalty(s),
                                            "24 h at the worst utility rate (MATSim abortedPlanScore)")
        return out

    def to_yaml(self) -> str:
        """The complete config as YAML: every key, defaults marked ``# default``,
        derived values ``# derived``. Loading it gives the same run."""
        derived = self.derived()
        when = datetime.datetime.now().isoformat(timespec="seconds")
        lines = [f"# mobslim output config, written {when}" +
                 (f" from {self.source}" if self.source else ""),
                 "# '# default': not in the input config; '# derived': computed from other values.",
                 "# Loading this file reproduces the run."]
        _emit(SCHEMA, self.values, "", 0, lines, self.defaults, derived)
        return "\n".join(lines) + "\n"

    def write_output_config(self, output_dir=None) -> Path:
        out = Path(output_dir or self.values["run"]["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        path = out / "output_config.yaml"
        path.write_text(self.to_yaml())
        return path

    # -- scenario
    def read_inputs(self):
        """Read network, plans, attributes and transit into a ``Scenario``
        (no derived person columns yet)."""
        from mobslim.formats import read_population
        from mobslim.formats.matsim_network import read_network_xml
        from mobslim.routing.teleport import TeleportedModeParams
        from mobslim.scenario import Scenario

        v = self.values
        network = read_network_xml(v["inputs"]["network"])
        teleported = {m: TeleportedModeParams(plain(p["max_speed"]), p["beeline_distance_factor"])
                      for m, p in v["modes"].items()}
        transit = None
        t = v["transit"]
        if t["schedule"] is not None:
            from mobslim.transit import TransitParams, read_transit

            transit = read_transit(t["schedule"], t["vehicles"], network,
                                   TransitParams(passenger_modes=tuple(t["passenger_modes"]),
                                                 vehicles=t["vehicles_on"],
                                                 schedule_modes=tuple(t["schedule_modes"])))
        population = read_population(v["inputs"]["plans"], network, keep_routes=True,
                                     attributes_file=v["inputs"]["person_attributes"],
                                     transit_schedule=transit)
        return Scenario(network, population, teleported_modes=teleported, transit=transit)

    def filtered_values(self):
        """``[(path, Filtered)]`` of every per-person value with rules."""
        out = []

        def walk(v, path):
            if isinstance(v, Filtered):
                if v.rules:
                    out.append((path, v))
            elif isinstance(v, dict):
                for k, x in v.items():
                    walk(x, f"{path}.{k}" if path else k)
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    walk(x, f"{path}[{i}]")

        walk(self.values, "")
        return out

    def precheck(self, scenario) -> None:
        """Check the config against the scenario; raise :class:`ConfigError`
        listing every problem."""
        problems = []
        attributes = scenario.population.attributes
        for path, f in self.filtered_values():
            for i, rule in enumerate(f.rules):
                for msg in check_condition(rule.when, attributes):
                    problems.append(f"{path}.rules[{i}].when: {msg}")
        v = self.values
        acts = v["scoring"]["activities"]
        live = scenario.population.live_batch()
        types, modes = live.act_types, live.modes
        missing = [t for t in types if t not in acts and not t.endswith(" interaction")]
        if missing:
            problems.append(f"scoring.activities: no parameters for activity type(s) "
                            f"{', '.join(map(repr, missing))} used in the plans")
        known = scenario.modes
        unknown = [m for m in modes if not known.is_known(m)]
        if unknown:
            problems.append(f"plans use leg mode(s) {', '.join(map(repr, unknown))} that are not on "
                            f"the network, not transit modes and not in 'modes' (teleported)")
        for name, strategies in v["replanning"]["strategy_sets"].items():
            for i, s in enumerate(strategies):
                if s["name"] == "change_trip_mode":
                    bad = [m for m in s["modes"] if not known.is_known(m)]
                    if bad:
                        problems.append(f"replanning.strategy_sets.{name}[{i}].modes: unknown "
                                        f"mode(s) {', '.join(map(repr, bad))}")
        explicit_paths = self._explicit_paths()
        for name in v["modes"]:
            explicit = any(k.startswith(f"modes.{name}.") for k in explicit_paths)
            if explicit and known.is_network_mode(name):
                problems.append(f"modes.{name}: '{name}' is a network mode; max_speed and "
                                "teleporting are only supported for teleported modes so far")
        if problems:
            raise ConfigError(problems, str(self.source or "mobslim config"))

    def _explicit_paths(self):
        """Leaf paths that were configured (not defaults)."""
        out = []

        def walk(v, path):
            if isinstance(v, dict) and not isinstance(v, Filtered):
                for k, x in v.items():
                    walk(x, f"{path}.{k}" if path else k)
            elif path not in self.defaults:
                out.append(path)

        walk(self.values, "")
        return out

    def build_scenario(self):
        """Read the inputs, precheck, and add the derived per-person values:
        the ``mobslim:car_availability`` attribute and per-person mode speeds."""
        from dataclasses import replace

        scenario = self.read_inputs()
        self.precheck(scenario)
        attributes = scenario.population.attributes
        car = self.values["persons"]["car_availability"]
        car_col = (car.evaluate(attributes, bool) if isinstance(car, Filtered)
                   else np.full(attributes.n_persons, bool(car)))
        population = scenario.population.with_attributes(
            attributes.with_column(CAR_AVAILABILITY, car_col))
        speeds = {m: p["max_speed"].evaluate(attributes) for m, p in self.values["modes"].items()
                  if has_rules(p["max_speed"])}
        return replace(scenario, population=population, person_max_speeds=speeds)

    # -- compiling
    def scoring_params(self):
        from mobslim.scoring import ActivityParams, ModeParams, ScoringParams

        s = self.values["scoring"]
        modes = {m: ModeParams(**{k: plain(x) for k, x in p.items()}) for m, p in s["modes"].items()}
        acts = {t: ActivityParams(**{k: plain(x) for k, x in p.items()})
                for t, p in s["activities"].items()}
        derived = self.derived()
        waiting_pt = plain(s["waiting_pt"]) if s["waiting_pt"] is not None else \
            plain(derived["scoring.waiting_pt"][0])
        stuck = plain(s["stuck_penalty"]) if s["stuck_penalty"] is not None else \
            derived["scoring.stuck_penalty"][0]
        return ScoringParams(
            performing=plain(s["performing"]), activities=acts, modes=modes,
            default_activity=None, stuck_penalty=stuck,
            marginal_utility_of_waiting_pt=waiting_pt,
            utility_of_line_switch=plain(s["utility_of_line_switch"]),
            late_arrival=plain(s["late_arrival"]), early_departure=plain(s["early_departure"]),
            waiting=plain(s["waiting"]), marginal_utility_of_money=plain(s["marginal_utility_of_money"]))

    def scoring_rules(self) -> list[tuple[str, Filtered]]:
        """``[(scoring field path, Filtered)]`` for the per-person scoring values."""
        s = self.values["scoring"]
        names = {"waiting_pt": "marginal_utility_of_waiting_pt"}
        out = []
        for key in ("performing", "late_arrival", "early_departure", "waiting", "waiting_pt",
                    "utility_of_line_switch", "marginal_utility_of_money", "stuck_penalty"):
            value = s[key]
            if key == "waiting_pt" and value is None:
                value = s["modes"]["pt"]["marginal_utility_of_traveling"]  # follows pt's rules
            if has_rules(value):
                out.append((names.get(key, key), value))
        for kind in ("modes", "activities"):
            for name, params in s[kind].items():
                for key, value in params.items():
                    if has_rules(value):
                        out.append((f"{kind}.{name}.{key}", value))
        return out

    def scorer(self):
        from mobslim.scoring import CharyparNagelScorer

        rules = self.scoring_rules()
        return CharyparNagelScorer(self.scoring_params(),
                                   person_rules=[ConfigScoringRules(rules)] if rules else ())

    def mobsim(self):
        from mobslim.mobsim import (
            AgentSim,
            AgentSimParams,
            DSim,
            DSimParams,
            HermesSim,
            HermesSimParams,
            PSim,
            PSimParams,
            QSim,
            QSimParams,
        )

        m = self.values["mobsim"]
        common = dict(flow_capacity_factor=m["flow_capacity_factor"],
                      storage_capacity_factor=m["storage_capacity_factor"],
                      stuck_time=m["stuck_time"], end_time=m["end_time"],
                      effective_cell_size=m["effective_cell_size"])
        common.update({k: m[k] for k in ("continuous_time", "junction_delays", "rounding")
                       if m[k] is not None})
        if m["type"] == "hermes":
            return HermesSim(HermesSimParams(**common, backend=m["hermes_backend"]))
        if m["type"] == "agentsim":
            return AgentSim(AgentSimParams(**common, blocked=m["agentsim_blocked"]))
        if m["type"] == "psim":
            return PSim(PSimParams(**common, backend=m["psim_backend"],
                                   threads=m["psim_threads"],
                                   partitioner=m["psim_partitioner"]))
        if m["type"] == "dsim":
            return DSim(DSimParams(**common, threads=m["dsim_threads"],
                                   partitioner=m["dsim_partitioner"],
                                   seed=self.values["run"]["seed"] % 2**64,
                                   buffer_in_storage=m["dsim_buffer_in_storage"]))
        return QSim(QSimParams(**common))

    def strategies(self, attributes=None):
        """A ``StrategyManager``, or a ``PersonStrategyManager`` if persons
        use different strategy sets (needs ``attributes``)."""
        from mobslim.replanning import PersonStrategyManager, StrategyManager

        r = self.values["replanning"]
        iterations = self.values["run"]["iterations"]
        frac = r["disable_innovation_after"]
        disable_at = None if frac is None else int(frac * (iterations - 1))
        managers = {name: StrategyManager([(s["weight"], _strategy(s)) for s in strategies],
                                          max_plans=r["max_plans"], disable_innovation_at=disable_at)
                    for name, strategies in r["strategy_sets"].items()}
        chosen = r["strategy_set"]
        if not has_rules(chosen):
            return managers[plain(chosen)]
        if attributes is None:
            raise ValueError("Per-person strategy sets need the person attributes.")
        names = list(managers)
        sets = chosen.evaluate(attributes, object)
        codes = np.array([names.index(s) for s in sets.tolist()], dtype=np.int32)
        return PersonStrategyManager([managers[n] for n in names], codes, names)

    def controller_params(self):
        from mobslim.controller import ControllerParams

        run, tt = self.values["run"], self.values["travel_time"]
        return ControllerParams(
            iterations=run["iterations"], seed=run["seed"],
            learning_rate=self.values["scoring"]["learning_rate"],
            travel_time_bin_size=tt["bin_size"], travel_time_horizon=tt["horizon"],
            output_dir=run["output_dir"], write_events_interval=run["write_events_interval"],
            activity_links=self.activity_link_params())

    def activity_link_params(self):
        """The ``activity_links`` section as :class:`~mobslim.activity_links.ActivityLinkParams`."""
        from mobslim.activity_links import (
            ActivityLinkParams,
            LinkFilter,
            SnapTier,
        )

        a = self.values["activity_links"]
        av = a["avoid"]
        tiers = ()
        if av["attribute"] is not None:
            if not av["values"] and av["min_value"] is None:
                raise ValueError("activity_links.avoid needs values or min_value with attribute.")
            tiers = (SnapTier((LinkFilter(av["attribute"], tuple(av["values"]), av["min_value"]),),
                              av["unless_beyond"]),)
        return ActivityLinkParams(modes=tuple(a["modes"]), tiers=tiers,
                                  snap_node_activities=a["snap_node_activities"])

    def transit_router_factory(self, scenario, params):
        from mobslim.replanning.context import (
            TransitRouterFactory,
            default_transit_router,
        )

        if scenario.transit is None:
            return default_transit_router
        r = self.values["routing"]["transit"]
        mode = scenario.transit.params.passenger_modes[0]
        return TransitRouterFactory.from_scoring(params, mode, **r)

    def controller_kwargs(self, scenario=None) -> dict:
        """Keyword arguments for :class:`~mobslim.controller.Controller`
        (``scenario`` defaults to :meth:`build_scenario`)."""
        if scenario is None:
            scenario = self.build_scenario()
        scorer = self.scorer()
        return dict(scenario=scenario, mobsim=self.mobsim(), scorer=scorer,
                    strategies=self.strategies(scenario.population.attributes),
                    transit_router_factory=self.transit_router_factory(scenario, scorer.params),
                    params=self.controller_params(), listeners=[OutputConfigWriter(self)])


_SELECTORS = {"change_exp_beta", "select_exp_beta", "best_score", "keep_selected", "select_random"}


def _strategy(s: dict):
    from mobslim.replanning import (
        BestScoreSelector,
        ChangeExpBetaSelector,
        ChangeTripMode,
        ExpBetaSelector,
        KeepSelectedSelector,
        PlanStrategy,
        RandomSelector,
        Reroute,
        TimeAllocationMutator,
    )

    name = s["name"]
    if name == "change_exp_beta":
        return PlanStrategy(ChangeExpBetaSelector(s["beta"]))
    if name == "select_exp_beta":
        return PlanStrategy(ExpBetaSelector(s["beta"]))
    if name == "best_score":
        return PlanStrategy(BestScoreSelector())
    if name == "keep_selected":
        return PlanStrategy(KeepSelectedSelector())
    if name == "select_random":
        return PlanStrategy(RandomSelector())
    if name == "reroute":
        return PlanStrategy(RandomSelector(), [Reroute()])
    if name == "time_allocation_mutator":
        return PlanStrategy(RandomSelector(), [TimeAllocationMutator(s["mutation_range"]), Reroute()])
    if name == "change_trip_mode":
        return PlanStrategy(RandomSelector(), [
            ChangeTripMode(s["modes"], s["ignore_car_availability"],
                           availability_attribute=CAR_AVAILABILITY), Reroute()])
    raise KeyError(name)  # pragma: no cover - the schema lists every name


def _stuck_penalty(s: dict) -> float:
    """MATSim's abortedPlanScore from the (default) betas."""
    performing = plain(s["performing"])
    late, early, waiting = plain(s["late_arrival"]), plain(s["early_departure"]), plain(s["waiting"])
    worst_travel = min([0.0] + [plain(m["marginal_utility_of_traveling"]) for m in s["modes"].values()])
    return min(min(late, early), min(worst_travel - performing, waiting - performing)) * 24.0


class ConfigScoringRules:
    """The config's per-person scoring values as a person scoring rule
    (:mod:`mobslim.scoring.person`): each field's rules in order, so the
    last matching rule wins."""

    def __init__(self, items: list[tuple[str, Filtered]]):
        self.items = list(items)

    def apply(self, builder) -> None:
        for field, f in self.items:
            for rule in f.rules:
                builder.set(field, rule.value, where=evaluate(rule.when, builder.attributes))

    def __repr__(self) -> str:
        return f"ConfigScoringRules({[f for f, _ in self.items]})"


class OutputConfigWriter(ControllerListener):
    """Logs the config's defaults and writes ``output_config.yaml`` at startup."""

    def __init__(self, config: MobslimConfig):
        self.config = config

    def on_startup(self, controller) -> None:
        self.config.log_defaults()
        out = controller.params.output_dir
        if out is not None:
            path = self.config.write_output_config(out)
            logger.info("wrote %s", path)


# ------------------------------------------------------------- YAML output

def _when_yaml(raw) -> str:
    if isinstance(raw, str):
        return _scalar_yaml(raw)
    return yaml.safe_dump(raw, default_flow_style=True, width=10_000).strip()


def _emit(spec, value, path, indent, lines, defaults, derived, key=None):
    pad = "  " * indent
    head = f"{pad}{key}:" if key is not None else None
    if isinstance(spec, Param):
        note = ""
        if path in derived:
            v, why = derived[path]
            note = f"  # derived: {why}"
            value = v
        elif path in defaults:
            note = "  # default"
        if isinstance(value, Filtered):
            lines.append(head + note)
            dnote = "  # default" if f"{path}.default" in defaults else ""
            lines.append(f"{pad}  default: {_scalar_yaml(value.default)}{dnote}")
            lines.append(f"{pad}  rules:")
            for rule in value.rules:
                lines.append(f"{pad}    - when: {_when_yaml(rule.raw_when)}")
                lines.append(f"{pad}      set: {_scalar_yaml(rule.value)}")
            if not value.rules:
                lines[-1] += " []"
        else:
            lines.append(f"{head} {_scalar_yaml(value)}{note}")
        return
    if isinstance(spec, Section):
        if head is not None:
            lines.append(head)
        for k, s in spec.fields.items():
            _emit(s, value[k], f"{path}.{k}" if path else k, indent + (head is not None), lines,
                  defaults, derived, k)
        return
    if isinstance(spec, MapOf):
        lines.append(head + (" {}" if not value else ""))
        for k, v in value.items():
            _emit(spec.item, v, f"{path}.{k}", indent + 1, lines, defaults, derived, _scalar_yaml(k))
        return
    # a strategy list
    lines.append(head + ("  # default" if path in defaults else ""))
    for i, s in enumerate(value):
        sp = STRATEGIES[s["name"]]
        first = True
        for k, p in sp.fields.items():
            ip = f"{path}[{i}].{k}"
            note = "  # default" if ip in defaults and path not in defaults else ""
            prefix = f"{pad}  - " if first else f"{pad}    "
            lines.append(f"{prefix}{k}: {_scalar_yaml(s[k])}{note}")
            first = False
