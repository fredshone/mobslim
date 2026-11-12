"""The mobslim config schema: every section, parameter, type and default.

The tree below drives loading (types, ranges, defaults), the precheck
(unknown keys), the output config and the parameter reference
(:func:`reference_markdown`). Defaults are MATSim's, so a config and its
MATSim equivalent run the same.

Spec kinds:

* :class:`Param`: one value. ``filterable`` params may also be given as
  ``{default: ..., rules: [{when: ..., set: ...}]}`` (see
  :mod:`mobslim.config.conditions`).
* :class:`Section`: a mapping with fixed keys.
* :class:`MapOf`: a mapping with free keys (mode names, activity types,
  strategy set names), each an ``item``; ``defaults`` are entries present
  unless overridden (a configured entry falls back to its default entry
  field by field).
* :class:`StrategyList`: a list of strategies, each picked by ``name``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

from mobslim.routing.traveltime import (
    DEFAULT_HORIZON,
    DEFAULT_OFF_PEAK_BIN_SIZE,
    DEFAULT_PEAK,
)


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED = _Required()

# value types: float int bool str time path strlist
TYPES = ("float", "int", "bool", "str", "time", "path", "strlist")


@dataclass(frozen=True)
class Param:
    type: str
    default: Any = REQUIRED
    doc: str = ""
    nullable: bool = False
    filterable: bool = False
    choices: tuple = ()
    min: Optional[float] = None
    min_exclusive: bool = False
    matsim: str = ""  # the MATSim param it corresponds to
    null_means: str = ""  # what null stands for (derived values)

    def __post_init__(self):
        assert self.type in TYPES, self.type


@dataclass(frozen=True)
class Section:
    fields: Mapping[str, "Spec"]
    doc: str = ""


@dataclass(frozen=True)
class MapOf:
    item: "Spec"
    defaults: Mapping[str, Any] = field(default_factory=dict)
    doc: str = ""
    key: str = "name"


@dataclass(frozen=True)
class StrategyList:
    doc: str = ""


Spec = Union[Param, Section, MapOf, StrategyList]


# ------------------------------------------------------------------ scoring

def _util(default, doc, matsim, **kw):
    return Param("float", default, doc, filterable=True, matsim=matsim, **kw)


SCORING_MODE = Section({
    "constant": _util(0.0, "utility per leg", "constant"),
    "marginal_utility_of_traveling": _util(-6.0, "util/h of travel time",
                                           "marginalUtilityOfTraveling_util_hr"),
    "marginal_utility_of_distance": _util(0.0, "util/m", "marginalUtilityOfDistance_util_m"),
    "monetary_distance_rate": _util(0.0, "money/m (times marginal_utility_of_money)",
                                    "monetaryDistanceRate"),
    "daily_constant": _util(0.0, "util, once per day if the mode is used", "dailyUtilityConstant"),
    "daily_monetary_constant": _util(0.0, "money, once per day if the mode is used",
                                     "dailyMonetaryConstant"),
}, "Utility parameters of one leg mode (MATSim modeParams).")

MATSIM_DEFAULT_MODES = ("car", "pt", "walk", "bike", "ride", "other")


def _time(doc, matsim, default=None, **kw):
    return Param("time", default, doc, nullable=default is None, filterable=True, matsim=matsim, **kw)


SCORING_ACTIVITY = Section({
    "typical_duration": Param("time", REQUIRED, "typical duration", filterable=True, min=0,
                              min_exclusive=True, matsim="typicalDuration"),
    "priority": Param("float", 1.0, "priority", filterable=True, min=0, min_exclusive=True,
                      matsim="priority"),
    "zero_utility": Param("str", "relative", "how t_0 is computed", choices=("relative", "uniform"),
                          matsim="typicalDurationScoreComputation"),
    "opening_time": _time("opening time (null: always open)", "openingTime"),
    "closing_time": _time("closing time (null: never closes)", "closingTime"),
    "latest_start_time": _time("later starts pay late_arrival", "latestStartTime"),
    "earliest_end_time": _time("earlier ends pay early_departure", "earliestEndTime"),
    "minimal_duration": _time("shorter stays pay early_departure", "minimalDuration"),
}, "Utility parameters of one activity type (MATSim activityParams).")

SCORING = Section({
    "learning_rate": Param("float", 1.0, "score averaging: new = (1-lr) old + lr observed",
                           min=0, min_exclusive=True, matsim="learningRate"),
    "performing": _util(6.0, "util/h of performing an activity", "performing"),
    "late_arrival": _util(-18.0, "util/h after latest_start_time", "lateArrival"),
    "early_departure": _util(0.0, "util/h before earliest_end_time / minimal_duration",
                             "earlyDeparture"),
    "waiting": _util(0.0, "util/h at a closed activity", "waiting"),
    "waiting_pt": _util(None, "util/h waiting for pt", "waitingPt", nullable=True,
                        null_means="pt's marginal_utility_of_traveling"),
    "utility_of_line_switch": _util(-1.0, "util per pt line switch", "utilityOfLineSwitch"),
    "marginal_utility_of_money": _util(1.0, "util per money unit", "marginalUtilityOfMoney"),
    "stuck_penalty": _util(None, "util for a person who got stuck", "abortedPlanScore",
                           nullable=True,
                           null_means="24 h at the worst of late_arrival, early_departure, "
                                      "waiting and travel, minus performing (MATSim)"),
    "modes": MapOf(SCORING_MODE, {m: {} for m in MATSIM_DEFAULT_MODES},
                   "Per leg mode. MATSim's six default modes are always present.", key="mode"),
    "activities": MapOf(SCORING_ACTIVITY, {},
                        "Per activity type. A type in the plans without an entry is an error.",
                        key="type"),
}, "Charypar-Nagel scoring (MATSim scoring module).")


# --------------------------------------------------------------- replanning

def _strategy(**params):
    return Section({"name": Param("str", REQUIRED, "strategy name"),
                    "weight": Param("float", REQUIRED, "relative weight", min=0),
                    **params})


_BETA = Param("float", 1.0, "logit scale on scores", min=0, matsim="scoring.brainExpBeta")

STRATEGIES: dict[str, Section] = {
    "change_exp_beta": _strategy(beta=_BETA),
    "select_exp_beta": _strategy(beta=_BETA),
    "best_score": _strategy(),
    "keep_selected": _strategy(),
    "select_random": _strategy(),
    "reroute": _strategy(),
    "time_allocation_mutator": _strategy(
        mutation_range=Param("time", 1800.0, "end times shift by U(-range, range)", min=0,
                             matsim="timeAllocationMutator.mutationRange")),
    "change_trip_mode": _strategy(
        modes=Param("strlist", REQUIRED, "modes to choose from (at least two)",
                    matsim="changeMode.modes"),
        ignore_car_availability=Param("bool", True, "false: respect persons.car_availability",
                                      matsim="changeMode.ignoreCarAvailability")),
}
STRATEGY_DOCS = {
    "change_exp_beta": "switch plans by score differences (MATSim ChangeExpBeta)",
    "select_exp_beta": "pick a plan with P ~ exp(beta * score) (SelectExpBeta)",
    "best_score": "pick the best-scored plan (BestScore)",
    "keep_selected": "keep the selected plan (KeepLastSelected)",
    "select_random": "pick a random plan (SelectRandom)",
    "reroute": "copy a random plan and reroute it (ReRoute)",
    "time_allocation_mutator": "copy a random plan, mutate end times, reroute",
    "change_trip_mode": "copy a random plan, change its mode, reroute (ChangeTripMode)",
}

DEFAULT_STRATEGY_SET = [
    {"name": "change_exp_beta", "weight": 0.8},
    {"name": "reroute", "weight": 0.1},
    {"name": "time_allocation_mutator", "weight": 0.1},
]

REPLANNING = Section({
    "max_plans": Param("int", 5, "plans kept per person", min=1, matsim="maxAgentPlanMemorySize"),
    "disable_innovation_after": Param(
        "float", None, "fraction of iterations after which innovation stops (null: never)",
        nullable=True, min=0, matsim="fractionOfIterationsToDisableInnovation"),
    "strategy_set": Param("str", "default", "the strategy set a person uses", filterable=True),
    "strategy_sets": MapOf(StrategyList(), {"default": DEFAULT_STRATEGY_SET},
                           "Named, weighted strategy lists.", key="set"),
}, "Replanning: which strategies persons use (MATSim replanning module).")


# ---------------------------------------------------------------- the rest

MODE = Section({
    "max_speed": Param("float", REQUIRED, "m/s; for a teleported mode, its speed", filterable=True,
                       min=0, min_exclusive=True, matsim="teleportedModeSpeed"),
    "beeline_distance_factor": Param("float", 1.3, "teleported distance = beeline x this",
                                     min=0, min_exclusive=True, matsim="beelineDistanceFactor"),
}, "A mode that is not on the network: teleported (MATSim teleportedModeParameters).")

SCHEMA = Section({
    "version": Param("int", 1, "config format version", choices=(1,)),
    "run": Section({
        "iterations": Param("int", 11, "mobsim runs (MATSim: lastIteration - firstIteration + 1)",
                            min=1, matsim="controller.lastIteration"),
        "seed": Param("int", 4711, "random seed", matsim="global.randomSeed"),
        "output_dir": Param("path", "output", "where outputs go (relative to this file)",
                            matsim="controller.outputDirectory"),
        "write_events_interval": Param("int", 10, "events every N iterations (0: last only)",
                                       min=0, matsim="controller.writeEventsInterval"),
        "link_times_interval": Param("int", 0, "record the persons' link entry times every N "
                                     "iterations (0: last only; others record how far each "
                                     "leg got)", min=0),
    }, "The iteration loop (MATSim controller and global modules)."),
    "inputs": Section({
        "network": Param("path", REQUIRED, "MATSim network XML", matsim="network.inputNetworkFile"),
        "plans": Param("path", REQUIRED, "plans: MATSim XML (*.xml, *.xml.gz) or a Parquet "
                       "population directory", matsim="plans.inputPlansFile"),
        "person_attributes": Param("path", None, "extra person attributes: objectAttributes XML "
                                   "or a Parquet persons table", nullable=True,
                                   matsim="plans.inputPersonAttributesFile"),
    }, "Input files (paths relative to this file)."),
    "transit": Section({
        "schedule": Param("path", None, "transit schedule XML (null: no transit)", nullable=True,
                          matsim="transit.transitScheduleFile"),
        "vehicles": Param("path", None, "transit vehicles XML", nullable=True,
                          matsim="transit.vehiclesFile"),
        "passenger_modes": Param("strlist", ["pt"], "leg modes that ride transit",
                                 matsim="transit.transitModes"),
        "vehicles_on": Param("str", "network", "transit vehicles drive the network or keep the "
                             "schedule", choices=("network", "schedule")),
        "schedule_modes": Param("strlist", [], "route modes (e.g. rail) whose vehicles keep the "
                                "schedule even with vehicles_on: network (MATSim's deterministic pt)",
                                matsim="SBBPt.deterministicServiceModes"),
    }, "Scheduled transit (MATSim transit module)."),
    "modes": MapOf(MODE, {"walk": {"max_speed": 3 / 3.6}, "bike": {"max_speed": 15 / 3.6}},
                   "Teleported modes. walk and bike are always present (MATSim's defaults).",
                   key="mode"),
    "mobsim": Section({
        "type": Param("str", "qsim", "the mobsim", choices=("qsim", "hermes", "agentsim", "psim", "dsim"),
                      matsim="controller.mobsim"),
        "flow_capacity_factor": Param("float", 1.0, "link flow capacity scale", min=0,
                                      min_exclusive=True, matsim="flowCapacityFactor"),
        "storage_capacity_factor": Param("float", 1.0, "link storage scale", min=0,
                                         min_exclusive=True, matsim="storageCapacityFactor"),
        "stuck_time": Param("float", 11.0, "s a blocked vehicle waits before it is forced on "
                            "(MATSim's stuckTime 10 plus its one-second step)", min=0,
                            matsim="stuckTime + timeStepSize"),
        "end_time": Param("time", None, "stop the day here (null: when everyone is done)",
                          nullable=True, min=0, matsim="endTime"),
        "effective_cell_size": Param("float", 7.5, "m of road per vehicle", min=0,
                                     min_exclusive=True),
        "continuous_time": Param("bool", None, "exact times (true) or whole seconds (false); "
                                 "null: the mobsim's own (qsim, agentsim: true; others: false)",
                                 nullable=True),
        "junction_delays": Param("bool", None, "a vehicle crosses into its next link one "
                                 "second after it could leave, and an activity starts one "
                                 "second after the arrival, as MATSim steps; null: the "
                                 "mobsim's own (dsim: true; others: false)", nullable=True),
        "rounding": Param("str", None, "with whole seconds, road and teleport times round to "
                          "the nearest second or down; null: the mobsim's own (dsim: down; "
                          "others: nearest)", nullable=True, choices=("nearest", "down")),
        "hermes_backend": Param("str", "auto", "hermes only: kernel",
                                choices=("auto", "rust", "python")),
        "agentsim_blocked": Param("str", "poll", "agentsim only: blocked agents",
                                  choices=("poll", "chain")),
        "psim_backend": Param("str", "auto", "psim only: kernel",
                                  choices=("auto", "rust", "python")),
        "psim_threads": Param("int", None, "psim only: threads (null: one per CPU); "
                                  "results do not depend on it", nullable=True, min=1),
        "psim_partitioner": Param("str", "morton", "psim only: how nodes are split "
                                      "between threads", choices=("morton", "metis")),
        "dsim_threads": Param("int", None, "dsim only: worker threads (null: one per CPU); "
                               "trajectories do not depend on it", nullable=True, min=1),
        "dsim_partitioner": Param("str", "metis", "dsim only: how nodes are split between "
                                   "workers (metis needs pymetis)", choices=("metis", "morton")),
        "dsim_buffer_in_storage": Param("bool", True, "dsim only: a vehicle waiting to "
                                         "cross a node still fills its link (false: it frees "
                                         "its space, as matsim-rust)"),
    }, "The mobility simulation (MATSim qsim / hermes modules)."),
    "travel_time": Section({
        "bin_size": Param("time", 900.0, "s per travel-time bin", min=0, min_exclusive=True,
                          matsim="travelTimeCalculator.traveltimeBinSize"),
        "horizon": Param("time", DEFAULT_HORIZON, "s covered by the travel-time table", min=0,
                         min_exclusive=True),
        "off_peak_bin_size": Param("time", DEFAULT_OFF_PEAK_BIN_SIZE,
                                   "s per bin before peak_start and after peak_end (a "
                                   "multiple of bin_size; equal to it: uniform bins, as "
                                   "MATSim)", min=0, min_exclusive=True),
        "peak_start": Param("time", DEFAULT_PEAK[0], "start of the bins of bin_size", min=0),
        "peak_end": Param("time", DEFAULT_PEAK[1], "end of the bins of bin_size", min=0),
    }, "Travel-time learning between iterations."),
    "activity_links": Section({
        "modes": Param("strlist", ["car"], "activities are assigned to links that allow one of "
                       "these modes (none allowed by any link: every link)",
                       matsim="XY2Links on the car network"),
        "snap_node_activities": Param("bool", False, "also assign activities placed at a node "
                                      "without coordinates (else they stay on their node)"),
        "avoid": Section({
            "attribute": Param("str", None, "link attribute to match, e.g. type or "
                               "osm:way:highway (or length, freespeed, capacity, lanes); "
                               "null: avoid nothing", nullable=True),
            "values": Param("strlist", [], "avoid links whose attribute is one of these"),
            "min_value": Param("float", None, "avoid links whose (numeric) attribute is at "
                               "least this", nullable=True),
            "unless_beyond": Param("float", 1000.0, "m: use avoided links only if no other "
                                   "link is this close", min=0),
        }, "Links to keep activities off, e.g. motorways."),
    }, "Assigning activities to the link with the nearest midpoint before the first "
       "iteration (activities with a link in the plans keep it)."),
    "routing": Section({
        "transit": Section({
            "search_radius": Param("float", 1000.0, "m around origin/destination for stops", min=0),
            "extension_radius": Param("float", 200.0, "m added beyond the nearest stop", min=0),
            "max_transfer_distance": Param("float", 100.0, "m of walk between stops", min=0),
            "min_transfer_time": Param("time", 0.0, "s per transfer", min=0),
        }, "Transit router (RAPTOR); utilities come from scoring."),
    }, "Routing."),
    "scoring": SCORING,
    "replanning": REPLANNING,
    "weave": Section({
        "strands": Param("int", None, "simulations per iteration, each with a different cohort "
                         "innovating (null: no weaving, the plain controller)", nullable=True, min=1),
        "innovation_rate": Param("float", None, "share of persons innovating in each strand",
                                 nullable=True, min=0,
                                 null_means="the innovative strategies' share of the weights"),
        "alpha": Param("float", 1.0, "who innovates: weight (1 + iterations since the last "
                       "innovation) ** alpha (0: uniform random)", min=0),
        "reducer": Param("str", "mean", "how a plan's scores from several strands combine",
                         choices=("mean", "precision")),
        "workers": Param("int", None, "worker processes; strand 0 runs in the main process "
                         "(0: every strand in the main process)", nullable=True, min=0,
                         null_means="strands - 1, at most the physical cores - 1"),
        "travel_time_smoothing": Param("float", 1.0, "weight of each iteration's travel times "
                                       "in the learned estimate (1: the latest only)", min=0,
                                       min_exclusive=True),
    }, "Weaving: parallel strands of innovation per iteration (mobslim.weave)."),
    "persons": Section({
        "car_availability": Param("bool", True, "whether the person has a car (read by "
                                  "change_trip_mode with ignore_car_availability: false)",
                                  filterable=True, matsim="carAvail / hasLicense attributes"),
    }, "Person-level parameters that are not scoring."),
}, "A mobslim run.")


def spec_at(path: str) -> Spec:
    """The spec of a dotted path (map entries by any key)."""
    spec: Spec = SCHEMA
    for part in path.split("."):
        if isinstance(spec, Section):
            spec = spec.fields[part]
        elif isinstance(spec, MapOf):
            spec = spec.item
        else:
            raise KeyError(path)
    return spec


def reference_markdown() -> str:
    """A Markdown table of every parameter: path, type, default, filterable, MATSim name."""
    lines = ["| parameter | type | default | per person | MATSim | description |",
             "|---|---|---|---|---|---|"]

    def fmt(v):
        if v is REQUIRED:
            return "**required**"
        return "null" if v is None else f"`{v}`"

    def walk(spec, path):
        if isinstance(spec, Param):
            default = fmt(spec.default) + (f" ({spec.null_means})" if spec.null_means else "")
            choices = f" ({' / '.join(map(str, spec.choices))})" if spec.choices else ""
            lines.append(f"| `{path}` | {spec.type}{choices} | {default} | "
                         f"{'yes' if spec.filterable else ''} | {spec.matsim} | {spec.doc} |")
        elif isinstance(spec, Section):
            for k, s in spec.fields.items():
                walk(s, f"{path}.{k}" if path else k)
        elif isinstance(spec, MapOf):
            walk(spec.item, f"{path}.<{spec.key}>")
        else:
            for name, s in STRATEGIES.items():
                lines.append(f"| `{path}[] name: {name}` | strategy | | | | {STRATEGY_DOCS[name]} |")
                for k, p in s.fields.items():
                    if k not in ("name", "weight"):
                        walk(p, f"{path}[{name}].{k}")

    walk(SCHEMA, "")
    return "\n".join(lines) + "\n"
