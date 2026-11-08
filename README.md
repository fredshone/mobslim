# MobSlim

A modular, MATSim-inspired, agent-based transport simulation in Python.

MobSlim follows [MATSim](https://matsim.org)'s co-evolutionary loop
(simulate, score, replan) and its queue-model traffic dynamics, but is
organised as a set of small, swappable components. It is designed as a
workbench for experiments: compiled (Rust) replacements for heavy modules,
parallel optimisation, and learning-based replanning.

* [Installation](#installation)
* [Quickstart](#quickstart)
* [Command line](#command-line)
* [Configuration](#configuration)
  * [The mobslim config](#the-mobslim-config)
  * [The MATSim config](#the-matsim-config)
* [Documentation](#documentation)

## Installation

MobSlim uses [uv](https://docs.astral.sh/uv/) and Python 3.12+:

```bash
git clone https://github.com/fredshone/mobslim.git
cd mobslim
uv sync                                   # pure Python
uv run --with jupyter jupyter lab         # demos are in demos/
```

**Optional compiled kernels.** The kernels of `HermesSim`
(`rust/mobslim-hermes`) and `PSim` (`rust/mobslim-psim`) and the routers (`rust/mobslim-router`: Dijkstra/ALT and RAPTOR, plus activity-to-link snapping) are Rust
extensions. They give the same results as the Python code, only faster.
`DSim` (`rust/mobslim-dsim`, matsim-rust's parallel design, for
comparison) exists only as a Rust kernel; its METIS partitioning needs the
`metis` extra (`pymetis`).
They need [cargo](https://rustup.rs):

```bash
uv sync --group rust                      # builds and installs them
```

Later `uv run` commands keep them; a plain `uv sync` removes them again, so
use `uv sync --group rust` from then on.

**Optional Parquet support** (fast population files): `uv sync --extra parquet`.

## Quickstart

```python
import logging
from mobslim import Controller, ControllerParams, Scenario, read_network_xml, read_population_xml

logging.basicConfig(level=logging.INFO, format="%(message)s")

network = read_network_xml("scenarios/equil/network.xml")
population = read_population_xml("scenarios/equil/plans100.xml", network)

result = Controller(Scenario(network, population), params=ControllerParams(iterations=20)).run()
print(result.metrics_frame()[["score_executed", "score_best", "car_travel_time"]])
```

```
it 0: score executed=68.263 best=68.263 stuck=0 car tt=1695s
it 1: score executed=68.339 best=68.396 stuck=0 car tt=1668s
...
it 19: score executed=69.966 best=71.956 stuck=0 car tt=1613s
```

MobSlim logs its progress to the `mobslim` logger, so without
`logging.basicConfig` a run is silent.

Demos: `demos/equil.ipynb` (MATSim's equil test case, with animation),
`demos/grid.ipynb` (route choice on a grid, with a custom strategy mix) and
`demos/pt_simple.ipynb` (transit: RAPTOR routing, timetable and network vehicles).

## Command line

Installing MobSlim adds a `mobslim` command (also `python -m mobslim`):

```bash
uv run mobslim run scenarios/equil/mobslim.yaml            # run a mobslim config
uv run mobslim run scenarios/pt-simple/config.xml          # run a MATSim config
uv run mobslim run config.yaml -n 5 -o output/test         # override iterations / output directory
uv run mobslim check config.yaml                           # load and precheck against the inputs
uv run mobslim resolved config.yaml                        # print the complete config, with defaults
uv run mobslim reference                                   # every parameter: type, default, MATSim name
uv run mobslim from-matsim config.xml -o config.yaml       # translate a MATSim config
uv run mobslim convert plans.xml.gz plans_pq --network network.xml   # population XML <-> Parquet
uv run mobslim info plans_pq --network network.xml         # summarise a population file
```

A run writes `metrics.csv` (one row per iteration), `ITERS/it.N/events.xml.gz`
(MATSim events) and, for a mobslim config, `output_config.yaml` to the
output directory.

## Configuration

A run can be configured in one of three ways. All three build the same
`Controller` (scenario, mobsim, scorer, replanning strategies, loop
settings), so they differ only in how you write it down:

| | how | use it for |
|---|---|---|
| **Python** | `Controller(scenario, mobsim=..., scorer=..., strategies=...)` | experiments and custom components (your own scorer, strategy or mobsim) |
| **mobslim config** | `Controller.from_mobslim_config("config.yaml")` | runs described in one YAML file, with per-person parameters |
| **MATSim config** | `Controller.from_matsim_config("config.xml")` | running an existing MATSim scenario as it is |

A run uses either a mobslim config or a MATSim config, never both: there is
no layering of one on top of the other. To move a MATSim scenario to a
mobslim config, translate it once (see below); the translation runs
identically.

### The mobslim config

One YAML file configures the whole run. Its sections mirror MATSim's
modules:

| section | configures | MATSim module |
|---|---|---|
| `run` | iterations, seed, output directory, events output | `controller`, `global` |
| `inputs` | network, plans and person attribute files (paths relative to the config) | `network`, `plans` |
| `transit` | transit schedule and vehicles, passenger modes | `transit` |
| `modes` | teleported modes: `max_speed`, `beeline_distance_factor` | `routing` |
| `mobsim` | `qsim`, `hermes`, `agentsim`, `psim` or `dsim`, capacity factors, stuck time, end time, clock (`continuous_time`, `junction_delays`, `rounding`) | `qsim`, `hermes` |
| `travel_time` | travel-time bins learned between iterations | `travelTimeCalculator` |
| `routing` | transit router search radii and transfers | `swissRailRaptor` |
| `scoring` | utilities, per mode and per activity type | `scoring` |
| `replanning` | strategy sets, plan memory, when innovation stops | `replanning` |
| `persons` | person-level values that are not scoring (`car_availability`) | person attributes |

Only what differs from the defaults needs to be written. A minimal config
for MATSim's equil scenario (`scenarios/equil/mobslim.yaml` is a fuller one):

```yaml
inputs:
  network: network.xml
  plans: plans100.xml
scoring:
  activities:
    h: {typical_duration: "12:00:00"}
    w: {typical_duration: "08:00:00", opening_time: "07:00:00"}
```

```python
from mobslim import Controller

result = Controller.from_mobslim_config("scenarios/equil/mobslim.yaml").run()
```

Times are seconds or `"HH:MM:SS"`, utilities are per hour, speeds in m/s.

**Per-person values.** Scoring utilities, mode speeds
(`modes.<mode>.max_speed`), `persons.car_availability` and the strategy set
can differ by person. Instead of a plain value, give a `default` and
`rules` that test person attributes. Rules apply in order and **the last
matching rule wins**:

```yaml
scoring:
  modes:
    walk:
      marginal_utility_of_traveling:
        default: -10
        rules:
          - {when: age > 50, set: -15}
          - {when: age > 70, set: -30}      # a 75-year-old gets -30
modes:
  walk:
    max_speed: {default: 1.2, rules: [{when: age > 70, set: 0.8}]}
persons:
  car_availability: {default: true, rules: [{when: has_license == False, set: false}]}
```

Conditions are written as Python-style strings (`age > 50`,
`18 <= age < 65`, `has_license == False and age >= 18`,
`subpopulation in ["freight", "taxi"]`) or in a structured form
(`{age: {gt: 50}}`, `{all: [...]}`, `{any: [...]}`, `{not: ...}`). Strings
are parsed, never executed. A person missing an attribute matches no
comparison. MATSim's subpopulations are simply rules on the
`subpopulation` attribute.

**Strategy sets.** Replanning strategies come in named, weighted sets, each
strategy with its own parameters, and `strategy_set` picks each person's set:

```yaml
replanning:
  strategy_set:
    default: default
    rules: [{when: subpopulation == "freight", set: freight}]
  strategy_sets:
    default:
      - {name: change_exp_beta, weight: 0.8}
      - {name: change_trip_mode, weight: 0.2, modes: [car, pt, walk], ignore_car_availability: false}
    freight:
      - {name: best_score, weight: 0.9}
      - {name: reroute, weight: 0.1}
```

**Defaults and the output config.** Every parameter has MATSim's default.
Each default used is logged, and each run writes
`<output_dir>/output_config.yaml`: the complete config, every default
marked `# default`. Loading that file reproduces the run.

**Precheck.** Before the run starts, the config is checked against itself
and the scenario. One error lists every problem found:

```
config.yaml: 3 problem(s):
  scoring.performin: unknown parameter; did you mean 'performing'?
  scoring.modes.walk.marginal_utility_of_traveling.rules[0].when: no person attribute 'agee' (attributes are: age, has_license, subpopulation)
  scoring.activities: no parameters for activity type(s) 'shop' used in the plans
```

**Command line.** `mobslim check`, `resolved`, `reference` and
`from-matsim` (see [Command line](#command-line)) work on these configs.

### The MATSim config

`Controller.from_matsim_config("config.xml")` reads a MATSim `config.xml`
and the files it names (paths relative to the config, as in MATSim). It
translates the `controller`, `qsim`/`hermes`, `scoring`, `replanning`,
`routing` and `transit` modules. Params that mobslim cannot honour are
logged as a warning, or raise `UnsupportedConfigError` with `strict=True`.
`MatsimConfig` loads, edits and writes a config from Python.

`mobslim from-matsim` translates a MATSim config into a mobslim config.
Subpopulation scoring sets become rules, per-subpopulation strategies
become strategy sets, and `changeMode.ignoreCarAvailability=false` becomes
`persons.car_availability` rules. MATSim params that can't be honoured are
listed in a comment at the top of the file.

## Documentation

* [Developer guide](docs/developer.md): architecture, data model, package
  layout, conventions, a tour of the Python API, extending MobSlim, testing,
  benchmarks and verification against MATSim.
* [Design](docs/design.md): the reasoning behind the design, and how
  MobSlim differs from MATSim.
* [Design detail](docs/simulation/design-detail.md): how the five mobsims differ from
  each other and from their MATSim equivalents, in plain terms.
* API reference: build it with
  `uv run --group docs sphinx-build docs docs/_build`, then open
  `docs/_build/index.html`.

### A note on names

MobSlim keeps MATSim's names so MATSim users feel at home:

| name | what it is |
|---|---|
| equil | MATSim's classic small test scenario (`scenarios/equil`) |
| QSim | the queue-model traffic simulation (MATSim's default mobsim) |
| HermesSim | a faster queue model on a whole-second clock (after MATSim's Hermes) |
| PSim | a whole-second queue model on several threads, same results for any number of threads (MobSlim's own) |
| DSim | matsim-rust's parallel design (workers per network partition, neighbour messages) and queue dynamics, reimplemented to compare with PSim |
| AgentSim | the same queue model, written agent by agent (MobSlim's own) |
| Charypar-Nagel scoring | MATSim's default utility function: activities earn, travel costs |
| ExpBeta / ChangeExpBeta | choose among remembered plans with logit probabilities over their scores |
| Reroute, TimeAllocationMutator, ChangeTripMode | innovation: new route, shifted end times, new mode |
| RAPTOR | the transit router (as MATSim's SwissRailRaptor) |
| `seed=4711` | MATSim's default random seed |

How the five mobsims differ from each other and from MATSim, in plain
terms, is in [`docs/simulation/design-detail.md`](docs/simulation/design-detail.md).
