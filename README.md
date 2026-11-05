# MobSlim

A modular, MATSim-inspired, agent-based transport simulation in Python.

MobSlim follows [MATSim](https://matsim.org)'s co-evolutionary loop
(simulate, score, replan) and its queue-model traffic dynamics, but is
organised as a set of small, swappable components. It is designed as a
workbench for experiments: compiled (Rust) replacements for heavy modules,
parallel optimisation, and learning-based replanning.

* [Installation](#installation)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
  * [The mobslim config](#the-mobslim-config)
  * [The MATSim config](#the-matsim-config)
* [Developer guide](#developer-guide)
  * [Architecture](#architecture)
  * [Package layout](#package-layout)
  * [Conventions](#conventions)
  * [API tour](#api-tour)
  * [Extending MobSlim](#extending-mobslim)
  * [Testing](#testing)
  * [Verification against MATSim](#verification-against-matsim)
* Design rationale: [DESIGN.md](DESIGN.md)

## Installation

MobSlim uses [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:fredshone/mobslim.git
cd mobslim
uv sync
uv run --with jupyter jupyter lab   # demos are in demos/
```

`uv sync` also builds the Rust extensions, `HermesSim`'s kernel
(`rust/mobslim-hermes`) and the routers (`rust/mobslim-router`: Dijkstra/ALT and RAPTOR), which need
[cargo](https://rustup.rs). Without cargo, use `uv sync --no-group rust`;
`HermesSim` and routing then run in Python (same results, slower).

## Quickstart

```python
import logging
from mobslim import Controller, ControllerConfig, Scenario, read_network_xml, read_population_xml

logging.basicConfig(level=logging.INFO, format="%(message)s")

network = read_network_xml("scenarios/equil/network.xml")
population = read_population_xml("scenarios/equil/plans100.xml", network)

result = Controller(Scenario(network, population), config=ControllerConfig(iterations=20)).run()
print(result.metrics_frame()[["score_executed", "score_best", "car_travel_time"]])
```

```
it 0: score executed=68.263 best=68.263 stuck=0 car tt=1695s
...
it 9: score executed=68.858 best=70.095 stuck=0 car tt=1617s
```

Demos: `demos/equil.ipynb` (MATSim's equil test case, with animation),
`demos/grid.ipynb` (route choice on a grid, with a custom strategy mix) and
`demos/pt_simple.ipynb` (transit: RAPTOR routing, timetable and network vehicles).

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
| `mobsim` | `qsim`, `hermes` or `agentsim`, capacity factors, stuck time, end time | `qsim`, `hermes` |
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

**Command line.**

```bash
uv run python -m mobslim.config check config.yaml       # load and precheck against the inputs
uv run python -m mobslim.config resolved config.yaml    # print the complete config, with defaults
uv run python -m mobslim.config reference               # every parameter: type, default, MATSim name
uv run python -m mobslim.config from-matsim config.xml -o config.yaml   # translate a MATSim config
```

### The MATSim config

`Controller.from_matsim_config("config.xml")` reads a MATSim `config.xml`
and the files it names (paths relative to the config, as in MATSim). It
translates the `controller`, `qsim`/`hermes`, `scoring`, `replanning`,
`routing` and `transit` modules. Params that mobslim cannot honour are
logged as a warning, or raise `UnsupportedConfigError` with `strict=True`.
`MatsimConfig` loads, edits and writes a config from Python.

`from-matsim` (above) translates a MATSim config into a mobslim config.
Subpopulation scoring sets become rules, per-subpopulation strategies
become strategy sets, and `changeMode.ignoreCarAvailability=false` becomes
`persons.car_availability` rules. MATSim params that can't be honoured are
listed in a comment at the top of the file.

---

## Developer guide

### Architecture

Each box is a component behind a small interface (a `typing.Protocol`),
and each arrow is plain, immutable data. Replace any box with any object that
has the same methods.

```text
       Network                      Population
  (arrays, int ids)              (persons → plans)
          │                              │
          └──────────────┬───────────────┘
                         ▼
                     Scenario
                         │
                         ▼
          route_scenario (once: free-speed routes)
                         │
┌─ Controller · repeat each iteration ──────────────────────────┐
│                        ▼                                      │
│     ┌───────────────────────────────────┐                     │
│ ┌──►│ 1. Replanning       (it ≥ 1)      │◄── ReplanningContext│
│ │   │    StrategyManager                │    travel times +   │
│ │   │    on a ReplanningRunner          │    Router (Dijkstra)│
│ │   └─────────────────┬─────────────────┘          ▲          │
│ │                     │ selected plans             │          │
│ │                     ▼                            │          │
│ │   ┌───────────────────────────────────┐          │          │
│ │   │ 2. Mobsim (QSim) on a PlanBatch   │          │          │
│ │   └─────────────────┬─────────────────┘          │          │
│ │                     │ Trajectories (arrays)      │          │
│ │                     │ + events, only for         │          │
│ │                     │   registered handlers      │          │
│ │           ┌─────────┴─────────┐                  │          │
│ │           ▼                   ▼                  │          │
│ │   ┌───────────────┐   travel_time_table          │          │
│ │   │ 3. Scorer     │   (numpy) ───────────────────┘          │
│ │   │ (vectorised)  │   TravelTimeTable                       │
│ │   └───────┬───────┘                                         │
│ │           └─────────┐                                       │
│ │                     │ scores → plans in memory              │
│ │                     ▼                                       │
│ └──────────── 4. metrics, next iteration                      │
│                       │                                       │
└───────────────────────┼───────────────────────────────────────┘
                        ▼
          RunResult (scenario, travel times, metrics)

  ControllerListeners hook in at every stage:
  startup · iteration_start · after_replanning · before_mobsim ·
  after_mobsim · scoring · iteration_end · shutdown
```

What one iteration does (`Controller.run`):

| Step | Component (interface) | In → out |
|---|---|---|
| 0. prepare (once) | `route_scenario` | un-routed legs → routed at free speed |
| 1. replanning (it ≥ 1) | `StrategyManager` on a `ReplanningRunner` | `Population` + `ReplanningContext` → new `Population` |
| 2. mobsim | `Mobsim` | `Scenario` → `Trajectories` (arrays); events only for registered handlers |
| 3. scoring | `Scorer` | `Trajectories` (vectorised) or `ExperiencedPlan` → score of the selected plan |
| 4. learning | `Trajectories.travel_time_table` | trajectories → `TravelTimeTable` for the next replanning |
| 5. metrics | `iteration_metrics` | scores and trajectories → `dict` row |

### Data architecture

MobSlim is built around a few immutable values. The stages are functions
from one value to the next. They never share mutable state, so any stage
can be replaced, run in another process, or ported as long as it
reads and writes the same data.

**The data model.** A `Scenario` holds everything that carries over from
one iteration to the next. Everything inside refers to nodes, links and
persons by dense `int` index. External ids only appear in the `IdMap`s.

```text
Scenario                                    frozen
├── network: Network                        struct of arrays
│   ├── node_ids: IdMap                     external id ⇄ node index
│   ├── node_xy           (n_nodes, 2) float
│   ├── link_ids: IdMap                     external id ⇄ link index
│   ├── from_node, to_node  (n_links,) int ─────────► node index
│   ├── length, freespeed, capacity, lanes  (n_links,) float
│   └── link_modes        tuple[frozenset[str]], one per link
│
├── population: Population                  ordered, immutable, stored as tables
│   ├── ids: IdMap                          person id ⇄ person index (= row)
│   ├── plans: PlanStore                    every stored plan as CSR columns,
│   │                                       append-only, compacted when half unused
│   ├── plan_ids  (n_persons, n_slots) int  ─► plans row; -1 = empty slot
│   ├── scores    (n_persons, n_slots) float NaN = unscored
│   ├── selected  (n_persons,) int          selected slot
│   ├── n_plans   (n_persons,) int          plan memory size, ≤ max_plans
│   │   population[i] ─► Person(id, plans, selected, scores)   value view
│   │   population.selected_batch() ─► PlanBatch               vectorised gather
│   └── Plan(elements)                      content only, no score; read from a
│                                           population it is a lazy view of a row
│           ├── Activity(type, node, end_time, duration, x, y, link)
│           │                     node ─────────────► node index
│           │                     link ─────────────► link index (optional; ends at node)
│           └── Leg(mode, route, travel_time)
│               └── Route(links, travel_time, distance)
│                                 links ────────────► link indices
│
└── teleported_modes: {mode: TeleportedModeParams(speed, beeline factor)}
```

**Data produced in each iteration.** These values are derived from the
mobsim's trajectories. All are rebuilt each iteration. Only the scores and
the travel-time table feed into the next one.

```text
 Scenario ──► PlanBatch ──► Mobsim ──► Trajectories ─ ─ ─► events, only for
 (selected    flat arrays,             aligned with the     registered handlers
  plans)      CSR offsets              PlanBatch:           (EventLog, XML, yours)
                                       act_start/end   per activity
                                       leg_departure/arrival/distance per leg
                                       link_enter      per route link
                                       stuck           per person
                                              │
          ┌───────────────────────────────────┼──────────────────────┐
          ▼                                   ▼                      ▼
   Scorer.score_trajectories           TravelTimeTable         ExperiencedPlan
   one float per person                times: (n_links,        objects, built only
          │                            n_bins) float           on request
          │                            bin_size: seconds       (state.experienced)
          │                                   │
          │                                   ▼
          │                             ReplanningContext
          │             network, travel_time,
          │             teleported_modes,
          │             iteration
          │                   │
          │                   ▼
          │             Replanning ──► new plans,
          ▼                            new selection
   apply_scores                        │
   scores[i, selected[i]] =            │
   (1-lr)·old + lr·new                 │
          │                            │
          └──────────────┬─────────────┘
                         ▼
               new Population ──► new Scenario   (next iteration)
                         │
                         ▼
               iteration_metrics ──► dict row
               RunResult.metrics, metrics.csv
```

What crosses a process or language boundary:

| Value | Shape | Sent to |
|---|---|---|
| `Network` | numpy arrays + `IdMap` (pickled without caches) | mobsim, routers, worker processes |
| `Person` / `Plan` | tuples of frozen dataclasses with `int` ids | replanning workers |
| `PlanBatch` | flat arrays with CSR offsets; type/mode codes | mobsim |
| `Trajectories` | float arrays aligned with the `PlanBatch` | scoring, travel times, metrics |
| `TravelTimeTable` | one `(n_links, n_bins)` float array | routers, worker processes |
| events | flat records: `time`, `person` and `int` ids (optional) | handlers, writers |
| scores | one `float` per person index | `apply_scores` |

### Package layout

```
src/mobslim/
├── ids.py            IdMap: external id <-> dense int index
├── timeutils.py      parse_time / format_time (seconds from midnight)
├── network.py        Network (struct of arrays), NetworkBuilder, read_network_xml, grid/linear networks
├── population.py     Activity, Leg, Route, Plan, Person, Population, read_population_xml
├── scenario.py       Scenario = network + population + teleported modes
├── modes.py          Modes: network vs teleported modes, teleport times (scenario.modes)
├── events.py         typed events, EventHandler, EventsManager, EventLog, EventsXMLWriter
├── experience.py     ExperiencedPlan + ExperiencedPlanCollector (what actually happened)
├── planbatch.py      PlanBatch: plans as flat arrays (what a mobsim executes), PlanBatchBuilder
├── planstore.py      PlanStore: a population's plans as append-only columns (take, append, compact)
├── formats/
│   ├── __init__.py   read_population / write_population: format chosen by path
│   ├── parquet.py    Parquet population directory (persons, plans, activities, legs tables)
│   ├── matsim.py     MATSim population_v6 XML writer
│   └── __main__.py   python -m mobslim.formats convert | info
├── trajectories.py   Trajectories (mobsim output, arrays) + TrajectoryCollector
├── config/
│   ├── matsim.py     MATSim config.xml: load, edit, translate to Controller arguments
│   ├── mobslim.py    the mobslim YAML config: load, defaults, precheck, output config, compile
│   ├── schema.py     every mobslim config parameter: type, default (MATSim's), per person or not
│   ├── conditions.py person filters: parse and evaluate conditions on person attributes
│   ├── convert.py    MATSim config.xml -> mobslim config
│   └── __main__.py   python -m mobslim.config check | resolved | reference | from-matsim
├── scoring/
│   ├── __init__.py   Scorer protocol, score_all, apply_scores
│   ├── params.py     ScoringParams, field paths, compiled parameter tables
│   ├── person.py     PersonParams (sets + per-person overrides) and person rules
│   └── charypar_nagel.py  CharyparNagelScorer: one-pass vectorised kernel + per-plan reference
├── mobsim/
│   ├── base.py       Mobsim protocol, SimulationError
│   ├── _common.py    link parameters, route validation, event dispatch shared by both sims
│   ├── qsim.py       QSim: event-driven queue model (flow, storage, spillback, stuck time)
│   ├── agentsim.py   AgentSim: same queue model, agent-driven transitions, polls when blocked
│   ├── _agentsim_transit.py  AgentSim's transit: trips and passengers as agents (tickets)
│   ├── _transit.py   the QSim's transit component
│   ├── _transit_tables.py    transit tables and results shared by the Python sims
│   └── hermes.py     HermesSim: whole-second queue model, Python reference kernel (+ Rust)
├── routing/
│   ├── base.py       Router, TravelTime, RouterFactory protocols
│   ├── dijkstra.py   time-dependent DijkstraRouter (Python reference, route_many)
│   ├── compiled.py   RustRouter (Dijkstra or ALT, parallel route_many), default_router
│   ├── traveltime.py FreeSpeedTravelTime, TravelTimeTable, TravelTimeCollector,
│   │                 travel_time_table
│   └── teleport.py   TeleportedModeParams, teleport()
├── replanning/
│   ├── context.py    ReplanningContext (modes, travel times, routers; picklable)
│   ├── selectors.py  BestScore, Random, ExpBeta, ChangeExpBeta, WorstScore, KeepSelected
│   ├── modules.py    ReRoute, TimeAllocationMutator, ChangeTripMode, route_plans
│   ├── strategy.py   PlanStrategy, StrategyManager, person_rng
│   ├── runners.py    SerialRunner, ProcessPoolRunner (per person)
│   └── vectorised.py VectorRunner: replanning on the population's tables (default)
├── controller.py     Controller, ControllerConfig, RunResult
├── listeners.py      ControllerListener, IterationState, ProgressLogger, MetricsCSVWriter,
│                     EventsXMLListener, EventRecorder
├── metrics.py        iteration_metrics
├── transit/          schedule/vehicle tables, builder, MATSim readers, RAPTOR router, pt walks
└── animate.py        matplotlib animation of events
```

Dependencies point downwards in this order: data model (`network`,
`population`, `scenario`), then `events`, then the stages (`mobsim`,
`routing`, `scoring`, `replanning`), then `controller`. The stages never
import each other's implementations, only the data model and protocols.

### Conventions

* **Immutable data.** `Activity`, `Leg`, `Route`, `Plan` and `Person` are
  frozen dataclasses. Change them with `dataclasses.replace` or the helpers
  (`person.add_plan`, `person.with_selected_score`, `population.replace`,
  `population.with_selected_scores`, ...). No component edits its inputs.
  `Population` arrays are read-only.
* **Plans and scores are separate.** A `Plan` is content only. Scores and
  selection are per-person state: `person.scores[i]` belongs to
  `person.plans[i]`, and a `Population` keeps them as arrays.
* **Integer indices everywhere inside.** Nodes, links and persons are dense
  `int`s. Translate at the edges: `network.node_index("12")`,
  `network.link_ids.id(7)`, `population.ids.id(0)`. Events carry indices.
* **Units.** Seconds from midnight, meters, m/s, and **vehicles per hour per
  link** for capacity (not per lane; the loader normalises `capperiod`).
* **Activity timing.** `end_time` is a clock time; `duration` is a length;
  the last activity has neither.
* **Determinism.** Every random choice uses an explicit `random.Random`.
  Replanning gets `person_rng(seed, iteration, person_index)`, so runs are
  reproducible and parallel runs equal serial runs.
* **Logging, not printing.** Progress goes to the `mobslim` logger.

### API tour

#### Network

```python
from mobslim import NetworkBuilder, grid_network, read_network_xml

net = read_network_xml("scenarios/equil/network.xml")   # MATSim network_v1/v2
grid = grid_network(size=4, spacing=50, freespeed=10, capacity=900)

b = NetworkBuilder()
b.add_node("a", 0, 0); b.add_node("b", 100, 0)
b.add_link("ab", "a", "b", length=100, freespeed=13.9, capacity=1800, lanes=1, modes=["car"])
net = b.build()

net.n_links, net.modes            # 1, frozenset({'car'})
net.length, net.capacity          # numpy arrays indexed by link
net.out_links("car")              # CSR adjacency (offsets, links) for routing
net.nearest_node(x, y)            # spatial lookup
net.to_networkx()                 # MultiDiGraph view for analysis/plotting
```

#### Population and plans

```python
from mobslim import Plan, Population, read_population, read_population_xml

home, work = net.node_index("a"), net.node_index("b")
plan = (Plan.builder()
        .act("home", home, end_time=8 * 3600)    # leave at 08:00
        .leg("car")                              # routed later
        .act("work", work, duration=8 * 3600)    # stay 8 h
        .leg("walk")                             # teleported mode
        .act("home", home)                       # last activity: no end
        .build())
population = Population.from_plans({"alice": plan})
population = read_population_xml("plans.xml", net)    # or from MATSim XML
population = read_population("plans_pq", net)         # or any format, by path (see below)
# an activity may also sit on a link (its to-node is the node); car legs to it
# then end by driving through that link, as in MATSim:
#   .act("work", work, link=net.link_index("ab"), duration=8 * 3600)

population[0]                     # Person(id, plans, selected, scores) view
population.selected_plans()       # list[Plan], by person index
population.selected_scores()      # float array, NaN = unscored
population.scores                 # (n_persons, n_slots) array of all scores
population.selected_batch()       # the selected plans as a PlanBatch, no plan objects
```

Populations are stored as a column table of plans (`population.plans`, a
`PlanStore`); `Plan` values read from a population are lazy views of its
rows. **Files:** `read_population(path, net)` and
`write_population(population, path, net)` choose the format by path:
`*.xml`/`*.xml.gz` is MATSim XML, anything else a Parquet population
directory (`persons`, `plans`, `activities`, `legs` tables with external ids;
`pip install mobslim[parquet]`). Parquet keeps every plan, score, route
and person attribute, and reads several times faster than XML.
`python -m mobslim.formats convert plans.xml.gz plans_pq --network network.xml`
converts between them (`info` summarises a file).

#### Scenario and mobsim

```python
from mobslim import Scenario, QSim, QSimConfig, TeleportedModeParams, route_scenario
from mobslim.events import EventLog, EventsManager

scenario = Scenario(net, population,
                    teleported_modes={"walk": TeleportedModeParams(speed=1.2)})

scenario = route_scenario(scenario)             # route network legs at free speed
log = EventLog()
traj = QSim(QSimConfig(flow_capacity_factor=0.1, storage_capacity_factor=0.1,  # 10% sample
                       stuck_time=10)).run(scenario, EventsManager([log]))  # needs routed plans
log.to_dataframe()                              # events: only built because a handler asked

traj.leg_arrival - traj.leg_departure           # travel time per leg (NaN = stuck)
traj.travel_time_table(scenario.network)        # learned link travel times
traj.experienced_plans()                        # ExperiencedPlan objects, on request
traj.plans                                      # the PlanBatch that was executed
```

The mobsim needs routed plans. The controller routes them before the first
iteration; outside it, use `scenario = route_scenario(scenario)`.

#### Routing and travel times

```python
from mobslim.routing import DijkstraRouter, FreeSpeedTravelTime, TravelTimeTable

router = DijkstraRouter(net, "car", FreeSpeedTravelTime(net))
route = router.route(origin, destination, departure_time=8 * 3600)
route.links, route.travel_time, route.distance  # links: tuple; route.link_array: int32 view
routes = router.route_many(origins, destinations, departure_times)  # shares searches

from functools import partial
from mobslim.routing import RustRouter, default_router   # needs rust/mobslim-router
RustRouter(net, "car", table)                    # same routes, parallel route_many
RustRouter(net, "car", table, algorithm="alt")   # A* with landmarks: same times, faster
Controller(scenario, router_factory=partial(RustRouter, algorithm="alt", threads=4))
# default everywhere: default_router (RustRouter if built, else DijkstraRouter)

table = result.travel_time                       # learned, (n_links, n_bins), 15-min bins
table.link_travel_time(link, 8 * 3600)
```

#### Scoring

```python
from mobslim import ActivityParams, CharyparNagelScorer, ModeParams, ScoringParams

scorer = CharyparNagelScorer(ScoringParams(
    performing=6.0,
    activities={"home": ActivityParams(12 * 3600), "work": ActivityParams(8 * 3600)},
    modes={"car": ModeParams(marginal_utility_of_traveling=-6.0),
           "walk": ModeParams(constant=-1.0)},
))
```

#### Replanning

```python
from mobslim.replanning import (ChangeExpBetaSelector, ChangeTripMode, PlanStrategy,
                                RandomPlanSelector, ReRoute, StrategyManager,
                                TimeAllocationMutator, ProcessPoolRunner)

strategies = StrategyManager(
    [
        (0.7, PlanStrategy(ChangeExpBetaSelector())),                         # choose
        (0.1, PlanStrategy(RandomPlanSelector(), [ReRoute()])),               # new route
        (0.1, PlanStrategy(RandomPlanSelector(), [TimeAllocationMutator(), ReRoute()])),
        (0.1, PlanStrategy(RandomPlanSelector(), [ChangeTripMode(["car", "walk"]), ReRoute()])),
    ],
    max_plans=5,
    disable_innovation_at=40,     # selection only from iteration 40
)
```

Replanning runs on the population's tables by default (`VectorRunner`):
strategies are drawn, plans selected and removed with array operations, and
the new plans of all innovating persons go through each module and the
router as one `PlanBatch`. The built-in selectors (`select_many`) and modules
(`mutate_batch`) have vectorised forms that apply the same rules as their
per-person forms (`select`, `mutate`); a custom strategy, selector or module
without one simply runs per person for the persons that drew it. Random
numbers come from one numpy generator per `(seed, iteration)`, so runs are
deterministic. `SerialRunner` and `ProcessPoolRunner` are the per-person
path (a `random.Random` per person; identical to each other), useful for
custom Python strategies at scale or for comparing.

#### Controller

```python
from mobslim import Controller, ControllerConfig, EventRecorder

recorder = EventRecorder()                       # keeps last iteration's events
controller = Controller(
    scenario,
    scorer=scorer,
    strategies=strategies,
    # runner=ProcessPoolRunner(workers=8),       # per-person replanning in parallel
    listeners=[recorder],
    config=ControllerConfig(iterations=50, seed=1, output_dir="output",
                            write_events_interval=10),
)
result = controller.run()
result.metrics_frame()            # one row per iteration: scores, stuck, per-mode tt/distance
result.scenario.population        # final plans with scores
```

With `output_dir` set: `output/metrics.csv` and
`output/ITERS/it.N/events.xml.gz` (MATSim format).

> **Parallel replanning notes.** `ProcessPoolRunner` uses the `spawn` start
> method by default, which is portable and safe with threads. In a script,
> put the run behind `if __name__ == "__main__":`. Custom strategies, modules
> and router factories must be importable by the workers: define them in a
> module, not in a notebook cell. Alternatively, on Linux, use
> `ProcessPoolRunner(start_method="fork")`.

### Extending MobSlim

Every extension point is a protocol. There is nothing to register and no
base class to inherit (except the optional no-op bases for handlers and
listeners).

| To change… | Implement | Plug in via |
|---|---|---|
| traffic simulation | `Mobsim.run(scenario, events) -> Trajectories` | `Controller(mobsim=...)` |
| routing algorithm | `Router.route(o, d, t) -> Route` + a factory `(network, mode, travel_time) -> Router` | `Controller(router_factory=...)` |
| travel-time model | `TravelTime.link_travel_time(link, t)` and `link_travel_times(t)` | your router factory, or a listener |
| utility function | `Scorer.score(ExperiencedPlan) -> float`, optionally `score_trajectories(Trajectories) -> array` | `Controller(scorer=...)` |
| plan choice | `PlanSelector.select(person, rng) -> index`, optionally `select_many(scores, n_plans, selected, rng) -> slots` | `PlanStrategy(selector, ...)` |
| plan innovation | `PlanInnovationModule.mutate(plan, ctx, rng) -> Plan`, optionally `mutate_batch(batch, persons, ctx, rng) -> PlanBatch` | `PlanStrategy(..., modules=[...])` |
| whole replanning policy | `Strategy.apply(person, ctx, rng) -> Person` + `is_innovative` | `StrategyManager([(w, strategy)])` |
| how replanning is executed | `ReplanningRunner.run(...)`, `close()` | `Controller(runner=...)` |
| analysis / output | `EventHandler` (`handles`, `handle`, `reset`, `finish`) | add in `ControllerListener.on_before_mobsim` |
| loop behaviour | `ControllerListener` hooks | `Controller(listeners=[...])` |

**An event handler, attached through a listener:**

```python
from collections import Counter
from mobslim import ControllerListener
from mobslim.events import EventHandler, LinkEnter

class LinkVolumes(EventHandler):
    handles = (LinkEnter,)
    def reset(self, iteration): self.volumes = Counter()
    def handle(self, event): self.volumes[event.link] += 1

class RecordVolumes(ControllerListener):
    def __init__(self): self.by_iteration = {}
    def on_before_mobsim(self, state):
        self.handler = state.events.add_handler(LinkVolumes())
    def on_after_mobsim(self, state):
        self.by_iteration[state.iteration] = self.handler.volumes
```

**An innovation module** (it must be pure, and defined at module level if
you use `ProcessPoolRunner`):

```python
from dataclasses import replace
from mobslim import Activity

class ShiftDepartures:
    """Move every end_time by a fixed number of seconds."""
    def __init__(self, seconds): self.seconds = seconds
    def mutate(self, plan, ctx, rng):
        return plan.with_elements(
            replace(e, end_time=e.end_time + self.seconds)
            if isinstance(e, Activity) and e.end_time is not None else e
            for e in plan.elements
        )

PlanStrategy(RandomPlanSelector(), [ShiftDepartures(-900), ReRoute()])
```

**A drop-in router** (`routing/compiled.py` is a full example):

```python
class FastRouter:                               # any object with .route(...)
    def __init__(self, network, mode, travel_time):
        offsets, links = network.out_links(mode)
        self._impl = mobslim_rs.Router(offsets, links, network.to_node, travel_time.times)
    def route(self, origin, destination, departure_time):
        links, tt, dist = self._impl.route(origin, destination, departure_time)
        return Route(tuple(links), tt, dist)
    def route_many(self, origins, destinations, departure_times):  # optional
        ...                                     # used by batched rerouting

Controller(scenario, router_factory=FastRouter)
```

**A custom mobsim** executes each person's selected plan and returns
`Trajectories`. A native mobsim fills the arrays directly from a
`PlanBatch` (plain numeric arrays in, plain numeric arrays out). A mobsim
that only emits events can wrap a `TrajectoryCollector`. `AgentSim` is a
built-in alternative: `Controller(scenario, mobsim=AgentSim(AgentSimConfig(retry_interval=1.0)))`
(see DESIGN.md §3.3); it runs transit too. `HermesSim` is the fast one: the same queue model on a
whole-second clock, run by a compiled Rust kernel (`rust/mobslim-hermes`,
built by `uv sync` when cargo is installed, with a Python fallback):
`Controller(scenario, mobsim=HermesSim())`. It runs transit too, with
vehicles on the timetable or in the queue model. The tests in
`tests/test_qsim.py` serve as the behavioural specification for alternative
implementations and run against every registered mobsim, and `tests/test_trajectories.py` checks that the arrays and
the events agree.

### Testing

```bash
uv run pytest                 # unit + integration tests, executes demo notebooks, coverage
uv run pytest tests -q -o addopts=""   # fast: tests only
uvx ruff check src tests benchmarks
```

`tests/test_qsim.py` pins down the traffic dynamics: free-speed times,
fractional flow headways, per-link capacity, storage and spillback, FIFO,
stuck time and gridlock, activity timing, teleportation. `tests/test_controller.py`
checks learning on equil and on a grid, reproducibility, identical
serial/parallel replanning, listener hook order and output files.

### Benchmarks

`benchmarks/` times each stage (loading, initialisation, mobsim, event
handling, scoring, routing, replanning) and a full 10-iteration run on
equil, in about 10 s. Save a baseline before a design change and compare
after:

```bash
uv run python -m benchmarks --save base
uv run python -m benchmarks --compare base     # ▲ slower / ▼ faster, per benchmark
uv run python -m benchmarks --profile mobsim.qsim
```

The report also splits the full run by stage, and checks that results such
as event counts and final scores are unchanged. See
[benchmarks/README.md](benchmarks/README.md) for adding benchmarks and
scenarios.

### Verification against MATSim

`verification/` measures how close mobslim is to Java MATSim 2025.0 on
MATSim's own equil and pt-tutorial scenarios. It works in three layers:

* scoring: the same executed day scored by both scorers, which should agree
  exactly;
* mobsim: each mobsim on MATSim's plans and routes, which should be close;
* the full loop over several seeds, which should fall inside MATSim's seed
  spread.

It needs Java 21+ and Maven:

```bash
uv run python -m verification run      # every case, then verification/reports/latest.md
uv run python -m verification freeze   # refresh tests/fixtures/matsim_reference
```

`tests/test_matsim_reference.py` reruns the first two layers on frozen MATSim
runs, so no Java is needed. See [verification/README.md](verification/README.md)
for usage, and [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) for the known
differences and findings.
