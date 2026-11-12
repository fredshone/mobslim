# Developer guide

How MobSlim is built and how to extend it. For installing and running, see
the [README](../README.md); for the reasoning behind the design, see
[design.md](design.md).

* [Architecture](#architecture)
* [Data architecture](#data-architecture)
* [Package layout](#package-layout)
* [Conventions](#conventions)
* [API tour](#api-tour)
* [Extending MobSlim](#extending-mobslim)
* [Testing](#testing)
* [Documentation](#documentation)
* [Benchmarks](#benchmarks)
* [Verification against MATSim](#verification-against-matsim)

## Architecture

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
| 0. prepare (once) | `route_scenario` | activities → assigned links (`assign_activity_links`); un-routed legs → routed at free speed |
| 1. replanning (it ≥ 1) | `StrategyManager` on a `ReplanningRunner` | `Population` + `ReplanningContext` → new `Population` |
| 2. mobsim | `Mobsim` | `Scenario` → `Trajectories` (arrays); events only for registered handlers |
| 3. scoring | `Scorer` | `Trajectories` (vectorised) or `ExperiencedPlan` → score of the selected plan |
| 4. learning | `Trajectories.travel_time_table` | trajectories → `TravelTimeTable` for the next replanning |
| 5. metrics | `iteration_metrics` | scores and trajectories → `dict` row |

## Data architecture

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
│           │                     link ─────────────► link index (assigned in prepare; ends at node)
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
                                       leg_entered     per leg (links entered)
                                       link_enter      per route link (optional)
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

## Package layout

```
src/mobslim/
├── ids.py            IdMap: external id <-> dense int index
├── timeutils.py      parse_time / format_time (seconds from midnight)
├── network.py        Network (struct of arrays), NetworkBuilder, grid/linear networks
├── population.py     Activity, Leg, Route, TransitPassengerRoute, Plan, Person, Population
├── attributes.py     PersonAttributes: person attributes as columns
├── scenario.py       Scenario = network + population + teleported modes + transit
├── modes.py          Modes: network vs teleported modes, teleport times (scenario.modes)
├── events.py         typed events, EventHandler, EventsManager, EventLog, EventsXMLWriter
├── experience.py     ExperiencedPlan + ExperiencedPlanCollector (what actually happened)
├── planbatch.py      PlanBatch: plans as flat arrays (what a mobsim executes), PlanBatchBuilder
├── planstore.py      PlanStore: a population's plans as append-only columns (take, append, compact)
├── trajectories.py   Trajectories (mobsim output, arrays) + TrajectoryCollector
├── formats/          every file reader and writer
│   ├── __init__.py   read_population / write_population: format chosen by path
│   ├── matsim_network.py     read_network_xml (network_v1/v2)
│   ├── matsim_plans.py       read_population_xml (plans_v4/population_v6), write_population_xml
│   ├── matsim_attributes.py  MATSim typed attributes and objectAttributes files
│   ├── matsim_transit.py     read_transit: transit schedule and vehicles
│   ├── parquet.py    Parquet population directory (persons, plans, activities, legs tables)
│   └── __main__.py   python -m mobslim.formats convert | info
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
│   ├── hermes.py     HermesSim: whole-second queue model, Python reference kernel (+ Rust)
│   ├── psim.py       PSim: whole-second steps on network partitions, Python reference kernel (+ Rust, threaded)
│   ├── partition.py  splitting nodes between threads: Morton (traffic-balanced) or METIS (as matsim-rust)
│   └── dsim.py       DSim: matsim-rust's parallel design, Rust kernel only (rust/mobslim-dsim)
├── routing/
│   ├── base.py       Router, TravelTime, RouterFactory protocols
│   ├── dijkstra.py   time-dependent DijkstraRouter (Python reference, route_many)
│   ├── compiled.py   RustRouter (Dijkstra or ALT, parallel route_many), default_router
│   ├── traveltime.py FreeSpeedTravelTime, TravelTimeTable, TravelTimeCollector,
│   │                 travel_time_table
│   └── teleport.py   TeleportedModeParams, teleport()
├── replanning/
│   ├── prepare.py    route_scenario: route un-routed legs before the first iteration
│   ├── context.py    ReplanningContext (modes, travel times, routers; picklable)
│   ├── selectors.py  BestScore, Random, ExpBeta, ChangeExpBeta, WorstScore, KeepSelected selectors
│   ├── modules.py    Reroute, TimeAllocationMutator, ChangeTripMode, route_plans
│   ├── strategy.py   PlanStrategy, StrategyManager, PersonStrategyManager, person_rng
│   ├── runners.py    SerialRunner, ProcessPoolRunner (per person)
│   └── vectorised.py VectorRunner: replanning on the population's tables (default)
├── transit/          schedule/vehicle tables, TransitBuilder, RAPTOR router, pt walks
├── controller.py     Controller, ControllerParams, RunResult
├── listeners.py      ControllerListener, IterationState, ProgressLogger, MetricsCSVWriter,
│                     EventsXMLListener, EventRecorder
├── metrics.py        iteration_metrics
├── cli.py            the `mobslim` command (run, check, convert, ...)
└── animate.py        matplotlib animation of events
```

Dependencies point downwards in this order: data model (`network`,
`population`, `scenario`), then `events` and `formats`, then the stages (`mobsim`,
`routing`, `scoring`, `replanning`), then `controller`. The stages never
import each other's implementations, only the data model and protocols.

## Conventions

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

## API tour

### Network

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
net.link_attribute("type")        # extra link attribute (MATSim: loaded lazily on first use)
net.to_networkx()                 # MultiDiGraph view for analysis/plotting
```

### Population and plans

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
the `parquet` extra: `uv sync --extra parquet`). Parquet keeps every plan, score, route
and person attribute, and reads several times faster than XML.
`mobslim convert plans.xml.gz plans_pq --network network.xml`
converts between them (`info` summarises a file).

### Scenario and mobsim

```python
from mobslim import Scenario, QSim, QSimParams, TeleportedModeParams, route_scenario
from mobslim.events import EventLog, EventsManager

scenario = Scenario(net, population,
                    teleported_modes={"walk": TeleportedModeParams(speed=1.2)})

scenario = route_scenario(scenario)             # route network legs at free speed
log = EventLog()
traj = QSim(QSimParams(flow_capacity_factor=0.1, storage_capacity_factor=0.1,  # 10% sample
                       stuck_time=10)).run(scenario, EventsManager([log]))  # needs routed plans
log.to_dataframe()                              # events: only built because a handler asked

traj.leg_arrival - traj.leg_departure           # travel time per leg (NaN = stuck)
traj.travel_time_table(scenario.network)        # learned link travel times
traj.experienced_plans()                        # ExperiencedPlan objects, on request
traj.plans                                      # the PlanBatch that was executed
```

The mobsim needs routed plans. The controller routes them before the first
iteration; outside it, use `scenario = route_scenario(scenario)`.

### Routing and travel times

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

### Scoring

```python
from mobslim import ActivityParams, CharyparNagelScorer, ModeParams, ScoringParams

scorer = CharyparNagelScorer(ScoringParams(
    performing=6.0,
    activities={"home": ActivityParams(12 * 3600), "work": ActivityParams(8 * 3600)},
    modes={"car": ModeParams(marginal_utility_of_traveling=-6.0),
           "walk": ModeParams(constant=-1.0)},
))
```

### Replanning

```python
from mobslim.replanning import (ChangeExpBetaSelector, ChangeTripMode, PlanStrategy,
                                RandomSelector, Reroute, StrategyManager,
                                TimeAllocationMutator, ProcessPoolRunner)

strategies = StrategyManager(
    [
        (0.7, PlanStrategy(ChangeExpBetaSelector())),                         # choose
        (0.1, PlanStrategy(RandomSelector(), [Reroute()])),               # new route
        (0.1, PlanStrategy(RandomSelector(), [TimeAllocationMutator(), Reroute()])),
        (0.1, PlanStrategy(RandomSelector(), [ChangeTripMode(["car", "walk"]), Reroute()])),
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

### Controller

```python
from mobslim import Controller, ControllerParams, EventRecorder

recorder = EventRecorder()                       # keeps last iteration's events
controller = Controller(
    scenario,
    scorer=scorer,
    strategies=strategies,
    # runner=ProcessPoolRunner(workers=8),       # per-person replanning in parallel
    listeners=[recorder],
    params=ControllerParams(iterations=50, seed=1, output_dir="output",
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

## Extending MobSlim

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

PlanStrategy(RandomSelector(), [ShiftDepartures(-900), Reroute()])
```

**A drop-in router** (`routing/compiled.py` is a full example):

```python
from mobslim import Route
import my_kernel                                # your own (e.g. compiled) router

class FastRouter:                               # any object with .route(...)
    def __init__(self, network, mode, travel_time):
        offsets, links = network.out_links(mode)
        self._impl = my_kernel.Router(offsets, links, network.to_node, travel_time)
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
built-in alternative: `Controller(scenario, mobsim=AgentSim(AgentSimParams(retry_interval=1.0)))`
(see [design.md](design.md) §3.3); it runs transit too. `HermesSim` is the fast one: the same queue model on a
whole-second clock, run by a compiled Rust kernel (`rust/mobslim-hermes`,
built by `uv sync` when cargo is installed, with a Python fallback):
`Controller(scenario, mobsim=HermesSim())`. It runs transit too, with
vehicles on the timetable or in the queue model. `PSim` runs on
several threads (`rust/mobslim-psim`), with results that do not depend
on their number; it pays off on large networks, and runs transit too:
`Controller(scenario, mobsim=PSim(PSimParams(threads=4)))`.
`DSim` reimplements matsim-rust's parallel design and queue dynamics to
compare against it (`docs/design.md` §3.3). The tests in
`tests/test_qsim.py` serve as the behavioural specification for alternative
implementations and run against every registered mobsim, and `tests/test_trajectories.py` checks that the arrays and
the events agree.

## Testing

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

## Documentation

```bash
uv run --group docs sphinx-build docs docs/_build    # HTML in docs/_build/index.html
```

The API reference (`docs/api.rst`) is generated from the docstrings, which
use reStructuredText roles (``:class:`Population` ``) and may use Google-style
`Args:` sections. A new public module needs a line in `docs/api.rst`.

## Continuous integration

GitHub Actions runs on pushes to `main` and on pull requests:

- `.github/workflows/ci.yml` runs `pytest` twice: once Python-only, where the
  tests that need the compiled kernels skip, and once with `--group rust`, which
  builds the kernels with cargo.
- `.github/workflows/docs.yml` builds the docs with `-W`, so a Sphinx warning
  fails the build. On `main` it publishes them to GitHub Pages. For that, set
  the repository's Pages source to "GitHub Actions".

## Benchmarks

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
[benchmarks/README.md](../benchmarks/README.md) for adding benchmarks and
scenarios.

## Verification against MATSim

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
runs, so no Java is needed. See [verification/README.md](../verification/README.md)
for usage, and [reports/latest.md](../verification/reports/latest.md) for the
latest numbers and the known differences.
