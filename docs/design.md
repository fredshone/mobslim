# MobSlim Design

MobSlim is a Python, MATSim-inspired, agent-based transport simulation. It
aims to *behave like* MATSim (same co-evolutionary loop, same queue-model
traffic dynamics, same plan selection ideas), not to reproduce it exactly.
Its main purpose is to be a **workbench for experiments**:

1. **Swap computationally heavy modules for compiled (Rust) implementations**,
   especially the mobsim, routing and event processing.
2. **Parallelise the optimiser**, starting with replanning.
3. **Try learning-based replanning** (e.g. RL policies) against the same
   simulation and scoring.

Every design decision below is judged against those three goals.

> Developer-facing API docs, a diagram and recipes are in the
> [README](README.md#developer-guide). This document records the design and
> the reasoning behind it.

---

## 1. Design principles

| Principle | Why | Where |
|---|---|---|
| **Small interfaces (`typing.Protocol`) at every stage boundary** | Any stage can be replaced by any object with the same methods, including a PyO3 class | `Mobsim`, `Router`, `TravelTime`, `Scorer`, `PlanSelector`, `PlanInnovationModule`, `Strategy`, `ReplanningRunner`, `EventHandler`, `ControllerListener` |
| **Immutable data** | Plans can be shared between iterations, components and processes without defensive copies or aliasing bugs | `Activity`, `Leg`, `Route`, `Plan`, `Person` are frozen dataclasses; `Population` arrays are read-only; `Network`, `Scenario`, `TravelTimeTable` and events are immutable by convention (events are slotted but not frozen: frozen construction is ~3x slower on the hottest path) |
| **Dense integer ids + arrays** | Hot loops index lists and arrays; a compiled extension or worker process receives plain arrays | `IdMap`; `Network` is struct-of-arrays; routes are packed int32 buffers; events reference persons, links and nodes by index |
| **Pure replanning** | `(person, context, rng) -> person` can run in any order, on any process, with identical results | `StrategyManager.replan`, `person_rng(seed, iteration, person_index)` |
| **Arrays in, arrays out; events on request** | The mobsim runs on encoded plans and returns a trajectory table; everything the loop needs is a numpy reduction over it. Events are an optional view for handlers that want them, so a run with no handlers builds no event objects | `PlanBatch` → `Mobsim` → `Trajectories`; `EventsManager` + handlers (`EventsXMLWriter`, `EventLog`, yours) |
| **Plans and experience are separate** | What the agent *intended* (plan) is never overwritten by what *happened* (experienced plan) | `Plan` vs `ExperiencedPlan` |
| **Explicit, reproducible runs** | Experiments must be comparable | `ControllerParams(seed=...)`, per-person RNGs, deterministic mobsim tie-breaking |

---

## 2. The loop

MobSlim follows MATSim's co-evolutionary algorithm. Agents keep a small
memory of scored plans. Each iteration they execute one, are scored on the
outcome, and then either choose among their plans or invent a new one.

```
            ┌──────────────────────────── Controller ─────────────────────────────┐
            │                                                                      │
 Scenario ─▶│  prepare (route at free speed)                                       │
            │      │                                                               │
            │      ▼                                                               │
            │  ┌─────────────┐  selected   ┌──────────┐  events, if handlers       │
            │  │ Replanning  │── plans ───▶│  Mobsim  │─ ─▶ EventsManager ─▶ output│
            │  │ (it ≥ 1)    │ (PlanBatch) │  (QSim)  │                            │
            │  └──────▲──────┘             └────┬─────┘                            │
            │         │                         │ Trajectories (arrays)            │
            │         │ travel times            ├───────────────────┐              │
            │         │                         ▼                   ▼              │
            │         │                 travel_time_table  score_trajectories      │
            │         │                         │                   │ scores array │
            │         └──── TravelTimeTable ◀───┘                   ▼              │
            │                                               Population.scores      │
            │                                                                      │
            └── listeners: startup · iteration_start · after_replanning · before/after_mobsim
                           · scoring · iteration_end · shutdown
```

Per iteration (`Controller.run`):

1. **Replanning** (from iteration 1). The `ReplanningRunner` applies
   `StrategyManager.replan` to every person, given a read-only
   `ReplanningContext` that holds the scenario's `Modes`, last iteration's
   travel times and the router factory.
2. **Mobsim.** `Mobsim.run(scenario, events)` executes every person's
   *selected* plan (encoded as a `PlanBatch`) and returns `Trajectories`.
   It emits typed events only for handlers registered in `before_mobsim`.
3. **Scoring.** `score_all` scores every person from the trajectories:
   vectorised when the scorer has `score_trajectories`, otherwise
   `Scorer.score` on each `ExperiencedPlan`. Scores are written into the
   population's score array, optionally averaged with `learning_rate`.
4. **Learning and metrics.** `Trajectories.travel_time_table` bins link
   travel times for the next replanning; `iteration_metrics` summarises the
   iteration. Both are numpy reductions over the trajectory arrays.

---

## 3. Components

### 3.1 Data model

| Type | Module | Notes |
|---|---|---|
| `IdMap` | `ids` | External id ↔ dense `int` index |
| `Network` | `network` | Arrays: `from_node`, `to_node`, `length` [m], `freespeed` [m/s], `capacity` [veh/h, per link], `lanes`, `link_modes`. Links are keyed by link id, so **parallel links are supported**. Derived, cached views: per-mode CSR adjacency (`out_links`), free-speed times, spatial index, `to_networkx()` |
| `Activity(type, node, end_time, duration, x, y, link)` | `population` | Free-string types. A location (`x, y`) on a link (its to-node is `node`), as MATSim's activities are. Activities with coordinates but no link are assigned the link with the nearest midpoint before the first iteration (`activity_links`, 3.1a); activities placed at a node without coordinates stay there |
| `Leg(mode, route, travel_time)` | `population` | `route` is a `Route(links, travel_time, distance)` for network modes and `None` for teleported modes. A route stores its links as packed native int32 `bytes` (`data`); `links` decodes a tuple, `link_array` is a zero-copy numpy view |
| `Plan(elements)` | `population` | Validated alternating `Activity, Leg, …, Activity`. Content only: no score. Immutable; a plan read from a population is a lazy view of a `PlanStore` row (elements built on first access), compared by content |
| `Person(id, plans, selected, scores)` | `population` | Plan memory as a value; `scores[i]` belongs to `plans[i]` (`None` = unscored). What strategies and selectors work on |
| `Population` | `population` | Tables, row = *person index*: a plan store `plans` (a `PlanStore`) plus `plan_ids` and `scores` `(n_persons, n_slots)` and `selected`, `n_plans` arrays. `population[i]` builds a `Person`; `selected_batch()` gathers the selected plans as a `PlanBatch` |
| `PlanStore` | `planstore` | Every stored plan as CSR columns (the `PlanBatch` layout plus route times and distances). Immutable, append-only snapshots sharing buffers; `take(rows)` gathers a `PlanBatch`, `compact(rows)` drops unused rows. `Population.replace` appends only new plans and compacts when more than half the rows are unused |
| `PersonAttributes` | `attributes` | Per-person attributes as columns (one read-only numpy array per attribute, row = person index; `NaN`/`None` = missing). `population.attributes`; read from MATSim `<attributes>`, `plans_v4` person fields or an `objectAttributes` file. `to_frame()` gives a pandas `DataFrame`. Kept by every derived population; not part of `Person` |
| `Scenario(network, population, teleported_modes)` | `scenario` | Everything a mobsim needs |
| `Modes(network, teleported)` | `modes` | The one answer to "network or teleported, and how long?"; built once as `scenario.modes` and shared by the mobsim, preparation and `ReplanningContext.modes` |
| `PlanBatch` | `planbatch` | Plans as flat arrays: activities, legs and route links in contiguous slices located by offsets (CSR); activity types and modes as codes; `NaN` for missing times. What a mobsim executes. Route validation is one vectorised pass (`invalid_legs`) |
| `Trajectories` | `trajectories` | The mobsim's output, aligned with the `PlanBatch`: `act_start`/`act_end` per activity, `leg_departure`/`leg_arrival`/`leg_distance` per leg, `link_enter` per route link, `stuck` per person; `NaN` = did not happen |
| `ExperiencedPlan` | `experience` | What actually happened, as objects: activity start/end, leg departure/arrival, distance, links, `stuck`. Built from `Trajectories` on request (and cached) |

**Activity timing** follows MATSim's `tryEndTimeThenDuration`. An activity
with `end_time` ends at that clock time, or immediately if the agent arrives
late. Otherwise it lasts `duration`. The last activity has neither. Unlike the
previous version, a late arrival therefore no longer pushes the rest of the
day back.

**Plans are referenced, scores are arrays.** Scores and the selected plan
change every iteration for every person; plan content changes only for the
few persons who innovate. Keeping the score inside the immutable `Plan`
forced a new `Plan`, `Person` and `Population` for every score update.
Instead a `Population` holds a plan store and per-person arrays, so:

* scoring is one array update (`apply_scores` → `with_selected_scores`);
* score statistics (`iteration_metrics`) are numpy reductions;
* preparation routes each *distinct* plan once (`map_plans`);
* the score matrix is ready for vectorised plan selection, and the arrays
  are what a compiled component or a worker process would want.

The plans themselves are columns too (`PlanStore`): the mobsim's input is
a vectorised gather (`selected_batch`, 14x faster than encoding plan objects
at 200k persons) and a plan costs about 300 B instead of 840 B. Per-person
strategies still see `Person` values whose plans are lazy views, so
`(person, context, rng) -> person` stays pure and picklable; the default
`VectorRunner` (3.6) replans on the tables instead. Population files:
`mobslim.formats` reads and writes MATSim XML and a Parquet population
directory (persons, plans, activities, legs tables) by path.

### 3.1a Activity locations and links (`activity_links`)

Activities are locations, not nodes. Plans usually give coordinates only,
so `route_scenario` (MATSim's `PrepareForSim`, run once by
`Controller.prepare`) assigns every activity that has coordinates but no
link to a link, as MATSim's `XY2Links` does. The activity's node becomes
the link's to-node and its coordinates are kept. Network legs leave from
the end of the link and arrive by driving through it; teleported legs and
pt walks use the coordinates. Routes of legs next to a newly assigned
activity are dropped and routed again.

**Rule.** The link whose *midpoint* (the average of its two nodes; links
are straight lines) is nearest. It is found with a static k-d tree in
`rust/mobslim-router` (`PointIndex`), queried in batches over
`$MOBSLIM_ROUTER_THREADS` threads (about 0.1 µs per activity with 8
threads), or else one query at a time with a `fastquadtree` point quadtree
(about 2 µs). Between equally near links, the k-d tree takes the lowest
link index; the quadtree may take any one of them. The rule is simple and
fast, but approximate next to long links: a point beside the middle of a
long link can be nearer another link's midpoint.

**Preferences.** `ActivityLinkParams(modes, tiers)`: only links allowing
one of `modes` (default car) are eligible. Ordered tiers exclude links by
attribute filters up to a distance; e.g. `ActivityLinkParams.avoiding(
LinkFilter("type", values=("motorway",)), unless_beyond=1000)` keeps
activities off motorways unless no other link is within 1 km. Config: the
`activity_links` section. `ControllerParams(activity_links=None)` turns
assignment off.

**Link attributes.** `Network.link_attribute(name)` returns one column per
attribute (`None`/`NaN` where missing; the built-in `length`, `freespeed`,
`capacity`, `lanes` too). `read_network_xml` does not parse extra
attributes up front: a picklable `MatsimLinkAttributes(path)` source parses
the file on first use (a link's `type`/`origid` and its `<attributes>`).

### 3.2 Trajectories and events (`trajectories`, `events`)

**Trajectories are the primary output.** They hold the same information as
the event stream, at about 8 bytes per link traversal instead of two event
objects (≈150 bytes). A link is left when the next one is entered, or at
arrival for a leg's last link, so leave times are derived, not stored.
Travel times, scores and metrics are numpy reductions over these arrays, and
`ExperiencedPlan` objects are built only if something asks for them
(`IterationState.experienced`, analysis, RL observations).

**Events are an optional view.** Typed events named after MATSim's:
`ActivityStart`/`ActivityEnd`, `PersonDeparture`/`PersonArrival`,
`LinkEnter`/`LinkLeave`, `Teleportation`, `PersonStuck`. The `EventsManager`
dispatches by type to `EventHandler`s that declare `handles = (...)`.
Handlers get `reset(iteration)` before and `finish()` after each mobsim run.
The QSim builds an event only if some handler wants its type, so a run with
no handlers builds none (on grid this is ~0.2 s of a 0.4 s mobsim run).

Built-in handlers: `EventLog` (in memory, with `to_dataframe()`),
`EventsXMLWriter` (MATSim `events_v1` XML, gzip optional, for comparison
with Java tooling), and three collectors for event-only producers:
`TrajectoryCollector` (builds `Trajectories`), `TravelTimeCollector` and
`ExperiencedPlanCollector`. Tests check that the QSim's directly recorded
trajectories, and everything derived from them, agree with these
event-based collectors.

### 3.3 Mobsim (`mobsim`)

```python
class Mobsim(Protocol):
    def run(self, scenario: Scenario, events: EventsManager) -> Trajectories: ...
```

How the mobsims differ from each other and from their MATSim equivalents,
in plain terms and with measured effects, is in [The mobsims compared](mobsims.md).

**Timing parameters, shared by every mobsim** (`mobsim._common.check_timing`):
`continuous_time` (exact times or whole seconds), `junction_delays` (a
vehicle that could leave a link and crosses into a next one does so one
second later, and an activity starts one second after the arrival:
MATSim's and matsim-rust's step order) and `rounding` (`nearest` or `down`,
for road and teleport times in whole seconds; activity ends always round
up). QSim and AgentSim do both clocks; the others whole seconds only, and
reject `continuous_time=True`. In whole seconds QSim and AgentSim follow
HermesSim's rules (rounded road and teleport times, activity ends and forced
moves at the next whole second, exits at the first whole second at or after
the unrounded flow slot), so QSim equals HermesSim move for move, and
AgentSim does until spillback (`test_hermes.py`). Junction delays are an
extra second on the earliest exit of every link but a route's last, plus a
delayed activity start. In DSim they are its tick order instead (nodes
before links, as matsim-rust); without them it runs links before nodes, and
a vehicle reaching the end of a link crosses in the same tick.

Transit vehicles keep the timetable or drive the network per route:
`TransitParams.vehicles` sets the default and `schedule_modes` lists the
route modes that keep the timetable anyway (MATSim's deterministic pt,
`Transit.trip_on_network`). The kernels branch per trip and per route stop,
so the two kinds of line run side by side in one run. Transit takes the
same timing parameters in every mobsim
(`mobsim._transit_tables.transit_times` computes its times for all of them):
in whole seconds the timetable's times round up and walk, door and
per-passenger times round by `rounding`; with junction delays a transit
vehicle serves its stops on reaching the end of a link and crosses into the
next one a second after it could leave (after leaving the stop, if it has
one there), which is MATSim's order, and a passenger's activity starts a
second after the arrival. With whole seconds the QSim equals HermesSim on
transit move for move too, and AgentSim equals the QSim without spillback
(`test_hermes.py`, `test_agentsim_transit.py`).

`QSim.run` encodes the selected plans (`PlanBatch.from_plans`) and calls
`QSim.run_batch(scenario, plans, events)`, which is the array-in, array-out
core. Inside, the batch arrays are converted to Python lists once, because
reading single elements from lists is several times faster than from numpy
arrays in a Python loop.

`QSim` is an **event-driven** implementation of MATSim's queue model:

| Behaviour | MobSlim | MATSim default |
|---|---|---|
| Free-speed time | `length / freespeed` | same |
| Flow capacity | per link, `capacity × flow_capacity_factor`, fractional headways | same (fractional accumulation) |
| Storage capacity | `length × lanes / 7.5 m × storage_capacity_factor`, at least `freespeed_time × flow` and ≥ 1 | same |
| Spillback | yes; full links block upstream | same |
| FIFO | by vehicle identity | same |
| Stuck time | 10 s, then forced onto the next link | same (`stuckTime=10`) |
| Time | continuous (events), no fixed step | 1 s steps |
| Non-network modes | teleported: beeline × factor / speed | same |
| Sim end | `end_time=None` runs until everyone is done; with an end time, unfinished agents get `PersonStuck` | same |

Nothing is polled. Each link keeps exactly one pending "try to leave" wake-up
for its head vehicle, and a link that frees space wakes one waiting upstream
vehicle. Cost scales with the number of vehicle moves (≈1.2 scheduler
operations per move in stress tests), not with agents × seconds.

The QSim does not mutate the scenario, is deterministic (ties are broken by
insertion order), and validates routes before running.

#### AgentSim: the agent-driven alternative

`AgentSim` (`mobsim/agentsim.py`) runs the **same queue model**. Both sims
build their links with `mobsim._common.link_parameters`, so free-speed time,
per-link fractional flow headways and storage are identical by construction,
and FIFO is strict. They differ in *what is scheduled*:

| | `QSim` | `AgentSim` |
|---|---|---|
| Heap entries | links (try-leave for the head vehicle) and agents | agents only (persons and transit trips): each has at most one live pending *transition* (end activity and depart, leave link, end teleport, walk to a stop, a trip's next stop or door step). Waiting and riding passengers have none: the trip moves them |
| Waits with a known end (free speed, flow headway, activities, teleports) | scheduled exactly | scheduled exactly; when a head vehicle leaves, the next vehicle in that queue is scheduled at its earliest exit |
| Waiting for storage (spillback, departing onto a full link) | the blocked vehicle registers as a waiter; the link wakes it the moment a slot frees | the blocked vehicle **polls** every `retry_interval` (default 1 s), capped at `blocked_since + stuck_time` |
| Who gets a freed slot | first waiter in line | the first vehicle to retry |
| Gridlock with `stuck_time=inf` | heap drains | detected: all pending transitions are retries and nothing has moved for one retry interval |

Consequences:

* Without spillback the two sims give **identical trajectories**
  (`test_agentsim.py::test_matches_qsim_without_spillback`).
* Under spillback, blocked vehicles react up to `retry_interval` late,
  like MATSim's 1 s time step. `retry_interval` is the knob between the two.
* Flow capacity never depends on `retry_interval`. FIFO successors are
  scheduled, not polled, so a 0.5 s headway holds with 1 s retries.
* Each transition is the agent's own decision point, which is where a
  within-day (en-route) choice model would hook in.
* AgentSim is simpler to port (one agent-indexed heap, no waiter tokens or
  stale wake-ups). In Python it is about 10% faster on grid (≈0.26 s
  against ≈0.29 s with no event handlers).
* **Transit** is native (`mobsim/_agentsim_transit.py`): trips are agents whose heap entries go
  through a ticket, so a trip's next step can move earlier (a passenger
  reopening the doors) without a check every car pays for. Storage counts
  PCE. The rules are the QSim's (`test_transit.py` runs against both), and
  without spillback the trajectories are identical
  (`test_agentsim_transit.py`). Under spillback, buses poll like cars.

#### HermesSim: whole seconds and a compiled kernel

`HermesSim` (`mobsim/hermes.py`) is modelled on MATSim's Hermes. It keeps
the QSim's queue model and its wake-on-space handling of blocked vehicles
(no polling), and changes how the work is executed:

* **Whole-second clock and a bucket agenda.** Entries go into per-second
  buckets that are drained in push order, so with whole-second parameters
  HermesSim makes the same moves in the same order as the QSim
  (`test_hermes.py::test_matches_qsim_with_whole_second_parameters`).
  Free-speed and teleport times are rounded half up, and activity ends round
  up. The next flow slot is kept unrounded, so fractional headways don't
  drift (a 1.8 s headway averages 1.8 s).
* **A kernel over flat arrays.** `run_kernel` takes the `PlanBatch` arrays
  plus per-link parameters and returns the trajectory arrays. It exists
  twice: in Python (the readable reference and fallback) and in Rust
  (`rust/mobslim-hermes`, built by `uv sync` as the `mobslim_hermes`
  extension). The two must agree exactly, on trajectories and events
  (`test_hermes.py::test_rust_matches_python*`). `backend="auto"` uses Rust
  when it is installed.
* **Events are recorded compactly and built afterwards.** The kernel
  appends `(time, kind, person, ref, ref)` for each event type some handler
  wants, and the event objects are built after the run, in that order.

On the 40×40 grid sweep, the Rust kernel takes about 0.2 s for 1.4 M link
traversals (about 140–160 ns each, against 6.5 µs for the QSim), whatever
the congestion; a whole `HermesSim.run` takes about 0.26 s, 80% of it in
the kernel (gathering the plans with `selected_batch` takes 3 ms).

The kernel is bound by memory latency, not compute: the cost per traversal
grows with the scenario (about 100 ns on a 20×20 grid, 200 ns on 80×80) as
its state outgrows the caches. Its layout keeps a vehicle move to a few
cache lines: each link's hot state is one 64-byte `LinkHot`, link queues
are lists threaded through the vehicles' `VehHot` records, and a queue
entry carries the vehicle's next link, read when it entered the link,
while that part of its route was in cache.

Compared with Hermes itself, HermesSim does not poll blocked links every
second, keeps stuck time per vehicle (Hermes's is per link), and does not
copy Hermes's flow budget, which earns no credit on idle seconds.

This replaces the original instruction-stream `Sim`. That sim had the same
agent-driven loop, but whole-second rounding, per-lane flow, storage in metres, no
stuck time, and an exit that popped the queue head whoever was leaving, which
broke FIFO.

#### PSim: whole seconds on several threads

`PSim` (`mobsim/psim.py`, Rust kernel `rust/mobslim-psim`)
runs HermesSim's queue model with the network split across threads. HermesSim
cannot be split this way and keep its results: within one second its moves
chain through any number of links and nodes in one global order (that order
is what makes it equal the QSim), so threads would have to take turns at
every partition boundary. PSim changes the rules instead, so that
**results are identical for any number of threads and any partition**, and
between its Python reference kernel and the Rust kernel:

* **Each second is one synchronous step.** A node decides every move through
  it: vehicles leaving its incoming links, and persons departing onto its
  outgoing links, best first, while the target link has storage. A refused
  head stops its source for the step (FIFO).
* **State has one owner.** A link's queue and flow slot belong to its
  to-node, its storage and departure line to its from-node; a person belongs
  to whichever queue, line or agenda it is in.
* **Effects elsewhere are messages**, applied at the end of the step, even
  inside one partition: a vehicle entering a link, space freed on a link, an
  activity or teleport ending later. A node step reads only state its node
  owns.
* **Who goes first** at a node: the vehicle that has waited longest (since it
  was first refused, else since it could first leave), ties broken by a hash
  of the second and the source.

What that costs, against HermesSim: each link takes at least 1 s, freed
space is reused one second later, teleports take at least 1 s, and merging
traffic is ordered differently. A late arrival still departs at once when its
next leg starts at the node it arrived at (the usual case).

Measured on the mobsim sweep's grids (4-core laptop, Rust kernels):

| | HermesSim | PSim 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| grid 40, 1.4 M link traversals, free flow | 0.18 s | 0.36 s | 0.13 s | 0.15 s |
| grid 80, 10.9 M link traversals, free flow | 4.7 s | 6.6 s | **2.0 s** | 1.8 s |
| grid 40, capacity ×0.05, no stuck time (travel times ×26) | 0.23 s | 0.41 s | 0.29 s | 0.48 s |

* It pays off on large scenarios: 3.4× from 1 to 4 threads on grid 80, 2.4×
  faster than HermesSim. On one thread it is 1.4–2× slower than HermesSim
  (a priority queue per node step and messages instead of direct writes).
* Under heavy congestion each second has little work, and the per-second
  barrier dominates: threads barely help.
* Free-flow trajectories equal HermesSim's (tested). On MATSim's equil
  references (`tests/fixtures/matsim_reference`) it is as close to MATSim
  as HermesSim. With the default stuck time it also agrees under heavy
  congestion (within 0.2% of the QSim at capacity ×0.05 on grids 20 and
  40). **With the stuck time switched off it differs**: total travel time
  is 15% (grid 20) to 33% (grid 40) above the QSim's and HermesSim's, which
  agree within 1%. The cause is the merge
  order: when a full link frees a slot, HermesSim gives it to a vehicle that
  becomes ready in that same second before waking the longest waiter, which
  keeps streams moving; PSim gives it to the longest waiter. Neither
  is MATSim's rule (a random choice weighted by capacity), and in gridlock
  regimes small changes to it move results this much.

**Transit** keeps those rules. A route stop (its waiting passengers and the
trips there) belongs to the to-node of its link, which runs the trips' stop
tasks (doors, boarding, alighting, departures) at the start of its node step
and again whenever a step creates tasks due now, so a bus that leaves a stop
crosses the node in the same step. A trip in network mode appears at the end
of its first link at its departure time and takes its storage from the next
second (it is never refused there, as in MATSim, where departing vehicles
enter the buffer without storage). Passengers and trips that move to
another node or a later second are messages; a walk of 0 s on the spot
happens at once. Trajectories and events with transit do not depend on the
thread count or the partition either (`test_psim.py`, on a grid of bus
lines crossing the partitions), and on MATSim's pt-tutorial PSim
gives HermesSim's results.

#### DSim: matsim-rust's design, for comparison

`DSim` (`mobsim/dsim.py`, Rust kernel `rust/mobslim-dsim`, no Python
kernel) reimplements the parallel design of
[matsim-rust](https://github.com/matsim-vsp/matsim-rust) from its
documentation (no code taken: it is GPL-3.0), so the two designs can be
compared on the same plans:

* **Workers own their state.** Each partition's worker owns its nodes, the
  exit side of the links into them (queue, buffer, flow capacity), the entry
  side of the links out of them (storage, departures) and the persons there.
  Moves inside a partition take effect at once. A link between partitions is
  *split*: entering vehicles go downstream in a message, freed space comes
  back upstream.
* **Neighbour synchronisation.** Every second, a worker exchanges one message
  with each partition it shares a link with and waits for those only. Persons
  that change worker go in messages at least one second ahead; for workers
  that are not neighbours, a lookahead gate keeps the receiver from running
  ahead of them.
* **Every second is simulated** from the first activity end; idle links sleep
  until their first vehicle is due.
* **matsim-rust's queue dynamics**, which are closer to MATSim's QSim than
  HermesSim's: queue and buffer per link, flow capacity accumulated per
  second and consumed on entering the buffer, free-speed times floored to whole seconds, nodes serving
  in-links at random weighted by capacity (one seeded stream per node),
  arrivals using no flow capacity, stuck after waiting *more* than
  `stuck_time`, and activities starting one second after arrival.
* **Its default timing is matsim-rust's:** `junction_delays=True`,
  `rounding="down"`; both can be changed like any mobsim's.
* **One deviation from matsim-rust, by default:** a buffered vehicle keeps
  its storage until it crosses the node, as in the other mobsims, so a link
  never holds more than its storage. matsim-rust frees it on entering the
  buffer (`buffer_in_storage=False`), so a link holds about one second's
  flow more.
* **Partitioning by METIS**, as matsim-rust (unit weights, 3% imbalance,
  contiguous, few neighbours; `pymetis`, the `metis` extra), or Morton
  (`partitioner=`, also available to PSim). See
  `mobsim/partition.py`.

Trajectories (and the canonical event order) do not depend on the worker
count or the partition (`tests/test_dsim.py`, including random partitions);
they depend on `seed`. The shared mobsim spec runs it with seven documented
deviations (strict xfails in `tests/test_qsim.py`), all from the rules above.

Kernel times on the same laptop and day (PSim / DSim, Morton
partitions, no stuck time, DSim with matsim-rust's buffer storage;
best of 2–3):

| | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|
| grid 40, free flow | 0.72 / 1.28 s | 0.40 / 0.63 s | 0.26 / 0.36 s | 0.96 / 1.43 s |
| grid 80, free flow | 9.2 / 15.6 s | 5.3 / 8.1 s | 3.0 / 4.7 s | 8.4 / 8.8 s |
| grid 40, capacity ×0.05 | 0.83 / 7.65 s | 0.47 / 3.87 s | 0.47 / 2.14 s | 1.24 / 4.23 s |

* Both designs scale about equally (≈3× from 1 to 4 threads on grid 80);
  DSim's kernel is 1.5–1.8× slower under free flow and 4–9× under heavy
  congestion, where it steps every second and serves every full buffer.
* METIS cuts fewer links than Morton but, with unit node weights, balances
  nodes rather than traffic, and was slower on 4 threads on grid 80
  (DSim 6.9 s against 4.7 s).
* Results: free flow is about 5% slower than HermesSim (one second per
  link from buffer over the node). At capacity ×0.05 with the default stuck
  time, total travel time is 3–7% below the QSim's (grids 40 and 20); with
  the stuck time switched off, 25–40% below. matsim-rust's buffer storage
  changes either by at most 2 points. On MATSim's equil references it is
  the closest to MATSim's link times (mean absolute error 0.21 s against
  HermesSim's 0.47 s on equil-100; 1.03 s, or 0.48 s with matsim-rust's
  buffer storage, against 1.18 s on equil-2000-cap0.5); leg times are within
  0.05% for all. See [The mobsims compared](mobsims.md).

**Transit**, as MATSim does it: a transit vehicle serves a stop when it
reaches the front of its link's queue, before it moves to the buffer (at a
blocking stop it holds the queue; at a non-blocking one it leaves the queue
and rejoins at its front when it departs), so with junction delays it
crosses the node one tick after leaving the stop. A link runs its stops'
tasks in the links step, so everything a stop does stays on its link or goes
into a message for a later tick, and a trip's state travels with its
vehicle. Trips appear at the end of their first link as in PSim.
Passengers can walk to any stop, so with transit workers that are not
neighbours stay within one tick of each other (`gate` = 1). Results do not
depend on the worker count or partition (`test_dsim.py`). On MATSim's
pt-tutorial, with its default timing, it is the closest to MATSim of all
the mobsims (see [The mobsims compared](mobsims.md)).

### 3.4 Routing (`routing`)

```python
class TravelTime(Protocol):
    def link_travel_time(self, link: int, time: float) -> float: ...
    def link_travel_times(self, time: float) -> Sequence[float]: ...  # all links

class Router(Protocol):
    def route(self, origin: int, destination: int, departure_time: float) -> Route: ...
    # optional: def route_many(origins, destinations, departure_times) -> list[Route]

RouterFactory = Callable[[Network, str, TravelTime], Router]
```

* `TravelTimeTable` is an immutable `(n_links, n_bins)` array (15-min bins by
  default) with `free_speed(...)` and `blend(...)`. It is picklable and
  array-shaped, so it can go straight to worker processes or a compiled router.
* `travel_time_table(network, links, enter, leave)` learns a table from
  link traversals with `np.bincount`, binned by link **entry** time and never
  below free speed. `Trajectories.travel_time_table` feeds it the trajectory
  arrays; `TravelTimeCollector` feeds it from events, appending traversals to
  compact arrays rather than holding an `n_links × n_bins` table.
* Routers read the table one time bin at a time as a tuple of Python floats.
  These tuples are built lazily per bin, and identical bins (e.g. free-flow
  nights) share one tuple, so memory grows with the number of distinct bins
  used, not with `n_links × n_bins` per process.
* `DijkstraRouter` is a time-dependent Dijkstra over the per-mode CSR
  adjacency. It fetches the whole link vector once per settled node
  (`link_travel_times`), not once per link. It is the reference
  implementation.
* **Batches.** `route_many(origins, destinations, departure_times)` answers
  queries that share an origin and departure time with one search that runs
  until every destination is settled. Settled labels are final and the heap
  order (by `(time, node)`) is the same, so the routes are exactly those of
  one search per query. On a **static** model (`travel_time.is_static`: every
  bin the same, e.g. the free-speed table used by `prepare`) the departure
  time cannot matter, so searches start at 0 and all queries from one origin
  share a search.
* **`RustRouter`** (`routing/compiled.py`, the optional `rust/mobslim-router`
  extension) has two algorithms. `"dijkstra"` (default) is a port of
  `DijkstraRouter` with the same heap order, tie-breaking and arithmetic, so
  routes are identical (`tests/test_batch_routing.py` checks this, including
  on tie-heavy regular grids). Its `route_many` releases the GIL and runs
  over `threads` threads (default: `$MOBSLIM_ROUTER_THREADS` or the CPU
  count). `"alt"` is A* with 16 landmarks. Landmark lower bounds are computed
  once per network and mode on free-speed times and reused for every later
  table. That is valid because `travel_time_table` never puts a link below
  free speed (`RustRouter` checks this). ALT finds the same least times; where
  routes tie it may pick another. It is about 4–5× faster per query than
  Dijkstra on irregular networks, but gains little on regular grids, where
  every staircase path ties and A* must still search the whole o–d rectangle.
  It is opt-in: `partial(RustRouter, algorithm="alt")`.
* Travel times cross into Rust as the table's *distinct* columns plus a
  bin → column map, so free-flow bins are stored once.
* `default_router` is the default factory everywhere: `RustRouter` when it is
  installed and the travel times are a `TravelTimeTable` or
  `FreeSpeedTravelTime`, else `DijkstraRouter`. Both give the same routes, so
  results do not depend on whether the extension is built.
* Replanning holds a *router factory*, not a router. Routers are built lazily
  inside each process and cached in the `ReplanningContext`, but never pickled.
* **Batched rerouting.** The replanning runners replan with `ctx.batched()`.
  In that mode `Reroute` only clears routes. After everyone has been replanned,
  the runner routes all new plans together (`route_plans`, in rounds: every
  plan's first unrouted leg, then the next, with one `route_many` per mode
  per round, since a leg's departure depends on the legs before it). The
  routes equal routing during replanning (`SerialRunner(batch_routing=False)`),
  so `Reroute` should be the last module of a strategy. `route_scenario`
  (`prepare`) uses the same `route_plans`. `route_batch` is the same on a
  `PlanBatch` (the `VectorRunner`'s path): it walks every plan's clock leg
  position by leg position with the same arithmetic, so its routes equal
  `route_plans`'.

On the grid benchmark (441 nodes, 2,000 commuters), routing 4,000 congested
trips takes 1.7 s in Python, 94 ms with `RustRouter` on one thread and 30 ms
on 8 threads. `prepare` went from 1.6 s to 70 ms, and one replanning step from
about 400 ms to 60 ms. Most of what remains there is not routing.

### 3.5 Scoring (`scoring`)

```python
class Scorer(Protocol):
    def score(self, plan: ExperiencedPlan) -> float: ...
```

`CharyparNagelScorer` follows MATSim's default scoring (verified exact
against MATSim 2025.0, see `verification/`):

* activities: MATSim's `calcActScore`, i.e. logarithmic utility
  `β_perf · t_typ · ln(dur / t_0)` continued linearly below `t_0`, with
  `t_0` "uniform" (`t_typ · exp(−10 h / (t_typ · priority))`) or "relative"
  (`t_typ · exp(−1 / priority)`, MATSim's default); performed only while
  open (opening/closing times, the rest is waiting), late arrival after
  `latest_start_time`, early departure before `earliest_end_time` or below
  `minimal_duration`;
* overnight wrap of first and last activity when their types match;
* per-mode `constant + β_travel · hours + (β_dist + money rate · β_money) · meters`,
  plus daily constants per used mode;
* `stuck_penalty` for agents that did not finish.

Person attributes reach scorers through `score_all`: a scorer whose
`score_trajectories` (or `score`) takes an `attributes` keyword gets the
population's `PersonAttributes` (or, for `score`, the person's row as a dict).
`CharyparNagelScorer(params, subpopulations={...})` uses the `subpopulation`
attribute to score each subpopulation with its own `ScoringParams` (MATSim's
`scoringParameters` sets). Mobsims read attributes from
`scenario.population.attributes`.

Person-specific parameters (`scoring.person`): the scorer resolves the
attributes once per run into `PersonParams`, a parameter set per person plus
per-person *overrides* of any numeric field (`"performing"`,
`"modes.walk.marginal_utility_of_traveling"`,
`"activities.work.typical_duration"`, ...; NaN = the set's value). Overrides
come from `person_rules=[...]` applied in order (`Subpopulation`,
`IncomeDependentMoney`, `ModeConstants`, `FromAttributes`, or any callable on
the builder). `score_trajectories` then scores everyone in **one** vectorised
pass: parameter sets are compiled into tables indexed by set
(`scoring.params`), gathered per leg and per activity interval, and
overrides are applied by mode or activity name. The cost does not depend on
the number of sets or rules. The per-plan `score(plan, params=...)` is the
plain-Python reference the tests compare against, and `transit_weights`
derives per-person router weights from the same `PersonParams`.

Not modelled: money events (tolls, fares). They belong in new `Scorer`
implementations or new parameters.

### 3.6 Replanning (`replanning`)

```python
class PlanSelector(Protocol):
    def select(self, person: Person, rng: Random) -> int: ...

class PlanInnovationModule(Protocol):
    def mutate(self, plan: Plan, ctx: ReplanningContext, rng: Random) -> Plan: ...

class Strategy(Protocol):
    is_innovative: bool
    def apply(self, person: Person, ctx: ReplanningContext, rng: Random) -> Person: ...

class ReplanningRunner(Protocol):
    def run(self, population, manager, ctx, seed) -> Population: ...
    def close(self) -> None: ...
```

Person attributes reach strategies and modules through the context: the
runners pass each person `ctx.for_person(i)` (a shallow copy sharing the
routers), so `ctx.attribute("carAvail")` / `ctx.person_attributes()` read the
person being replanned, and `ctx.attributes` is the whole table. Attributes
travel to worker processes with the context, pickled once per call.
`ChangeTripMode(..., ignore_car_availability=False)` uses them as MATSim does
(no car for `carAvail == "never"` or `hasLicense == "no"`).

* **Selectors**: `KeepSelectedSelector`, `BestScoreSelector`, `RandomSelector`,
  `ExpBetaSelector` (logit), `ChangeExpBetaSelector` (MATSim's default
  switching rule), `WorstScoreSelector` (for removal). All except
  `KeepSelectedSelector` and `WorstScoreSelector` try unscored plans first.
* **Modules**: `Reroute` (time-aware, uses learned travel times),
  `TimeAllocationMutator`, and `ChangeTripMode` (combine it with `Reroute`).
* **`PlanStrategy(selector, modules)`**: selection-only without modules;
  with modules it copies the selected plan through them and adds the result as
  a new, unscored, selected plan.
* **`StrategyManager`** (immutable; choice tables built once): weighted strategy choice, plan memory limit
  (`max_plans=5`, removing the worst plan), and `disable_innovation_at`.
  `StrategyManager.default()` gives 80% ChangeExpBeta, 10% Reroute and 10%
  TimeAllocationMutator+Reroute.
* **Runners**: `SerialRunner`; `ProcessPoolRunner` splits the population into
  contiguous chunks and sends `(manager, context, chunk)` to a persistent
  process pool. The manager and context are written once per iteration to a
  temporary file that each worker loads once; chunks carry only persons. Each person's RNG is `person_rng(seed, iteration, index)`, so
  **serial and parallel runs are identical**, and tests check this.
* **`VectorRunner`** (the default): the same rules on the population's
  tables. One uniform per person picks strategies; selectors' `select_many`
  and the removal selector work on the `(n, n_slots)` score table; the
  selected plans of each innovative strategy are gathered as one `PlanBatch`,
  sent through the modules' `mutate_batch`, routed together (`route_batch`)
  and appended. Random numbers come from one Philox generator per
  `(seed, iteration)` (`replanning_rng`), so results are deterministic but
  differ from the per-person runners' in the draws. A strategy whose selector
  or module has no vectorised form runs per person (as `SerialRunner`) for
  the persons that drew it. About 12x faster than `SerialRunner` at 100k
  persons.

### 3.7 Controller (`controller`, `listeners`, `metrics`)

```python
Controller(scenario, *, mobsim=QSim(), scorer=CharyparNagelScorer(),
           strategies=StrategyManager.default(), runner=VectorRunner(),
           router_factory=DijkstraRouter, listeners=(), params=ControllerParams())
```

* `ControllerParams`: `iterations`, `seed`, `learning_rate`,
  `travel_time_bin_size`, `travel_time_horizon`, `output_dir`,
  `write_events_interval`.
* `Controller.from_matsim_config(path, strict=False, **overrides)`: a run from
  a MATSim `config.xml`. `mobslim.config.matsim` translates each module into
  the constructor arguments above (`controller_kwargs`) and lists every
  param mobslim cannot honour (logged as a warning, or
  `UnsupportedConfigError` if `strict`). The Python API is the one target:
  any other config front-end (`mobslim.config.<format>`) should also produce
  `Controller` keyword arguments.
* `ControllerListener` hooks: `on_startup`, `on_iteration_start`,
  `on_after_replanning`, `on_before_mobsim` (add event handlers here),
  `on_after_mobsim`, `on_scoring`, `on_iteration_end`, `on_shutdown`. Hooks
  receive an `IterationState` (`iteration`, `is_last`, `scenario`, `events`,
  `trajectories`, `travel_time`, `metrics`; `experienced` builds experienced
  plans from the trajectories on first access).
* Built-in listeners: `ProgressLogger` (via `logging`, never `print`),
  `MetricsCSVWriter` and `EventsXMLListener` (added automatically when
  `output_dir` is set), and `EventRecorder` (keeps events in memory, e.g. for
  animation).
* `run()` returns `RunResult(scenario, travel_time, metrics)`, with
  `metrics_frame()` as a DataFrame.

Output layout (with `output_dir`): `metrics.csv` and
`ITERS/it.N/events.xml.gz`.

---

## 4. How the design serves the three goals

### 4.1 Rust swaps

The seams are narrow and data-oriented:

| Module | Interface | Data across the boundary |
|---|---|---|
| Mobsim | `run_batch(scenario, plans, events) -> Trajectories` | In: network arrays and a `PlanBatch` (flat numeric arrays). Out: `Trajectories` (flat float arrays). No Python objects either way; events only if handlers are registered |
| Router | `route(o, d, t) -> Route`, `route_many(...)`, via a factory | CSR adjacency (`out_links`), distinct `TravelTimeTable` columns. Done: `RustRouter` |
| Transit router | `route_many_xy(o_xy, d_xy, t) -> [TransitPassengerRoute]`, via a factory | per-route stop-time matrices, stop to route-stop CSR, footpaths. Done: `RustRaptorRouter` |
| Travel-time learning | `travel_time_table(...)` | traversal arrays in, `(n_links, n_bins)` array out |
| Scoring | `score_trajectories(Trajectories) -> array`, or `score(ExperiencedPlan) -> float` | Arrays, or plain records |

The routers are done (`RustRouter`, identical to `DijkstraRouter`;
`RustRaptorRouter`, identical to `RaptorRouter`), and so is `HermesSim`'s
kernel, transit included. Next, `QSim` (the existing tests define its
behaviour). A Rust QSim implements `run_batch`: a `PlanBatch` in,
`Trajectories` out, both plain arrays, so nothing crosses the boundary as
Python objects. Optional events can be produced from the trajectories, in
Python, only when handlers are registered.

Routes cross in both directions without becoming Python ints. `route_many`
returns one int32 link array, each `Route` keeps its slice of it as packed
bytes, and `PlanBatch.from_plans` builds the mobsim's `route_links` with one
`b"".join` of those slices. On a 40k-link grid with 50k commuters (6.7M route
links) that cut route memory from 284 MB to 41 MB, converting 100k routes
from 0.9 s to 0.22 s, and encoding the `PlanBatch` from 0.62 s to 0.19 s.

### 4.2 Parallel optimisation

* Per-person replanning is pure and seeded per person, so `ProcessPoolRunner`
  matches `SerialRunner` bit for bit. The default `VectorRunner` needs no
  processes: it is array code plus the (threaded) batch router.
* `ReplanningContext` is picklable, and routers rebuild per process from the
  factory.
* Natural next steps behind the same interfaces: shared-memory travel-time
  tables, Ray or Dask runners, and running several mobsims concurrently, e.g.
  for different seeds or scenario variants, via the stateless `QSim`.

### 4.3 Learning-based replanning (planned, not yet implemented)

The pieces an RL setup needs are in place:

* **Action**: a `Strategy` (or `PlanInnovationModule`) that maps
  `(person, context)` to a new plan: route, departure time or mode.
* **Observation**: `Trajectories` (arrays, ready for a policy network) or
  `ExperiencedPlan`s, scores, and the `TravelTimeTable` in
  the `ReplanningContext`.
* **Reward**: `Scorer.score`, or the change in score between iterations.
* **Loop control**: `ControllerListener` hooks, or driving the stages
  directly.

A gym-style day-to-day environment (one `step` = one iteration) is planned
as a thin adapter over these pieces. Within-day decisions would need a mobsim
hook at departure or link-exit points. The QSim already pulls each agent's
next step lazily (`_depart`, `_try_leave`), which is where such a hook would go.

---

## 5. Deliberate differences from MATSim

* Activities are located at **nodes**; MATSim puts them on links. Routes
  run from the origin node to the destination node and include every
  traversed link. An activity read from a plans file keeps its link
  (`Activity.link`, whose to-node is the activity's node). A car leg to
  it is then routed to the link's from-node and through the link, so it
  ends on the activity's link as MATSim's routes do (the departure link is
  not traversed in either). Without the link, a leg reaches the node by the
  quickest link in, which on pt-tutorial skipped two thirds of home links.
* **Continuous-time** event-driven simulation rather than 1 s time steps.
* Routes in input plan files are **ignored** by default; legs are routed
  before the first iteration. `read_population_xml(keep_routes=True)` keeps
  car routes (MATSim's `type="links"` and `plans_v4` node routes).
* Activities keep the **coordinates** of the plans file (`Activity.x, y`);
  teleported legs and pt walks run between them, so beeline distances
  match MATSim's. `route_scenario` fills in node coordinates where there are
  none.
* **Transit** (`mobslim.transit`): one `pt` leg per trip
  carrying a `TransitPassengerRoute` (rides between route stops; walks implied), not
  stage activities. Trips, not vehicles, are simulated (no `Umlauf`).
  Vehicles run on the timetable (`vehicles="schedule"`, default) or in the
  queue network with exact dwell times (`"network"`). Routing is least-cost
  RAPTOR as in SwissRailRaptor, on the schedule (`RaptorRouter` in Python,
  `RustRaptorRouter` compiled, identical routes). Its utilities come from
  the scorer's default `ScoringParams` as SwissRailRaptor derives them
  (each time utility minus `performing`, `-utility_of_line_switch` per
  transfer; `TransitRouterFactory.from_scoring`), unless the controller is
  given a `transit_router_factory`. Persons can have their own weights
  (`RaptorWeights`, passed per query; the router's tables are shared):
  `Controller(transit_weights=fn)` maps the person attributes to them, and
  by default `CharyparNagelScorer` gives each subpopulation its scoring
  params' weights (MATSim's `DefaultRaptorParametersForPerson`). Reroute,
  the runners' batch routing and `route_scenario` honour them; a stored
  plan shared by persons with different weights is split. Every mobsim
  runs transit, with every timing setting.
* Car-type vehicles are 1 PCU; transit vehicles use their type's PCE.
* Scoring omits money events (see 3.5).
* Blocked departures onto a full link wait in a FIFO line per link, and only
  its front counts stuck time (MATSim's departure buffer). The departure
  link itself is not traversed, so its flow capacity is not used.

## 6. Roadmap

1. RL environment adapter (4.3).
2. A Rust `QSim` behind the existing protocols (the router is done).
   `PSim` runs a variant of HermesSim's queue model on several threads;
   open: single-thread speed, and its merge order under heavy congestion (3.3).
3. Vectorised selection for selection-only strategies (most replanning),
   over the population's score array, so only innovating persons become
   `Person` objects.
4. Transit: done in every mobsim (and every kernel), with every timing
   setting, with RAPTOR in Python and Rust, and checked against Java MATSim
   on pt-tutorial (`verification/`).
5. Money events in scoring, and `SubtourModeChoice`.
6. Validation against Java MATSim: done for equil and pt-tutorial
   (`verification/`); open gaps are listed in `verification/reports/latest.md`.
