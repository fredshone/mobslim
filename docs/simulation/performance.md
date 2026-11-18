# Performance

Simulations are fast and memory light. The following experiments compare MobSlim and MATSim at the *grid-40*, *grid-40-congested*, *grid-80*, *grid-80-congested*, *grid-160* and *grid-160-congested* scenarios, and on MATSim's Open Berlin at
1% (*open-berlin-1pct*) and 10% (*open-berlin-10pct*), on the same plans and routes, on the following machine:

```{include} ../_generated/perf_machine.md
```

## Summary

```{include} ../_generated/perf_headline.md
```

## Speed

* **MobSlim's HermesSim is 4.9–11x faster than MATSim Hermes**; PSim and DSim
  on 16 threads are 2.1–14x faster.
* **On the largest scenarios, PSim on 16 threads is the fastest**: 12x faster
  than MATSim Hermes on Open Berlin at 10% (7.63 s against 91.8 s), and
  8.7–14x on the grid-160 scenarios.
* **Open Berlin at 1% gains least** (HermesSim 4.9x, 1.94 s against 9.54 s):
  with 52,263 agents on 283,885 links, the mobsim's fixed costs per link and
  per stop weigh more than on the grids.
* **MobSlim's QSim and AgentSim are faster than MATSim's QSim on every
  scenario they both ran,** but 1.9–5.0x slower than MATSim Hermes.

```{image} ../_static/showcase/perf_time_light.svg
:class: only-light
:alt: Mobsim time per scenario and simulation
```
```{image} ../_static/showcase/perf_time_dark.svg
:class: only-dark
:alt: Mobsim time per scenario and simulation
```

## Max memory

* **Every MobSlim mobsim needs 52–94% less memory than MATSim Hermes**: 71–94%
  on the grids, 85–87% on Open Berlin at 10%, and 52–65% at 1%, where the
  network and transit schedule, not the agents, take most of it.
* **On Open Berlin at 10%, HermesSim, PSim and DSim peak at 6.26–7.26 GB**,
  against 49.27 GB for MATSim Hermes; on grid-160 at 2.57–3.18 GB, against
  40.68 GB.

```{image} ../_static/showcase/perf_memory_light.svg
:class: only-light
:alt: Peak process memory per scenario and simulation
```
```{image} ../_static/showcase/perf_memory_dark.svg
:class: only-dark
:alt: Peak process memory per scenario and simulation
```

## Scenarios

### grid-40

A 40×40 grid (6,560 links, 600 veh/h) with 26,240 car commuters leaving home
between 06:00 and 08:00: moderate congestion. The grid is slightly irregular:
each node is moved up to 20 m from its grid point, so links are about 160 to
240 m long rather than all 200 m, and routes are not ties between many paths
of equal length. At the busiest minute, 07:17, 2,151 cars are on the road and
only 52 of them are queued.

```{image} ../_static/showcase/net_grid-40_light.png
:class: only-light
:width: 75%
:align: center
:alt: The 40×40 grid at its busiest minute, with almost every car moving
```
```{image} ../_static/showcase/net_grid-40_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The 40×40 grid at its busiest minute, with almost every car moving
```

```{include} ../_generated/perf_grid-40.md
```

### grid-40-congested

The same at flow and storage capacity ×0.3, as if running a 30% sample at
full demand: queues and spillback. At the busiest minute, 08:00, 10,658 cars
are on the road and 9,211 of them are queued.

```{image} ../_static/showcase/net_grid-40-congested_light.png
:class: only-light
:width: 75%
:align: center
:alt: The congested 40×40 grid at its busiest minute, with queues
```
```{image} ../_static/showcase/net_grid-40-congested_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The congested 40×40 grid at its busiest minute, with queues
```

```{include} ../_generated/perf_grid-40-congested.md
```

### grid-80

The grid-40 scenario at four times the size: an 80×80 grid (25,920 links)
with 104,960 car commuters, about the same number per link. Trips are twice as
long, so there is about 8 times the work. At the busiest minute, 08:00, 44,658
cars are on the road and 34,080 of them are queued.

```{image} ../_static/showcase/net_grid-80_light.png
:class: only-light
:width: 75%
:align: center
:alt: The 80×80 grid at its busiest minute
```
```{image} ../_static/showcase/net_grid-80_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The 80×80 grid at its busiest minute
```

```{include} ../_generated/perf_grid-80.md
```

### grid-80-congested

grid-80 at flow and storage capacity ×0.3, as grid-40-congested is to
grid-40. At the busiest minute, 08:00, 73,905 cars are on the road and 67,237
of them are queued.

```{image} ../_static/showcase/net_grid-80-congested_light.png
:class: only-light
:width: 75%
:align: center
:alt: The congested 80×80 grid at its busiest minute, with queues
```
```{image} ../_static/showcase/net_grid-80-congested_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The congested 80×80 grid at its busiest minute, with queues
```

```{include} ../_generated/perf_grid-80-congested.md
```

### grid-160

grid-80 at four times the size again: a 160×160 grid (103,040 links) with
419,840 car commuters, about 8 times the work of grid-80. MATSim gets a 40 GB
heap here (`-Xmx40g`). At the busiest minute, 08:00, 297,581 cars are on the
road and 247,679 of them are queued.

```{image} ../_static/showcase/net_grid-160_light.png
:class: only-light
:width: 75%
:align: center
:alt: The 160×160 grid at its busiest minute
```
```{image} ../_static/showcase/net_grid-160_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The 160×160 grid at its busiest minute
```

```{include} ../_generated/perf_grid-160.md
```

### grid-160-congested

grid-160 at flow and storage capacity ×0.3. At the busiest minute, 08:00,
365,439 cars are on the road and 337,234 of them are queued.

```{image} ../_static/showcase/net_grid-160-congested_light.png
:class: only-light
:width: 75%
:align: center
:alt: The congested 160×160 grid at its busiest minute, with queues
```
```{image} ../_static/showcase/net_grid-160-congested_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The congested 160×160 grid at its busiest minute, with queues
```

```{include} ../_generated/perf_grid-160-congested.md
```

### open-berlin-1pct

[MATSim Open Berlin v6.4](https://github.com/matsim-scenarios/matsim-berlin),
TU Berlin VSP's calibrated model of Berlin and Brandenburg, at its 1% sample:
52,263 agents (residents, commercial traffic and freight) on a network of
119,174 nodes and 283,885 links, with Berlin's buses, trams, U-Bahn, S-Bahn
and regional trains on its timetable. Cars, trucks and freight vehicles drive
on the roads, at flow and storage capacity ×0.01; bike and ride are teleported
along their routes, and pt runs on the schedule. Every agent runs its
calibrated selected plan, with its routes. At the busiest minute, 08:03, 2,545
vehicles are on the road and 1,028 of them are queued. The snapshot shows road
vehicles only.

```{image} ../_static/showcase/net_open-berlin-1pct_light.png
:class: only-light
:width: 75%
:align: center
:alt: Open Berlin at 1% at its busiest minute, with queues in central Berlin
```
```{image} ../_static/showcase/net_open-berlin-1pct_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: Open Berlin at 1% at its busiest minute, with queues in central Berlin
```

```{include} ../_generated/perf_open-berlin-1pct.md
```

### open-berlin-10pct

The same at its 10% sample: 526,111 agents, at capacity ×0.1. MATSim gets a
100 GB heap here (`-Xmx100g`), and only the fast simulations run (MATSim
Hermes and MobSlim's HermesSim, PSim and DSim). At the busiest minute, 17:01,
27,538 vehicles are on the road and 10,871 of them are queued.

```{image} ../_static/showcase/net_open-berlin-10pct_light.png
:class: only-light
:width: 75%
:align: center
:alt: Open Berlin at 10% at its busiest minute, with queues in central Berlin
```
```{image} ../_static/showcase/net_open-berlin-10pct_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: Open Berlin at 10% at its busiest minute, with queues in central Berlin
```

```{include} ../_generated/perf_open-berlin-10pct.md
```

## Caveats

* **The time is the mobsim alone,** not loading, routing or replanning. For
  each simulation it is the fastest of 3 runs, or of 3 iterations for MATSim,
  so the JVM is warmed up. MobSlim's slowest runs are not repeated: QSim and
  AgentSim on both grid-160 scenarios, and DSim on 1 thread on
  grid-160-congested and Open Berlin at 10%, ran once. MATSim feeds its events to its standard handlers
  (travel times, scoring) while it runs. MobSlim runs as in an iteration
  that is not an output iteration (see *What an iteration needs* in
  [the design](../design.md#32-trajectories-and-events-trajectories-events)):
  HermesSim, PSim and DSim sum the link travel times while they simulate,
  and do not record when each link was entered. QSim and AgentSim record
  every link, and the travel times are summed from their trajectories
  afterwards. MobSlim then builds the same travel-time table and scores,
  and **that time is included** in MobSlim's figures.
* **Memory is the whole process's peak resident memory** (Python or the JVM,
  plus the loaded scenario, plus the mobsim). The JVM was allowed a 10 GB heap
  (`-Xmx10g`; 40 GB for grid-160, 20 GB for Open Berlin at 1% and 100 GB at
  10%), and Java grows its heap lazily, so MATSim usually needs less than it
  takes.
* **Open Berlin runs its calibrated selected plans,** one per agent, with
  the routes MATSim chose for them, and no replanning. Open Berlin is by TU
  Berlin's VSP and licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  see Ziemke, Kaddoura and Nagel (2019), *The MATSim Open Berlin Scenario: A
  multimodal agent-based transport simulation scenario based on synthetic
  demand modeling and open data*, Procedia Computer Science 151, 870–877.
  Its inputs are downloaded, not committed.
* **Open Berlin is run as plain MATSim runs it, on both sides.** Its own run
  adds Java (traffic dynamics, scoring, replanning) that the showcase does
  not use; instead:
  * only car, truck and freight are simulated on the roads (`qsim.mainMode`),
    each with its vehicle type's passenger car equivalents; bike and ride are
    teleported along their routes, as MATSim does for network modes that are
    not main modes;
  * MATSim's default queue dynamics (FIFO links), not Open Berlin's kinematic
    waves and passing queues, which MobSlim's mobsims do not have;
  * the access and egress walks of each trip are folded into its leg, so
    MobSlim starts the leg on the activity's link;
  * about 361 bike and ride legs whose activities are on links their mode
    cannot use are re-routed by MobSlim from the nearest node of that mode's
    network;
  * pt passengers board at the visit of the stop MATSim's router chose (the
    S41 and S42 ring lines visit each stop several times per trip).
* **Some agents get stuck on Open Berlin, on both sides**: removed after
  waiting longer than the stuck time, or still travelling at the end of the
  day. At 1%, 158–160 on MATSim (QSim, Hermes) and 244–251 on MobSlim; at
  10%, 1,624 on MATSim Hermes and 2,159–2,247 on MobSlim. MATSim's DSim
  runner does not count them.
* **MATSim's DSim comes from a weekly build** (`2027.0-2026w45`), run with the
  QSim's traffic model.
* **grid-320 (a 320×320 grid, 410,880 links, 1,679,360 car commuters, and
  its congested variant) is defined but not run by default** (`-k 320`): a day
  of it takes MATSim's QSim and DSim, and MobSlim's QSim and AgentSim, hours,
  so only MATSim Hermes (with `-Xmx100g`) and MobSlim's HermesSim, PSim and
  DSim run on it.
* **The multi-threaded runs use all 16 logical CPUs,** on 8 physical cores.
  On the smaller scenarios they gain little over one thread, and sometimes lose;
  on Open Berlin at 1%, PSim on 16 threads is only 1.3x faster than on one
  (3.43 s against 4.60 s). On the grid-160 scenarios and Open Berlin at 10%,
  MobSlim's PSim and DSim are 3.4–4.8x faster on 16 threads than on one, and
  MATSim's DSim 5.1–5.4x on grid-160.
* **Timings depend on the machine.** Run on mains power, with nothing else
  heavy running.

## Reproduce it

```bash
uv run python -m showcase perf        # runs everything, then writes the tables and charts
uv run python -m showcase report      # redraws them from showcase/results/perf.json
uv run python -m showcase networks    # redraws the network snapshots
```

`-k open-berlin` runs only Open Berlin. Its inputs (about 1.2 GB) are
downloaded from VSP's public SVN on the first run, into
`showcase/output/perf/open-berlin-v6.4/download/`, and the selected plans
and the config are written from them.

See `showcase/README.md` for the requirements (Java, Maven, a `matsim-libs`
checkout).
