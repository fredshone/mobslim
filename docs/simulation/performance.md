# Performance

Simulations are fast and memory light. The following experiments compare MobSlim and MATSim at the *grid-40*, *grid-40-congested*, *grid-80*, *grid-80-congested*, *grid-160*, *grid-160-congested* and *siouxfalls* scenarios, on the same plans and routes, on the following machine:

```{include} ../_generated/perf_machine.md
```

## Summary

```{include} ../_generated/perf_headline.md
```

## Speed

* **MobSlim's HermesSim is 6.3–11x faster than MATSim Hermes**; PSim and DSim
  are 1.1–14x faster.
* **On the largest scenarios, grid-160 and grid-160-congested, PSim on 16
  threads is the fastest**: 8.7–14x faster than MATSim Hermes (8.01 s against
  109 s on grid-160).
* **MobSlim's QSim and AgentSim are faster than MATSim's QSim on every
  scenario,** but 1.9–4.9x slower than MATSim Hermes.

```{image} ../_static/showcase/perf_time_light.svg
:class: only-light
:alt: Mobsim time per scenario and simulation
```
```{image} ../_static/showcase/perf_time_dark.svg
:class: only-dark
:alt: Mobsim time per scenario and simulation
```

## Max memory

* **Every MobSlim mobsim needs 71–94% less memory than MATSim Hermes.**
* **On grid-160, HermesSim, PSim and DSim peak at 2.57–3.18 GB**, against
  40.68 GB for MATSim Hermes and 15.25 GB for MATSim's QSim.

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

### siouxfalls

MATSim's Sioux Falls 2014: 84,110 agents travelling by car and by bus, with
2,040 bus departures driving in the same traffic as the cars. At the busiest
minute, 17:42, 4,117 cars are on the road and 2,378 of them are queued. The
snapshot shows cars only.

```{image} ../_static/showcase/net_siouxfalls_light.png
:class: only-light
:width: 60%
:align: center
:alt: Sioux Falls at its busiest minute, with queues on a few main roads
```
```{image} ../_static/showcase/net_siouxfalls_dark.png
:class: only-dark
:width: 60%
:align: center
:alt: Sioux Falls at its busiest minute, with queues on a few main roads
```

```{include} ../_generated/perf_siouxfalls.md
```

## Caveats

* **The time is the mobsim alone,** not loading, routing or replanning. For
  each simulation it is the fastest of 3 runs, or of 3 iterations for MATSim,
  so the JVM is warmed up. MobSlim's slowest runs are not repeated: QSim and
  AgentSim on both grid-160 scenarios, and DSim on 1 thread on
  grid-160-congested, ran once. MATSim feeds its events to its standard handlers
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
  (`-Xmx10g`; 40 GB for grid-160), and Java grows its heap lazily, so MATSim usually needs less
  than it takes.
* **Sioux Falls uses plans that MATSim learned over 10 iterations.** The
  shipped iteration-0 plans jam the network on every mobsim. MATSim Hermes
  learned them with 30% of agents innovating per iteration, and then every
  simulation ran on those plans.
* **MobSlim's AgentSim leaves 6,879 Sioux Falls agents still travelling at
  the end of the day.** No other MobSlim mobsim leaves any. AgentSim's
  junction priority differs from the QSim's once queues spill back (see
  [design detail](design-detail.md)).
* **MATSim's DSim comes from a weekly build** (`2027.0-2026w45`), run with the
  QSim's traffic model.
* **grid-320 (a 320×320 grid, 410,880 links, 1,679,360 car commuters, and
  its congested variant) is defined but not run by default** (`-k 320`): a day
  of it takes MATSim's QSim and DSim, and MobSlim's QSim and AgentSim, hours,
  so only MATSim Hermes (with `-Xmx100g`) and MobSlim's HermesSim, PSim and
  DSim run on it.
* **The multi-threaded runs use all 16 logical CPUs,** on 8 physical cores.
  On the smaller scenarios they gain little over one thread, and sometimes lose;
  on the grid-160 scenarios MobSlim's PSim and DSim are 3.4–4.8x faster on 16
  threads than on one, and MATSim's DSim 5.1–5.4x.
* **Timings depend on the machine.** Run on mains power, with nothing else
  heavy running.

## Reproduce it

```bash
uv run python -m showcase perf        # runs everything, then writes the tables and charts
uv run python -m showcase report      # redraws them from showcase/results/perf.json
uv run python -m showcase networks    # redraws the network snapshots
```

See `showcase/README.md` for the requirements (Java, Maven, a `matsim-libs`
checkout).
