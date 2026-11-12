# Performance

Simulations are fast and memory light. The following experiments compare MobSlim and MATSim at the *grid-40*, *grid-40-congested*, *grid-80*, *grid-80-congested* and *sioux-falls scenarios*, on the same plans and routes, on the following machine:

```{include} ../_generated/perf_machine.md
```

## Summary

```{include} ../_generated/perf_headline.md
```

## Speed

* **MobSlim's HermesSim is 8.5–14x faster than MATSim Hermes**; PSim and DSim are 1.2–19x faster.
* **On the largest scenarios, grid-80 and grid-80-congested, PSim on 16 threads is the fastest**:
  13–19x faster than MATSim Hermes.

```{image} ../_static/showcase/perf_time_light.svg
:class: only-light
:alt: Mobsim time per scenario and simulation
```
```{image} ../_static/showcase/perf_time_dark.svg
:class: only-dark
:alt: Mobsim time per scenario and simulation
```

## Max memory

* **Every MobSlim mobsim needs 71–91% less memory than MATSim Hermes**.

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

A 40×40 grid (6,560 links, 200 m, 600 veh/h) with 26,240 car commuters
leaving home between 06:00 and 08:00: moderate congestion. At the busiest
minute, 07:17, 2,145 cars are on the road and almost none are queued.

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

The same at flow and storage capacity ×0.1, as if running a 10% sample at
full demand: queues and spillback across the network. At the busiest minute,
08:00, 11,661 cars are on the road and 10,162 of them are queued.

```{image} ../_static/showcase/net_grid-40-congested_light.png
:class: only-light
:width: 75%
:align: center
:alt: The congested 40×40 grid at its busiest minute, with queues on most links
```
```{image} ../_static/showcase/net_grid-40-congested_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The congested 40×40 grid at its busiest minute, with queues on most links
```

```{include} ../_generated/perf_grid-40-congested.md
```

### grid-80

The grid-40 scenario at four times the size: an 80×80 grid (25,920 links)
with 104,960 car commuters, the same number per link. Trips are twice as
long, so there is about 8 times the work. At the busiest minute, 15:50,
16,441 cars are on the road and only 32 are queued.

```{image} ../_static/showcase/net_grid-80_light.png
:class: only-light
:width: 75%
:align: center
:alt: The 80×80 grid at its busiest minute, with almost every car moving
```
```{image} ../_static/showcase/net_grid-80_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The 80×80 grid at its busiest minute, with almost every car moving
```

```{include} ../_generated/perf_grid-80.md
```

### grid-80-congested

grid-80 at flow and storage capacity ×0.1, as grid-40-congested is to
grid-40. At the busiest minute, 08:00, 75,431 cars are on the road and
68,507 of them are queued.

```{image} ../_static/showcase/net_grid-80-congested_light.png
:class: only-light
:width: 75%
:align: center
:alt: The congested 80×80 grid at its busiest minute, with queues on most links
```
```{image} ../_static/showcase/net_grid-80-congested_dark.png
:class: only-dark
:width: 75%
:align: center
:alt: The congested 80×80 grid at its busiest minute, with queues on most links
```

```{include} ../_generated/perf_grid-80-congested.md
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
  so the JVM is warmed up. MATSim feeds its events to its standard handlers
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
  (`-Xmx10g`), and Java grows its heap lazily, so MATSim usually needs less
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
* **The multi-threaded runs use all 16 logical CPUs,** on 8 physical cores.
  On the smaller scenarios they gain little over one thread, and sometimes lose;
  on the grid-80 scenarios they are 2.4–4.3x faster than one thread.
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
