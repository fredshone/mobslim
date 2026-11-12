# Examples

Each example uses MATSim's equil scenario from `scenarios/equil/`.

## Run the controller

```python
from mobslim import (Controller, ControllerParams, Scenario, StrategyManager,
                     read_network_xml, read_population_xml)

network = read_network_xml("scenarios/equil/network.xml")
population = read_population_xml("scenarios/equil/plans100.xml", network)
scenario = Scenario(network, population)

result = Controller(
    scenario,
    strategies=StrategyManager.default(),  # 80% ChangeExpBeta, 10% reroute, 10% time mutation
    params=ControllerParams(iterations=20, seed=1),
).run()
print(result.metrics_frame()[["score_executed", "score_best"]].tail(3))
```

## Weave

`Weave` takes every `Controller` argument, plus the number of strands and
the share of persons innovating in each strand:

```python
from mobslim import HermesSim, Weave

if __name__ == "__main__":  # strands run in worker processes
    result = Weave(
        scenario,
        strands=4,             # four simulations per iteration
        innovation_rate=0.1,   # 10% innovate in each: 40% of persons per iteration
        mobsim=HermesSim(),
        params=ControllerParams(iterations=20, seed=1),
    ).run()
    m = result.metrics_frame()
    print(m[["score_performed", "coverage", "best_plan_changed"]].tail(3))
```

Without `innovation_rate`, the rate is the innovative strategies' share of
the strategy weights (20% for `StrategyManager.default()`), so an existing
setup carries over unchanged.

`score_performed` is the mean utility of everything simulated in the
iteration, over all strands. `coverage` is innovations per person.
`result.strand_metrics[i]` holds iteration `i`'s per-strand metrics.

## Choose where strands run

By default, strand 0 runs in the main process and the others run in
`strands - 1` worker processes, capped at one fewer than the physical cores
(further strands queue). Pass a loom to change that:

```python
from mobslim.weave import ProcessLoom, SerialLoom

Weave(scenario, strands=4, loom=SerialLoom())                # all in this process
Weave(scenario, strands=8, loom=ProcessLoom(workers=3))      # 8 strands on 3 workers + main
```

## Don't wait for the slowest strand

`AsyncWeave` takes the same arguments as `Weave`, but never waits for all
its strands at once. It merges them one at a time, in a fixed order, and
gives each strand its next job as soon as its result is merged, so no core
idles while the slowest strand finishes or while the main process merges:

```python
from mobslim import AsyncWeave

AsyncWeave(scenario, strands=8, innovation_rate=0.1).run()
```

The order of merges is fixed, so the same seed gives the same result on any
loom. Every strand after iteration 0 runs in a worker (by default one per
strand, up to the physical cores), so listeners see events and trajectories
in iteration 0 only. A strand's results are merged after the other strands'
merges since its job was built: `dropped_observations` counts the scores of
plans pruned meanwhile, which are dropped. Each merge pools the last
`strands` strands' travel times, so `travel_time_smoothing` applies per
merge rather than per iteration.

## Tune who innovates, and when

```python
from mobslim.weave import LinearRate, PrecisionReducer, WeightedStalenessScheduler

Weave(
    scenario,
    strands=8,
    schedule=LinearRate(0.1, 0.0, begin=50, finish=90),       # anneal innovation to zero
    scheduler=WeightedStalenessScheduler(alpha=2.0),          # favour the long-waiting more
    reducer=PrecisionReducer(max_count=10),                   # count-weighted scores
    travel_time_smoothing=0.5,                                # average travel times over iterations
)
```

`alpha=0` picks innovators uniformly at random. A large `alpha` picks the
persons who have waited longest first.

## From a config

Add a `weave` section to a mobslim YAML config:

```yaml
weave:
  strands: 8
  innovation_rate: 0.1     # null: the innovative strategies' share
  alpha: 1.0
  reducer: mean            # or precision
  workers: null            # null: strands - 1, at most physical cores - 1; 0: all in the main process
  travel_time_smoothing: 1.0
```

Or set the number of strands on the command line, for a YAML or a MATSim
config:

```bash
mobslim run scenarios/equil/mobslim.yaml --strands 8 --innovation-rate 0.1
```

In Python, `Weave.from_mobslim_config(path)` and
`Weave.from_matsim_config(path, strands=8)` work like their `Controller`
counterparts.
