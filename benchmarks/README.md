# Benchmarks

A quick benchmark suite for trying design changes. The full suite takes
about 9 s on the equil scenario.

```bash
uv run python -m benchmarks                        # run everything
uv run python -m benchmarks --save base            # ...and save as a baseline
# change something
uv run python -m benchmarks --compare base         # times relative to the baseline
uv run python -m benchmarks -k mobsim -k routing   # only matching benchmarks
uv run python -m benchmarks --profile mobsim.qsim  # cProfile one benchmark
uv run python -m benchmarks --profile-memory mobsim.qsim  # where it allocates
uv run python -m benchmarks --no-memory            # timings only (faster)
uv run python -m benchmarks --list
```

Results are saved to `benchmarks/results/<name>.json` (git-ignored, since
timings are machine-specific) together with the git commit, whether the
tree was dirty, and the Python version and machine. Pass a path ending in
`.json` to save or compare somewhere else.

## What is measured

Each benchmark times one stage. Inputs are built once beforehand and are
not timed.

| Benchmark | Times |
|---|---|
| `load.network` | `read_network_xml` |
| `load.population` | `read_population_xml` |
| `init.free_speed_table` | `TravelTimeTable.free_speed` |
| `init.route_scenario` | routing all legs at free speed (`Controller.prepare`) |
| `mobsim.qsim_no_events` | one QSim run with no event handlers: trajectories only, no event objects (the controller's default) |
| `mobsim.qsim` | one QSim run with a single `EventLog` handler |
| `mobsim.qsim_with_collectors` | the controller's whole mobsim stage: one QSim run, then the travel-time table and experienced plans from the trajectories |
| `mobsim.agentsim_no_events` | one AgentSim run with no event handlers (compare with `mobsim.qsim_no_events`) |
| `mobsim.agentsim` | one AgentSim run with a single `EventLog` handler (compare with `mobsim.qsim`) |
| `mobsim.agentsim_congested_poll` | one AgentSim run under heavy spillback (storage ×0.002, flow ×0.01, 120 s stuck time), blocked agents polling |
| `mobsim.agentsim_congested_chain` | the same run with `blocked="chain"`: identical results, far fewer retries |
| `mobsim.hermes_no_events` | one HermesSim run (Rust kernel if built) with no event handlers |
| `mobsim.hermes` | one HermesSim run with a single `EventLog` handler |
| `mobsim.hermes_congested` | HermesSim under the same heavy spillback as `agentsim_congested_*` |
| `mobsim.hermes_py_no_events` | HermesSim's Python reference kernel, no event handlers |
| `events.travel_times` | replaying recorded events into `TravelTimeCollector` and building the table |
| `events.experienced_plans` | replaying recorded events into `ExperiencedPlanCollector` |
| `events.trajectories` | replaying recorded events into `TrajectoryCollector` (the path for event-only mobsims) |
| `trajectories.travel_times` | travel-time table from the trajectory arrays (numpy) |
| `trajectories.experienced_plans` | building `ExperiencedPlan` objects from the trajectory arrays |
| `scoring.charypar_nagel` | scoring every experienced plan, one at a time |
| `scoring.charypar_nagel_vectorised` | scoring every person at once from the trajectories (numpy) |
| `scoring.vectorised_subpops16` | the same with 16 subpopulations spread over the persons (one pass whatever their number) |
| `scoring.person_specific` | the same with person rules: income-dependent money and per-person car constants |
| `scoring.apply_and_metrics` | `apply_scores` + `iteration_metrics` |
| `routing.dijkstra` | one time-dependent query per car trip, on congested travel times (Python) |
| `routing.dijkstra_many` | the same trips with `DijkstraRouter.route_many` (shared searches) |
| `routing.rust_1thread` | the same trips with `RustRouter.route_many` on one thread |
| `routing.rust` | the same trips with `RustRouter.route_many` on every CPU |
| `routing.rust_alt_1thread` | the same trips with `RustRouter`'s ALT (A*, landmarks) on one thread |
| `replanning.default_strategies` | one `SerialRunner` step with `StrategyManager.default()` |
| `replanning.default_strategies_vectorised` | the same step with the `VectorRunner` (the controller's default) |
| `replanning.reroute_all` | `Reroute` on every selected plan, plan by plan, Python router (the most routing replanning can do) |
| `replanning.reroute_all_batched` | the same, batched as the runners do it (`route_plans`), default router |
| `replanning.reroute_all_vectorised` | the same on columns (`clear_routes_batch` + `route_batch`), as the `VectorRunner` does it |
| `transit.raptor` | routing every pt leg with the Python `RaptorRouter.route_many_xy` (one search per shared origin and time) |
| `transit.raptor_rust_1thread` | the same legs with `RustRaptorRouter` on one thread (same routes) |
| `transit.raptor_rust` | the same legs with `RustRaptorRouter` on every CPU |
| `transit.qsim_schedule` / `transit.qsim_network` | one QSim run, transit vehicles on the timetable / in the queue model |
| `transit.hermes_schedule` / `transit.hermes_network` | the same with HermesSim (Rust kernel if built) |
| `transit.hermes_py_network` | HermesSim's Python reference kernel, queue-model vehicles |
| `run.controller` | `Controller.run`, 10 iterations, from a loaded scenario |
| `run.end_to_end` | load XML, then `Controller.run` for 10 iterations |

The `transit.*` benchmarks need a scenario with a schedule:
`-s pt_tutorial` (MATSim's pt-tutorial, bundled in `tests/fixtures`) or
`-s siouxfalls`, which is registered when `$MOBSLIM_SIOUXFALLS` points at a
directory with the files of
[matsim-eth/sioux-falls](https://github.com/matsim-eth/sioux-falls/tree/master/src/main/resources/ch/ethz/matsim/sioux_falls)
(`network.xml`, `population.xml.gz`, `schedule.xml`, `vehicles.xml`) or
MATSim's `examples/scenarios/siouxfalls-2014`. On other
scenarios they are skipped, and the AgentSim benchmarks are skipped on
transit scenarios (AgentSim does not run transit).

```bash
uv run python -m benchmarks -s pt_tutorial -k transit
MOBSLIM_SIOUXFALLS=~/Data/sioux_falls \
    uv run python -m benchmarks -s siouxfalls -k transit -k mobsim.hermes
```

When `run.controller` is selected, the report also splits that run into
stages (prepare, replanning, mobsim, scoring, travel times and metrics),
timed with a `ControllerListener`. Use it to see which stage to work on.

## Reading the report

```
benchmark                             min      median     ±stdev  rounds         vs base
mobsim.qsim                       5.04 ms     5.16 ms      2.1%      93         0.62x ▼
```

* **min** is the time per call that comparisons use; it is the least noisy
  number for CPU-bound code. **median** and **±stdev** (relative) show
  noise. Fast calls are repeated in an inner loop so each sample lasts at
  least 5 ms.
* **vs base** is `min / baseline min`. `▼` means faster, `▲` slower, when
  the change is above `--threshold` (default 10%).
* **results** lists a few numbers per benchmark, such as event counts and
  final scores. They show whether a change kept the simulation's
  behaviour. A refactor or speed-up should leave them alone; `CHANGED`
  marks a difference from the baseline.

## Memory

After timing, each benchmark is called once more under `tracemalloc`
(tracing slows Python down several times, so it never overlaps the timing
samples). The report gets three more columns:

* **peak mem**: the most memory in use at once during the call, above
  what was allocated before it. Temporaries count here.
* **retained**: what is still allocated when the call returns, with its
  return value alive: the result itself plus anything the call cached.
  Something that grows here run after run is a leak.
* **mem vs base**: `peak / baseline peak`, marked like `vs base`. Baselines
  saved before memory was measured show `new`.

Because the memory call follows the warm-up, caches the first call
filled (for example a travel-time table's per-bin views) are not counted
again. tracemalloc sees Python objects and numpy arrays, but not memory
allocated by other C/Rust extensions; for those, the report's last line
gives the process's peak RSS, which is also saved with the results
(`meta.max_rss`).

`--profile-memory NAME` runs one benchmark under tracemalloc and lists the
source lines whose allocations are still alive after the call, largest
first. `--memory-by filename` groups them by file, and `--memory-by
traceback` shows the full call stack of each site. `--top` sets the number
of rows.

`--no-memory` skips the memory call when you only need times.

`--budget` sets the seconds of timing per benchmark (default 0.5). Raise it
for steadier numbers, and lower it for a quicker check.

## Adding benchmarks and scenarios

A benchmark is a setup function registered in `suite.py`. It receives the
`Workload`, does any untimed preparation, and returns the callable to time:

```python
@benchmark("routing.my_router", info=lambda routes: {"routes": len(routes)})
def _my_router(w: Workload):
    """One query per car trip with MyRouter."""
    trips, router = w.trips(), MyRouter(w.network, "car", w.travel_time)
    return lambda: [router.route(o, d, t) for o, d, t in trips]
```

`Workload` caches the intermediate values of one iteration: `network`,
`population`, `scenario`, `routed`, `events`, `experienced`,
`travel_time`, `scores`, `scored`, plus `context()` and `trips()`.

A scenario is a `ScenarioSpec` in `scenarios.py`: a name, a description,
a network loader, a population loader and the iteration count for the
`run.*` benchmarks. `xml_scenario(...)` covers MATSim XML files. For
synthetic scenarios (network size, population size, congestion), register
a spec whose loaders call `grid_network` and build the population in code,
then pick it with `--scenario`. The benchmarks need no changes.

`tests/test_benchmarks.py` runs every benchmark once, so the suite keeps
working as the API changes.

## Mobsim scaling sweep

`benchmarks/mobsim_sweep.py` runs every mobsim variant on synthetic grids of
three sizes and four congestion levels, with `stuck_time=inf`. Each run gets
its own subprocess, so peak RSS is clean. `mobsim_sweep_report.py` draws the
charts. Results and discussion: [reports/mobsim_sweep.md](reports/mobsim_sweep.md).

```bash
uv run python -m benchmarks.mobsim_sweep                    # resumes; skips cells already in the JSON
uv run python -m benchmarks.mobsim_sweep --variants hermes --grids 40
uv run python -m benchmarks.mobsim_sweep_report
```
