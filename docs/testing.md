# Testing

```bash
uv run pytest                          # tests + demo notebooks (nbmake) + coverage report
uv run pytest tests -q -o addopts=""   # fast loop: tests only, no coverage
uvx ruff check src tests benchmarks verification
uv run python -m verification run      # against Java MATSim (needs Java + Maven; see verification/README.md)
```

Coverage HTML is written to `reports/coverage/`. The demo notebooks run in a
separate kernel, so `animate.py` and `transit/` show as uncovered even though
the demos exercise them.

## Layout

| File | What it pins down |
|---|---|
| `tests/conftest.py` | Fixtures (`equil_scenario`) and helpers: `chain_network`, `commuters`, `routed`, `simulate`, `make_mobsim`; `MOBSIMS` registers every mobsim the spec runs against |
| `test_network.py` | XML loading, `capperiod` normalisation, parallel links, per-mode adjacency, builders, spatial lookup, pickling |
| `test_population.py` | Plan validation, immutability, `end_time` vs `duration` semantics, plan memory and per-plan scores, population tables (plan store dedup, `NaN` ⇄ `None` scores, read-only arrays, updates that share unchanged tables), XML loading (node / link / x,y locations, scores and selection) |
| `test_planstore.py` | `PlanStore`: round trips, `take` equals encoding the plans, appends at the tip share buffers and keep old snapshots, appends to old snapshots branch, pickling; `Population.replace` appends only new plans (also after pickling and for views of another population), compaction keeps plans and bounds the store, `map_plan_store` with unused rows, `from_tables` checks, `validate_batch` |
| `test_formats.py` | Parquet round trips (plans, scores, selection, routes, every attribute type, pt rides with a schedule), readable tables, shuffled rows, dropped routes, location fallbacks, inconsistent tables rejected; MATSim XML writer round trip (`.gz` too) and when routes are written; time formatting; format detection; the `python -m mobslim.formats` CLI; config `inputs.plans` as Parquet |
| `test_events.py` | Dispatch by type, `EventLog`, MATSim XML writer, event order for a simple trip |
| `test_qsim.py` | **Behavioural spec of the mobsim**: free-speed times, fractional flow headways, per-link (not per-lane) capacity, capacity factors, FIFO by identity, storage/spillback, stuck time vs gridlock, departures onto a full link forced one per stuck time (MATSim's departure buffer), `end_time` aborts, activity timing, teleportation, route validation (no route, broken chain, wrong end node, link without the mode), no mutation, determinism |
| `test_agentsim.py` | AgentSim vs QSim: identical trajectories without spillback; flow headways below the retry interval are exact; blocked vehicles poll (`retry_interval`) where the QSim wakes them; stuck time caps polling; config validation |
| `test_hermes.py` | HermesSim: Rust and Python kernels give identical trajectories and events (congestion levels, finite and infinite stuck time, cut-off runs, teleports, overflow agenda); identical to the QSim with whole-second parameters, close to it otherwise; with every junction-delay and rounding setting, Rust equals Python, the QSim with `continuous_time=False` equals HermesSim move for move (also congested, fractional and with short or no stuck time), and AgentSim does without spillback; whole-second rounding rules; controller learns; backend selection; with transit (random corridor riders and cars, both vehicle modes, blocking stops, capacity, fractional door times; pt-tutorial), identical rides, stop times and vehicle link times; with transit and every junction-delay and rounding setting, the QSim with `continuous_time=False` equals HermesSim (both kernels) move for move |
| `test_dsim.py` | DSim: identical trajectories and events on 1, 2, 3 and 8 workers (free flow, congestion, gridlock, short stuck time, cut-off runs) and on random and METIS partitions (grid, equil), with either buffer-storage rule and with or without junction delays; the seed changes merges only under congestion; buffered vehicles keep their link's storage by default (the two-link loop locks) and free it with `buffer_in_storage=False` (it does not); matsim-rust's link timing and flow capacity at nodes; free flow within one second per link of HermesSim; METIS partitions are balanced and cut few links; PSim's results do not depend on the partitioner; with transit (bus lines crossing the partitions of a grid, both vehicle modes and rail lines on the timetable mixed with buses in traffic, blocking stops, congestion, either timing), identical trajectories, rides, stop times and events on 1, 2, 3 and 8 workers and on random partitions, and close to HermesSim with its timing; bad parameters are rejected |
| `test_psim.py` | PSim: Rust and Python kernels give identical trajectories and events on 1, 2, 3 and 8 threads (free flow, congestion, gridlock, short stuck time, cut-off runs; equil); random partitions give the same results; Rust equals Python with every junction-delay and rounding setting; the Python kernel ignores the partition; free flow equals HermesSim; freed space is reused one second later; `partition_nodes` is complete and balanced; with transit (bus lines crossing the partitions of a grid, both vehicle modes and rail lines on the timetable mixed with buses in traffic, blocking stops, congestion, junction delays), Rust equals Python on 1, 2, 3 and 8 threads and on random partitions, and trip times stay within a second of HermesSim's; bad parameters are rejected |
| `test_trajectories.py` | **Arrays and events agree**: on normal, gridlock and cut-off (mid-link, mid-teleport, mid-activity) runs, each mobsim's direct `Trajectories` equal those rebuilt from its events; experienced plans, vectorised scores and the travel-time table equal their event-based / per-plan counterparts; events are optional; `PlanBatch` layout; per-bin travel-time views are shared |
| `test_routing.py` | Dijkstra (parallel links, time-dependent choice, no-route), travel-time binning, free-speed floor, table blend/pickle, legs ending on the activity's link (both routers, the exceptions, route validation) |
| `test_scoring.py` | Activity/leg utilities, overnight wrap (also after midnight), stuck penalty, experienced plans from a real run; MATSim's `calcActScore` terms by hand: relative/uniform `t_0`, opening times, waiting, late arrival, early departure, minimal duration, daily constants, money per metre, unpriced NaN distances |
| `test_replanning.py` | Selectors, ReRoute / TimeAllocationMutator / ChangeTripMode purity, strategy plan memory, innovation switch-off, stable per-person RNG, **serial == parallel runner** |
| `test_vectorised_replanning.py` | **Vectorised == per-person rules**: every selector's `select_many` and each module's `mutate_batch` make the same decisions as `select` / `mutate` given the same draws; `route_batch` equals `route_plans` (time-dependent travel times, activity links, per-person transit weights); `VectorRunner` equals `SerialRunner` for deterministic strategies, falls back per person for custom modules, mixes both, handles `PersonStrategyManager`, keeps plan memory and store bounds over many iterations, strategy shares follow weights |
| `test_controller.py` | Equil and grid learn (scores up, travel times down), reproducibility, serial == parallel end to end, the vectorised runner is the default and deterministic, listener hook order, `EventRecorder`, output files, unknown modes, score averaging |
| `test_cli.py` | The `mobslim` command: `run` on a mobslim and a MATSim config with `-n`/`-o` overrides, `check` |
| `test_transit.py` | **Behavioural spec of transit** on a synthetic corridor, exact numbers: schedule tables, builder checks, `TransitPassengerRoute`, the ride CSR, activity coordinates for access walks and teleports, RAPTOR choices (transfer vs direct line, line-switch penalty, minimal transfer time, walking when transit cannot help), `route_many == route`, schedule- and network-mode vehicles (queue model, `awaitDeparture`, early running), capacity and denied boardings, serial/parallel dwell, blocking vs non-blocking stops in mixed traffic, stuck after the last departure, events, arrays-vs-events, pt scoring terms, learned transit link times; the timing parameters (stop times with and without junction delays, walks exact or rounded nearest or down); `schedule_modes` (a train keeps its timetable while the bus beside it queues in congestion, and queues too without it); runs against every mobsim (DSim's default timing, a second per junction, is a documented strict xfail where it adds up beyond the tolerance) |
| `test_agentsim_transit.py` | AgentSim's native transit: identical trajectories to the QSim without spillback (pt-tutorial in both vehicle modes; a random mixed-traffic corridor with serial/parallel doors and blocking/non-blocking stops; both also with junction delays and whole seconds), cars displaced by a bus rejoining from a stop, a passenger reopening the doors, PCE in storage; under spillback chain mode equals polling (also with 2.5-PCE buses), PCE occupancy is released, waiting passengers are never polled, stuck after the last trip with `stuck_time=inf`; a wrapper checks that only a link's head ever leaves it |
| `test_transit_scenarios.py` | MATSim's pt-simple, pt-simple-lineswitch and pt-tutorial: XML loading with MATSim's defaults, routes, walk times, vehicle-time windows (on QSim and AgentSim), the full loop with car/pt mode choice, serial == parallel; Sioux Falls (matsim-eth or siouxfalls-2014 files) loading, routing and a HermesSim run with nobody stuck, if `$MOBSLIM_SIOUXFALLS` is set; `RustRaptorRouter` returns the same routes as `RaptorRouter` on random queries under several configs |
| `test_matsim_reference.py` | **Against frozen Java MATSim 2025.0 runs** (`tests/fixtures/matsim_reference/`, written by `python -m verification freeze`): mobslim's scorer gives MATSim's executed scores exactly on MATSim's events (equil 100 and 2000 agents, pt-tutorial); QSim, HermesSim, PSim and DSim on MATSim's plans and routes are no further from MATSim than when frozen (hourly link volumes, link and trip times, pt stop times and boardings) |
| `test_benchmarks.py` | Every benchmark in `benchmarks/` still runs; save/compare and CLI work |
| `demos/*.ipynb` | Executed by nbmake as smoke tests |

## Writing tests for alternative implementations

`test_qsim.py` is the contract for any `Mobsim` (e.g. a Rust port). Its tests
go through the `simulate(scenario, **config)` helper, and an autouse fixture
reruns the whole module once per entry in `tests/conftest.py::MOBSIMS`
(currently `qsim`, `agentsim`, `agentsim_chain`, `hermes_py`, `psim_py` and,
when the Rust extensions are built, `hermes`, `psim` and `dsim`). Its
`test_timing_options` runs one trip under every combination of the timing
parameters a mobsim supports (and checks the others are rejected). A mobsim's
documented deviations from the spec are listed in `test_qsim.py::DEVIATIONS`
and run as strict xfails. Whole-second mobsims are listed in
`conftest.TIME_STEP`; tests that check exact fractional times allow one time
step for them. `test_trajectories.py` does the same, checking that each
mobsim's returned `Trajectories` agree with the events it emits. To test a
new implementation, add `"name": (MobsimClass, ConfigClass)` to `MOBSIMS`. The same
approach works for routers: the tests in `test_routing.py` build routers
directly and only depend on the `Router` protocol.
