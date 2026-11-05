# Test fixtures

| Directory | Source | Notes |
|---|---|---|
| `equil/` | MATSim `examples/scenarios/equil` | Car only |
| `pt_simple/` | MATSim `examples/scenarios/pt-simple` | One line, one stop pattern, one departure, one passenger. Same files as `scenarios/pt-simple` |
| `pt_lineswitch/` | MATSim `examples/scenarios/pt-simple-lineswitch` | Three lines. The passenger's second trip needs a walking transfer (stop `47` to stop `910`, 20 m). `transitschedule.xml` was renamed to `transitSchedule.xml` |
| `pt_tutorial/` | MATSim `examples/scenarios/pt-tutorial` | 900 persons on a car grid plus one train line (two routes, 50 departures each) on separate `train` links. `multimodalnetwork.xml` was renamed to `network.xml`, `transitschedule.xml` to `transitSchedule.xml`, and `population.xml` was gzipped |
| `matsim_reference/` | Java MATSim 2025.0 runs of `equil` (100 agents; 2,000 agents at half capacity) and `pt-tutorial`, written by `python -m verification freeze` | One directory per case. Each holds MATSim's iteration-0 `output_events.xml.gz` and `output_plans.xml.gz`, the inputs mobslim reads (`config.xml` with relative paths, the network and the transit files), and `expected.json`, the metrics mobslim reached when frozen. Used by `tests/test_matsim_reference.py`. Regenerate after an intended behaviour change (see `verification/README.md`) |

The MATSim files come from <https://github.com/matsim-org/matsim-libs>
(GPL-2.0), downloaded 2026-11-04 from the `main` branch. They are used
unchanged apart from the renames listed above.

Sioux Falls 2014 (`examples/scenarios/siouxfalls-2014`: 2,040 bus departures
on 10 lines, sharing `car,bus,pt` links with cars) is too large to commit. Its
tests and benchmarks look for it in `$MOBSLIM_SIOUXFALLS` and are skipped if
that is unset (see `TRANSIT_PLAN.md`).
