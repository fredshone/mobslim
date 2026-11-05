# Verification against Java MATSim

Measures how close mobslim is to MATSim 2025.0 on MATSim's own example
scenarios (equil and pt-tutorial), in three layers:

| Layer | Compares | Expected |
|---|---|---|
| **L1 scoring** | the same executed day scored by both scorers, in both directions | exact |
| **L2 mobsim** | every mobslim mobsim against MATSim's QSim (and Hermes) on MATSim's plans *and routes*, one iteration | close |
| **L3 loop** | the whole co-evolutionary loop, several seeds per side | inside MATSim's seed spread |

The design, the known differences and the findings are in
[`VERIFICATION_PLAN.md`](../VERIFICATION_PLAN.md); the latest numbers in
[`reports/latest.md`](reports/latest.md).

```bash
uv run python -m verification list
uv run python -m verification run                 # every case, then the report
uv run python -m verification run -k l1 -k l2     # by layer, or by name substring
uv run python -m verification run --mobsim qsim   # only some mobslim mobsims (L2)
uv run python -m verification run --force-matsim  # rerun MATSim even if cached
uv run python -m verification report              # rewrite the report from saved results
uv run python -m verification freeze              # refresh tests/fixtures/matsim_reference
```

**Requirements.** Java 21+ and Maven. The runner in `java/` is built on first
use (`mvn package`) against `org.matsim:matsim:2025.0`. Scenarios are read from
`~/projects/matsim-libs/examples/scenarios`; set `$MATSIM_LIBS` for another
checkout. L1 and L2 take a few minutes; L3 (3 cases x 5 seeds x 51 iterations
per side) about half an hour, almost all of it MATSim.

**Caching.** MATSim outputs go to `output/<case>/` (git-ignored). A run is
reused while its `manifest.json` key matches: MATSim version, the written
config, overrides and the Java sources.

**Without Java.** `tests/test_matsim_reference.py` reruns L1 (MATSim's events
scored by mobslim: exact) and L2 (QSim and HermesSim on MATSim's plans: no
further from MATSim than when frozen) on frozen MATSim runs in
`tests/fixtures/matsim_reference/`. After an intended behaviour change,
`python -m verification freeze` refreshes them (look at the report first).

## Layout

| File | Role |
|---|---|
| `java/RunCase.java` | MATSim's `Controler` on a config Python wrote (plus `--set module.param=value`) |
| `java/ScoreEvents.java` | MATSim's `EventsToScore` (Charypar-Nagel) on any events file |
| (`mobslim.config.matsim`) | Load/edit/write a MATSim `config.xml`; translate it into mobslim objects, listing what mobslim can't honour (moved into the package) |
| `matsim.py` | Build the runner, run MATSim (cached), score events with MATSim |
| `cases.py` | The cases: scenario, config edits, layer, seeds, references |
| `events_io.py` | `events_v1` XML -> link traversals, trips, activities, transit stop calls. Used for **both** sides |
| `experienced.py` | MATSim events -> mobslim `ExperiencedPlan`s (L1) |
| `matsim_events.py` | mobslim events -> what MATSim's scoring reads (links on activities, vehicle events, pt trips expanded) |
| `pt_routes.py` | MATSim's `default_pt` routes -> mobslim `PtRoute`s (L2 rides MATSim's routes) |
| `compare.py` | Metrics (scores, GEH, link and trip times, transit) and their green/amber/red bands |
| `runner.py` | `run_l1`, `run_l2`, `run_l3` |
| `report.py` | `reports/latest.md` (+ L3 plots) |
| `freeze.py` | `tests/fixtures/matsim_reference/` and the metrics pytest guards |
