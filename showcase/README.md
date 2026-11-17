# Showcase

Generates the evidence behind the documentation pages:

| Page | Command | Writes |
|---|---|---|
| [Comparison to MATSim](../docs/simulation/matsim.md) | `uv run python -m showcase correctness` | `docs/_static/showcase/corridor-data.js` (and prints a summary table per variant) |
| [Performance](../docs/simulation/performance.md) | `uv run python -m showcase perf` (then `report`) | `showcase/results/perf.json`, `docs/_generated/perf_*.md`, `docs/_static/showcase/perf_*.svg`; `networks`: `docs/_static/showcase/net_*.png` |
| [How the mobsims work](../docs/simulation/overview.md) | hand-written | |
| [Optimisation: performance](../docs/optimisation/performance.md) | `uv run python -m showcase weaving` (then `weaving-report`) | `showcase/results/weaving.json`, `docs/_generated/weaving_*.md`, `docs/_static/showcase/weaving_*.svg` |
| [Correctness: roundabouts](../docs/simulation/correctness/roundabouts.md) | `uv run python -m showcase opendd` | `showcase/results/opendd_<site>.json`, `replay-opendd-<site>-sim.js`; `replay-opendd-<site>-obs.js` (observed tracks, git-ignored) |
| [Correctness: highways](../docs/simulation/correctness/highways.md) | `uv run python -m showcase tgsim`, `tgsim-i90` | `showcase/results/tgsim.json`, `tgsim_i90.json`, `replay-tgsim-*-sim.js`; `replay-tgsim-*-obs.js` (observed tracks, git-ignored) |

Every replay (the Correctness pages) runs HermesSim only, on observed
demand, with rules of thumb for every link (`replay.ROUNDABOUT`,
`replay.FREEWAY`): nothing is fitted. Each run also rewrites
`docs/_generated/correctness_headline.md` and `correctness_groups.md`, the
tables on the Correctness pages; `python -m showcase replay-report` rewrites
them from `showcase/results/` alone.

`python -m showcase perf -k grid` runs only the matching cases. The grid-320
cases run only when asked for by size (`-k 320`), and then only MATSim Hermes
and MobSlim's HermesSim, PSim and DSim;
`python -m showcase report` redraws the tables and charts from the saved
results without rerunning anything. `python -m showcase networks` redraws the network snapshots
(a few minutes; it simulates each case once). `python -m showcase all` does everything.

**Requirements.** Java 25+ (a JDK, not just a JRE: MATSim's DSim weekly
build needs 25, the rest 21), Maven and a `matsim-libs` checkout, as for
`verification/` (see its README): the MATSim side reuses its runner and
cache. Sioux Falls comes from `matsim-libs/examples/scenarios/siouxfalls-2014`.
The grids are slightly irregular (each node moved up to 10% of the spacing,
so link lengths vary from about 160 to 240 m); their inputs are remade when
the case or the generator changes. The full `perf` run takes several hours on
an 8-core desktop, most of it the grid-160 cases (about 8 times the work of
grid-80), plus time to prepare the inputs. Do not run anything
else heavy at the same time.

## Layout

| File | Role |
|---|---|
| `corridor.py` | The showcase corridor scenario in two variants, `congested` (77 cars) and `free-flow` (20 cars), with the same network, buses and 19 passengers; written as MATSim XML to `scenarios/corridor/<variant>/` |
| `correctness.py` | Per variant: runs MATSim QSim, Hermes and DSim, then every MobSlim mobsim (on MATSim's clock) on MATSim's routed plans; parses all the events the same way; exports the animation data (all eight) |
| `replay.py` | What the replays share: HermesSim on observed demand as a table of link times, the link rules, the animation export (vehicles placed from link times, drawn beyond the view too) and the Correctness page's headline table |
| `perf.py` | The speed and memory cases; prepares routed inputs, runs every variant in a fresh process under `/usr/bin/time` |
| `dsim.py`, `java/dsim/` | MATSim's DSim, which is only in weekly builds: `pom.xml` pins `2027.0-2026w45` (used only for its classpath) and `DSimRun.java` runs it in one JVM, in place of `RunCase` and `PerfRun` |
| `java/src/PerfRun.java` | MATSim with no replanning and no output, timing each iteration's mobsim (compiled with `javac` against the verification runner's classpath) |
| `networks.py` | One HermesSim run per performance case, drawn as a network snapshot: every car at the busiest minute, moving or queued |
| `weaving/experiment.py` | Weaving on `grid-40-congested` (0.3 capacity; inputs made by `perf.prepare` when missing or out of date): 1 minute of wall-clock time per run at 30% innovation under a budget of 8 processes; the Controller, HermesSim with 1-8 strands, `AsyncWeave` with 2-8 strands, and MATSim Hermes; one fresh process per run, recording every mobsim run's start and end. `benchmarks/weave_profile.py` runs the same setup for a few iterations and prints where each iteration's time goes |
| `java/src/ReplanRun.java` | MATSim with replanning for a fixed wall-clock time: per iteration, the mean executed score and the mobsim's start and end |
| `weaving/report.py` | Tables and SVG charts from `results/weaving.json`: each run's mobsim runs in its first 10 s (cadence) above utility over the whole run |
| `opendd/` | openDD: the roundabouts rdb1 to rdb7 (CC BY-ND 4.0, not committed; unzip `opendd_v3-rdb<n>.zip` into `data/opendd/`; each roundabout downloaded is run). `load.py` reads the tracks and lane map and cuts lanes at the camera's view; `match.py` matches each vehicle to a lane path (Viterbi) with enter and exit times; `network.py` holds what each roundabout's map doesn't say (`SITES`: view cuts, lanes out of view, the animated recording) and builds the in-view lanes as a MobSlim network; `demand.py` turns vehicles into one-trip persons; `compare.py` compares time in view; `run.py` matches every recording (cached in `output/opendd/`) and evaluates the held-out recordings (`r_1_val`, or `r_C` for rdb3); `export.py` writes the animation data |
| `results/opendd_<site>.json` | The last openDD comparison, per roundabout: aggregate numbers only, committed so the docs build without the data |
| `tgsim/` | TGSIM I-395: two hours of a freeway merge (USDOT public data, not committed; save the trajectories CSV as `data/tgsim/I395_trajectories.csv`). `load.py` reads it (numbers have thousands separators; cached as parquet); `network.py` places the node lines and builds one multi-lane link per stretch of road (the merge link with three lanes of capacity, its end's); `demand.py` gives each vehicle its route and line-crossing times, and turns one minute (after a minute's warm-up) into persons; `compare.py` compares time in view; `run.py` evaluates every minute; `export.py` writes the animation data |
| `results/tgsim.json` | The last TGSIM comparison: aggregate numbers only, committed so the docs build without the data |
| `tgsim_i90/` | TGSIM I-90/I-94 Stationary: six runs at two sites, a diverge (1, 3, 5) and a weave (2, 4, 6), left lanes queued beside flowing right lanes (USDOT public data, not committed; save the trajectories CSV as `data/tgsim_i90/I90_94_stationary_trajectories.csv`). `load.py` turns each run into road coordinates; `network.py` builds the section as two roads (one per lane group), each with an exit link (the left lanes' at the flow observed leaving them) and a sink link; `demand.py` and `compare.py` as for I-395, per lane group; `run.py` measures each run's section and evaluates every minute; `export.py` writes the animation data |
| `results/tgsim_i90.json` | The last TGSIM I-90/I-94 comparison: aggregate numbers only, committed so the docs build without the data |
| `perf_report.py` | Markdown tables and SVG charts (light and dark) from `results/perf.json` |
| `results/weaving.json` | The last weaving results (machine-specific; committed so the docs build) |
| `results/perf.json` | The last measured results, committed so the docs build without Java (machine-specific) |
