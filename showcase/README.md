# Showcase

Generates the evidence behind three documentation pages:

| Page | Command | Writes |
|---|---|---|
| [Correctness](../docs/simulation/correctness.md) | `uv run python -m showcase correctness` | `docs/_static/showcase/corridor-data.js` (and prints a summary table per variant) |
| [Performance](../docs/simulation/performance.md) | `uv run python -m showcase perf` (then `report`) | `showcase/results/perf.json`, `docs/_generated/perf_*.md`, `docs/_static/showcase/perf_*.svg`; `networks`: `docs/_static/showcase/net_*.png` |
| [How the mobsims work](../docs/simulation/overview.md) | hand-written | |
| [Optimisation: performance](../docs/optimisation/performance.md) | `uv run python -m showcase weaving` (then `weaving-report`) | `showcase/results/weaving.json`, `docs/_generated/weaving_*.md`, `docs/_static/showcase/weaving_*.svg` |

`python -m showcase perf -k grid` runs only the matching cases;
`python -m showcase report` redraws the tables and charts from the saved
results without rerunning anything. `python -m showcase networks` redraws the network snapshots
(a few minutes; it simulates each case once). `python -m showcase all` does everything.

**Requirements.** Java 25+ (a JDK, not just a JRE: MATSim's DSim weekly
build needs 25, the rest 21), Maven and a `matsim-libs` checkout, as for
`verification/` (see its README): the MATSim side reuses its runner and
cache. Sioux Falls comes from `matsim-libs/examples/scenarios/siouxfalls-2014`.
The full `perf` run takes about 40 minutes on an 8-core desktop, most of it
the grid-80 cases, plus a few minutes the first time to prepare the inputs. Do not run anything
else heavy at the same time.

## Layout

| File | Role |
|---|---|
| `corridor.py` | The showcase corridor scenario in two variants, `congested` (77 cars) and `free-flow` (20 cars), with the same network, buses and 19 passengers; written as MATSim XML to `scenarios/corridor/<variant>/` |
| `correctness.py` | Per variant: runs MATSim QSim, Hermes and DSim, then every MobSlim mobsim (on MATSim's clock) on MATSim's routed plans; parses all the events the same way; exports the animation data |
| `perf.py` | The speed and memory cases; prepares routed inputs, runs every variant in a fresh process under `/usr/bin/time` |
| `dsim.py`, `java/dsim/` | MATSim's DSim, which is only in weekly builds: `pom.xml` pins `2027.0-2026w45` (used only for its classpath) and `DSimRun.java` runs it in one JVM, in place of `RunCase` and `PerfRun` |
| `java/src/PerfRun.java` | MATSim with no replanning and no output, timing each iteration's mobsim (compiled with `javac` against the verification runner's classpath) |
| `networks.py` | One HermesSim run per performance case, drawn as a network snapshot: every car at the busiest minute, moving or queued |
| `weaving/experiment.py` | Weaving on the 80x80 grid at 0.3 capacity, medium congestion (inputs made on first use): 3 minutes of wall-clock time per run at 30% innovation under a budget of 8 processes; HermesSim with 1-8 strands, and `AsyncWeave` with 2-8 strands; one fresh process per run. `benchmarks/weave_profile.py` runs the same setup for a few iterations and prints where each iteration's time goes |
| `weaving/report.py` | Tables and SVG charts (utility against wall-clock time) from `results/weaving.json` |
| `perf_report.py` | Markdown tables and SVG charts (light and dark) from `results/perf.json` |
| `results/weaving.json` | The last weaving results (machine-specific; committed so the docs build) |
| `results/perf.json` | The last measured results, committed so the docs build without Java (machine-specific) |
