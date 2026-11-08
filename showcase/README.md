# Showcase

Generates the evidence behind three documentation pages:

| Page | Command | Writes |
|---|---|---|
| [Correctness](../docs/simulation/correctness.md) | `uv run python -m showcase correctness` | `docs/_static/showcase/corridor-data.js` (and prints a summary table per variant) |
| [Performance](../docs/simulation/performance.md) | `uv run python -m showcase perf` (then `report`) | `showcase/results/perf.json`, `docs/_generated/perf_*.md`, `docs/_static/showcase/perf_*.svg`; `networks`: `docs/_static/showcase/net_*.png` |
| [How the mobsims work](../docs/simulation/overview.md) | hand-written | |

`python -m showcase perf -k grid` runs only the matching cases;
`python -m showcase report` redraws the tables and charts from the saved
results without rerunning anything. `python -m showcase networks` redraws the network snapshots
(a few minutes; it simulates each case once). `python -m showcase all` does everything.

**Requirements.** Java 21+, Maven and a `matsim-libs` checkout, as for
`verification/` (see its README): the MATSim side reuses its runner and
cache. Sioux Falls comes from `matsim-libs/examples/scenarios/siouxfalls-2014`.
The full `perf` run takes about an hour on a laptop. Do not run anything
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
| `perf_report.py` | Markdown tables and SVG charts (light and dark) from `results/perf.json` |
| `results/perf.json` | The last measured results, committed so the docs build without Java (machine-specific) |
