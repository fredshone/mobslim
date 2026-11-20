# MobSlim

A modular, MATSim-inspired, agent-based transport simulation in Python.
Installation, a quickstart, the `mobslim` command and the configuration
files are described in the
[README](https://github.com/fredshone/mobslim#readme).

* **[Simulation](simulation/index.md)**: the mobsims.
  * [How the mobsims work](simulation/overview.md): five engines behind one
    interface, and why they are fast.
  * [Examples](simulation/examples.md): the mobsims from Python and from a config.
  * [Design detail](simulation/design-detail.md): how the engines differ.
  * [Comparison to MATSim](simulation/matsim.md): Java MATSim and MobSlim,
    animated side by side on the same scenario.
  * [Correctness](simulation/correctness/index.md): observed traffic at seven
    roundabouts and on three highway sections, replayed through HermesSim.
  * [Performance](simulation/performance.md): speed and memory against
    MATSim Hermes, QSim and DSim, from 26,000 to 84,000 agents.

* **[Optimisation](optimisation/index.md)**: the co-evolutionary loop, and
  weaving: several simulations per iteration, so more people innovate.
  * [Overview](optimisation/overview.md): the `Controller` and `Weave` loops.
  * [Examples](optimisation/examples.md): both, from Python and from a config.
  * [Performance](optimisation/performance.md): weaving on Open Berlin at 1%, in wall-clock time.
  * [Details](optimisation/details.md): what to know when using weaving.

* **[Scoring](scoring/index.md)**: what an agent's day is worth.
  * [Correctness](scoring/correctness.md): MobSlim's scores against MATSim's,
    on the same days.
  * [Configuration](scoring/configuration.md): in a MATSim config, a MobSlim
    config, or Python.

```{toctree}
:maxdepth: 2

simulation/index
optimisation/index
scoring/index
developer
design
api
testing
```
