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
  * [Correctness](simulation/correctness.md): Java MATSim and every MobSlim
    mobsim, animated side by side on the same scenario.
  * [Performance](simulation/performance.md): speed and memory against
    MATSim Hermes, QSim and DSim, from 26,000 to 84,000 agents.

* **[Optimisation](optimisation/index.md)**: the co-evolutionary loop, and
  weaving: several simulations per iteration, so more people innovate.
  * [Overview](optimisation/overview.md): the `Controller` and `Weave` loops.
  * [Examples](optimisation/examples.md): both, from Python and from a config.
  * [Performance](optimisation/performance.md): weaving on Sioux Falls.
  * [Details](optimisation/details.md): what to know when using weaving.

```{toctree}
:maxdepth: 2

simulation/index
optimisation/index
developer
design
api
testing
```
