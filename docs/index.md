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

```{toctree}
:maxdepth: 2

simulation/index
developer
design
api
testing
```
