# Python implementation of MATSim

## Plan

Python interfaces for experimenting with MATSim flavoured simulations.

- Parallel sims
- Parallelised sims
- ML planning

## WIP

- API for (i) planners and (ii) simulations.
- Data design for (i) simulation instructions, and (ii) simulation events.
- Tooling for (i) initialisation e.g. loading scenarios, and (ii) reporting e.g. progress and utilization.
- Planner is responsible for making instructions:
    - Initial case
    - Based on prior events from one or many sims
- Sim is responsible for resolving instructions:
    - Stay - duration for an activity
    - Move - move from entity to entity
- Stay for fixed duration:
    - Facilities for activity
    - (depending on capacity e.g. car park or building or charging)
- Move from A to B:
    - Facility to Link
    - Link to Link
    - (Stop to Stop)
- Loading:
    - Plans
    - Networks
    - (Facilities)
    - (Transit)
    - (Vehicles)
- Reporting:
    - trip durations
    - trip distances
    - link speeds
    - (utility)

## Installation

Currently using [uv](https://docs.astral.sh/uv/).

So to run a demo notebook:

```
git clone git@github.com:fredshone/mobslim.git
cd mobslim

# then take a look at the demos

# if you need, spin up a notebook as follows:
uv run --with jupyter jupyter lab

```

