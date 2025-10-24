# Python implementation of MATSim

## WIP

- API for (i) planners and (ii) simulations.
- Data design for (i) simulation instructions, and (ii) simulaion events.
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

## Simulation

- Stay Instruction -> Wait(t)
- Move Instruction -> Check capacity

