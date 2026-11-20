# Performance

The following experiment considers the effectiveness of simulation parallelisation - aka **weaving**.

We use the *open-berlin-1pct* scenario from the [simulation performance](../simulation/performance.md#open-berlin-1pct) page: [MATSim Open Berlin v6.4](https://github.com/matsim-scenarios/matsim-berlin) at its 1% sample, 52,263 agents (residents, commercial traffic and freight) on 283,885 links, with cars, trucks and freight on the roads at flow and storage capacity ×0.01, and pt on Berlin's timetable. Every run starts from the agents' calibrated selected plans.

We compare a regular **Controller** run against **Weaves** with 1, 2, 4 and 8 strands, and **AsyncWeaves** with 2, 4 and 8 strands, with an 8 physical cores compute budget, plus Java MATSim with its Hermes mobsim (on 8 threads) for reference. Every run gets the same 20 minutes of wall-clock time from the same plans, with 30% plan innovation per strand (15% reroute, 15% time mutation plus reroute, 70% ChangeExpBeta), for every subpopulation. Open Berlin's own run sets its strategies in Java code. Here, both sides use these three instead. Both route by MATSim's time and distance cost, and mobslim bounds its A\* as MATSim does (`bound_factor=1`, off by default). This gives MATSim's kind of not-quite-least-cost routes, so the two meet the same congestion (see [the mobsim's notes](../simulation/design-detail.md)).

In the first 10 minutes:

* **Two strands end ahead of the Controller, but not by much.** At 600 s,
  2 strands are at 113.29 against 112.97 for the Controller, from a
  starting 110.74. That is 14% more improvement. 1 strand runs the
  Controller's loop and ends at 113.09, 0.12 above it, a guide to
  run-to-run noise. 4 strands end level with 1 strand (113.10).
* **8 synchronous strands do worst** (112.50). Each strand routes its own
  innovators (30% of the agents) on its share of the CPUs, and routing fills
  the time. An iteration takes 5.1 s for the Controller, of which about 60%
  is innovation (mostly car and pt routing) and about 27% is the mobsim. With
  8 strands an iteration takes 40 s, and every process spends 75–82% of it
  innovating and 12–13% in the mobsim. The Controller runs 118 iterations in
  the 10 minutes, and 8 strands run 15.
* **Asynchronous strands are erratic.** With 4 and 8 asynchronous strands,
  utility jumps by about 1.3 in the first 2 minutes and ends highest (113.94
  and 113.84). In a run made before the [scoring fixes](../scoring/correctness.md#open-berlin),
  the same two runs dipped below the start in their first rounds and ended
  lowest. One run each does not settle which it is. With 8 asynchronous
  strands, the workers are busy 58% of the time, and the main process's
  merging and planning take 22% of its time.
* **MATSim Hermes starts close and does not improve.** On the same plans its
  first iteration scores 111.73 against 110.74 for MobSlim (the gap is the
  simulation, see [Scoring correctness](../scoring/correctness.md)). It then
  falls to 110.45 after 17 iterations of about 35 s each.

## Experiment

Utility against wall-clock time for the synchronous `Weave`, and for `AsyncWeave`
(see [Don't wait for the slowest strand](examples.md#dont-wait-for-the-slowest-strand)),
which merges each strand as it finishes and starts it again straight away.
All MobSlim runs use single-threaded HermesSim.

Above each chart is what each run's processes did in their first 2 minutes,
one lane per process, the main process on top: mobsim runs (solid), innovation,
which is mostly routing (light), and the main process's planning and merging,
and scoring (grey). The gaps are waiting: for the slowest strand in the
synchronous `Weave`, or for the main process. Routing, not the mobsim, takes
most of a strand's time, and every strand routes its own innovators, so more
strands simulate more plans at once but each iteration takes longer. With
`AsyncWeave` and 8 strands, the workers wait between rounds.

```{image} ../_static/showcase/weaving_weaving_time_light.svg
:class: only-light
:alt: What each process did, and utility against wall-clock time for the Controller, MATSim Hermes and 1, 2, 4 and 8 strands
```
```{image} ../_static/showcase/weaving_weaving_time_dark.svg
:class: only-dark
:alt: What each process did, and utility against wall-clock time for the Controller, MATSim Hermes and 1, 2, 4 and 8 strands
```

```{image} ../_static/showcase/weaving_async_time_light.svg
:class: only-light
:alt: What each process did, and utility against wall-clock time for the Controller, MATSim Hermes, 8 strands and 2, 4 and 8 asynchronous strands
```
```{image} ../_static/showcase/weaving_async_time_dark.svg
:class: only-dark
:alt: What each process did, and utility against wall-clock time for the Controller, MATSim Hermes, 8 strands and 2, 4 and 8 asynchronous strands
```

```{include} ../_generated/weaving_summary.md
```

`python -m benchmarks.weave_profile` prints where each iteration's time goes.

## Reproduce it

Every comparison here is at equal wall-clock time with the following hardware:

```{include} ../_generated/weaving_machine.md
```

The scripts are `showcase/weaving/experiment.py`,
`showcase/java/src/ReplanRun.java` (MATSim) and
`showcase/weaving/report.py` (`python -m showcase weaving`, then
`python -m showcase weaving-report`).
