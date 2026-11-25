# Performance

The following experiment considers the effectiveness of simulation parallelisation - aka **weaving**.

We use the *open-berlin-1pct* scenario from the [simulation performance](../simulation/performance.md#open-berlin-1pct) page: [MATSim Open Berlin v6.4](https://github.com/matsim-scenarios/matsim-berlin) at its 1% sample, 52,263 agents (residents, commercial traffic and freight) on 283,885 links, with cars, trucks and freight on the roads at flow and storage capacity ×0.01, and pt on Berlin's timetable. Every run starts from Open Berlin's initial plans: its input plans before calibration, not yet routed.

We compare a regular **Controller** run against **Weaves** with 1, 2, 4 and 8 strands, and **AsyncWeaves** with 2, 4 and 8 strands, with an 8 physical cores compute budget, plus Java MATSim with its Hermes mobsim (on 8 threads) for reference. Every run gets the same 20 minutes of wall-clock time from the same plans, with 30% plan innovation per strand (15% reroute, 15% time mutation plus reroute, 70% ChangeExpBeta), for every subpopulation. Open Berlin's own run sets its strategies in Java code. Here, both sides use these three instead. Both route by MATSim's time and distance cost, and mobslim bounds its A\* as MATSim does (`bound_factor=1`, off by default). This gives MATSim's kind of not-quite-least-cost routes, so the two meet the same congestion (see [the mobsim's notes](../simulation/design-detail.md)).

In 20 minutes (utility as a 5-iteration rolling mean):

* **Few strands help a little; many do not.** From a starting 75.66, the
  Controller reaches 103.14. 1 strand runs the Controller's loop and reaches
  103.52, a guide to run-to-run noise (about 0.4). 2 strands reach 103.51,
  level with 1 strand. They lead early (101.75 at 300 s, against 101.40 for
  the Controller), but by less than the noise.
* **Strands do more work, but each simulation is worth less.** 2, 4 and 8
  strands run 14–15 simulations a minute, against 11 for the Controller, and
  innovate 25–35% more agents. Per round they also learn faster: at round
  30, 8 strands are at 102.07 against 100.76. But each round's strands all
  innovate on the same travel times and plans, and those are updated only
  once a round, so much of the extra work repeats itself. Per simulation the
  strands learn more slowly: after 140 simulations, 8 strands are at 101.49
  against 102.78 for the Controller. The two effects about cancel at 2
  strands, and 4 and 8 strands end behind (102.70 and 102.14). An iteration
  takes 5.3 s for the Controller and 34 s with 8 strands, whose processes
  spend 72–76% of it innovating (mostly car and pt routing) and 16–17% in
  the mobsim.
* **Asynchronous strands look best, but are not comparable.** 4 and 8
  asynchronous strands end highest (104.78 and 106.85), and 8 asynchronous
  strands' cars take 42 minutes a trip, against 52–54 minutes in the
  synchronous runs (49 with 4 asynchronous strands). Both come from how `AsyncWeave` picks innovators: each slot draws
  them only from agents not already in flight in another slot. With 8 slots at
  30%, the first three take 90% of the agents, the fourth the last 10%, and
  the other four run with no innovators at all. Half its simulations are
  agents repeating their selected plans, which score high and jam less, and
  utility is the mean of everything simulated. 2 asynchronous strands, which
  do not hit this cap, reach 103.09, level with the Controller. With 8
  asynchronous strands the workers are busy 57% of the time, and merging
  takes 29% of the main process's time.
* **MATSim Hermes starts higher but improves more slowly.** Its first
  iteration scores 81.70 against 75.66 for MobSlim, mostly because Hermes
  gets fewer agents stuck at 1% (see [the mobsim's
  notes](../simulation/design-detail.md)). It runs 35 iterations of about
  35 s each and reaches 102.52, below the Controller and every MobSlim run
  except 8 synchronous strands.

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
`AsyncWeave` and 8 strands, the workers wait for the main process's merges.

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
