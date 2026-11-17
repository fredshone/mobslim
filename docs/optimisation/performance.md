# Performance

The following experiment considers the effectiveness of simulation parallelisation - aka **weaving**.

We use the *grid-40-congested* scenario from the [simulation performance](../simulation/performance.md) page: a 40×40 grid with slightly irregular link lengths (6,560 links) and 26,240 car commuters leaving home between 06:00 and 08:00, with flow and storage capacity at 0.3, resulting in congestion.

We compare a regular **Controller** run against **Weaves** with 1, 2, 4 and 8 strands, and **AsyncWeaves** with 2, 4 and 8 strands, with an 8 physical cores compute budget, plus Java MATSim with its Hermes mobsim (on 8 threads) for reference. Every run gets the same minute of wall-clock time from the same initially routed plans, with 30% plan innovation per strand (15% reroute, 15% time mutation plus reroute, 70% ChangeExpBeta).

In the first minute:

* **Strands pull ahead within 10 seconds and stay ahead.** At 30 s, 4 and 8
  strands are at 138.25 against 138.02 for the Controller; at 60 s, 138.37
  against 138.26. 2 strands sit in between (138.13 and 138.33). 1 strand runs
  the Controller's loop and stays within 0.08 of it, a guide to run-to-run noise.
* **The Controller is quickest off the mark, because its iterations are
  cheapest.** At 5 s it is at 136.84 against 135.76 for 8 strands; by 10 s,
  8 strands have caught up (137.30 against 137.22). An iteration takes 0.24 s
  for the Controller and 0.88 s for 8 strands (255 and 68 iterations in the
  first minute).
* **Asynchronous strands keep busier but do not end higher.** In the first
  minute, 8 asynchronous strands complete 101 rounds against 68 iterations
  for 8 synchronous strands, and their processes spend 27% of the time in the
  mobsim against 19%. But at 60 s they are at 138.31 against 138.37. With 2
  and 4 strands, asynchronous runs are level with synchronous ones or 0.04
  behind.
* **MATSim Hermes is well behind after a minute: 137.11, 1.15 below the
  Controller,** after 44 iterations of about 1.4 s each. Its first iteration
  also scores lower on the same plans: 122.05 against 126.91 for every
  MobSlim run.

## Experiment

Utility against wall-clock time for the synchronous `Weave`, and for `AsyncWeave`
(see [Don't wait for the slowest strand](examples.md#dont-wait-for-the-slowest-strand)),
which merges each strand as it finishes and starts it again straight away.
All MobSlim runs use single-threaded HermesSim.

Above each chart is every run's cadence over its first 10 seconds: one bar
per mobsim run, with one lane per process running them at the same time. The gaps
are time spent on anything else: mostly each strand routing its innovators
(with 8 strands, about 0.45 s against 0.17 s of mobsim per iteration), then
the main process building jobs and merging, and, for the synchronous `Weave`,
waiting for the slowest strand. More strands simulate more plans at
once, but each iteration takes longer.

```{image} ../_static/showcase/weaving_weaving_time_light.svg
:class: only-light
:alt: Mobsim runs and utility against wall-clock time for the Controller, MATSim Hermes and 1, 2, 4 and 8 strands
```
```{image} ../_static/showcase/weaving_weaving_time_dark.svg
:class: only-dark
:alt: Mobsim runs and utility against wall-clock time for the Controller, MATSim Hermes and 1, 2, 4 and 8 strands
```

```{image} ../_static/showcase/weaving_async_time_light.svg
:class: only-light
:alt: Mobsim runs and utility against wall-clock time for the Controller, MATSim Hermes, 8 strands and 2, 4 and 8 asynchronous strands
```
```{image} ../_static/showcase/weaving_async_time_dark.svg
:class: only-dark
:alt: Mobsim runs and utility against wall-clock time for the Controller, MATSim Hermes, 8 strands and 2, 4 and 8 asynchronous strands
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
