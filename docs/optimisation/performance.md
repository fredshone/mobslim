# Performance

The following experiment considers the effectiveness of simulation parallelisation - aka **weaving**.

We use the *grid-40-congested* scenario from the [simulation performance](../simulation/performance.md) page: a 40×40 grid with slightly irregular link lengths (6,560 links) and 26,240 car commuters leaving home between 06:00 and 08:00, with flow and storage capacity at 0.3, resulting in congestion.

We compare a regular **Controller** run against **Weaves** with 1, 2, 4 and 8 strands, and **AsyncWeaves** with 2, 4 and 8 strands, with an 8 physical cores compute budget, plus Java MATSim with its Hermes mobsim (on 8 threads) for reference. Every run gets the same minute of wall-clock time from the same initially routed plans, with 30% plan innovation per strand (15% reroute, 15% time mutation plus reroute, 70% ChangeExpBeta).

In the first minute:

* **Strands pull ahead within 10 seconds and stay ahead.** At 30 s, 4 and 8
  strands are at 138.28 against 138.08 for the Controller; at 60 s, 138.37
  and 138.38 against 138.28. 2 strands sit in between (138.19 and 138.34).
  1 strand runs the Controller's loop and stays within 0.08 of it, a guide to
  run-to-run noise.
* **The Controller is quickest off the mark, because its iterations are
  cheapest.** At 5 s it is at 136.90 against 135.76 for 8 strands; by 10 s,
  8 strands have caught up (137.47 against 137.32). An iteration takes 0.21 s
  for the Controller and 0.80 s for 8 strands (287 and 75 iterations in the
  first minute).
* **Routing, not the mobsim, fills most of a strand's time.** With 8
  strands, each process spends 42% of the first minute innovating (mostly
  routing) and 22% in the mobsim.
* **Asynchronous strands keep busier but do not end higher.** In the first
  minute, 8 asynchronous strands complete 112 rounds against 75 iterations
  for 8 synchronous strands, and their processes spend 34% of the time in the
  mobsim against 22%. But at 60 s they are at 138.32 against 138.38, and
  their main process, merging and planning one strand at a time, is busy
  most of the time. With 2 and 4 strands, asynchronous runs are level with
  synchronous ones or 0.03 behind.
* **MATSim Hermes is well behind after a minute: 137.11, 1.17 below the
  Controller,** after 44 iterations of about 1.4 s each. Its first iteration
  also scores lower on the same plans: 122.05 against 126.91 for every
  MobSlim run.

## Experiment

Utility against wall-clock time for the synchronous `Weave`, and for `AsyncWeave`
(see [Don't wait for the slowest strand](examples.md#dont-wait-for-the-slowest-strand)),
which merges each strand as it finishes and starts it again straight away.
All MobSlim runs use single-threaded HermesSim.

Above each chart is what each run's processes did in their first 10 seconds,
one lane per process, the main process on top: mobsim runs (solid), innovation,
which is mostly routing (light), and the main process's planning and merging,
and scoring (grey). The gaps are waiting: for the slowest strand in the
synchronous `Weave`, or for the main process. Routing, not the mobsim, takes
most of a strand's time, and every strand routes its own innovators, so more
strands simulate more plans at once but each iteration takes longer. With
`AsyncWeave` and many strands, the main process, which merges and plans one
strand at a time, becomes the limit.

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
