# Performance

The following experiment considers the effectiveness of simulation parallelisation - aka **weaving**.

We use a scenario with 80×80 network grid (25,920 links) and 104,960 car commuters leaving home between 06:00 and 08:00, with flow and storage capacity at 0.3 resulting in some congestion.

We compare a regular **Controller** run against **Weaves** with 1, 2, 4 and 8 strands, and **AsyncWeaves** with 2, 4 and 8 strands, with an 8 physical cores compute budget. Every run gets the same 3 minutes of wall-clock time from the same initially routed plans, with 30% plan innovation per strand (15% reroute, 15% time mutation plus reroute, 70% ChangeExpBeta).

* **At this innovation rate, weaving does not beat the Controller within 3 minutes.**
  2 strands, and 2 or 8 asynchronous strands, end level with it (132.1 to 132.3
  against 132.15); 4 and 8 synchronous strands fall behind (130.5 and 130.4).
* **Not waiting for the slowest strand matters more with more strands.** 8
  asynchronous strands end at 132.21 against 130.44 for 8 synchronous strands, and
  avoid the early dip in utility.
* **Strands slow each other down.** An iteration takes 2.2 s for the Controller and
  11 s for 8 strands: simulation is limited by memory bandwidth, and at 30%
  innovation each strand also reroutes three times as many agents.

Every run is still improving at 3 minutes. Differences of about 0.3 are within
run-to-run noise: 1 strand runs the Controller's loop, and ends 0.24 lower.

## Experiment

Utility against wall-clock time for the synchronous `Weave`, and for `AsyncWeave`
(see [Don't wait for the slowest strand](examples.md#dont-wait-for-the-slowest-strand)),
which merges each strand as it finishes and starts it again straight away.
All runs use single-threaded HermesSim.

```{image} ../_static/showcase/weaving_weaving_time_light.svg
:class: only-light
:alt: Utility against wall-clock time for the Controller and 1, 2, 4 and 8 strands
```
```{image} ../_static/showcase/weaving_weaving_time_dark.svg
:class: only-dark
:alt: Utility against wall-clock time for the Controller and 1, 2, 4 and 8 strands
```

```{image} ../_static/showcase/weaving_async_time_light.svg
:class: only-light
:alt: Utility against wall-clock time for the Controller, 8 strands and 2, 4 and 8 asynchronous strands
```
```{image} ../_static/showcase/weaving_async_time_dark.svg
:class: only-dark
:alt: Utility against wall-clock time for the Controller, 8 strands and 2, 4 and 8 asynchronous strands
```

```{include} ../_generated/weaving_summary.md
```

`python -m benchmarks.weave_profile` prints where each iteration's time goes.

## Reproduce it

Every comparison here is at equal wall-clock time with the following hardware:

```{include} ../_generated/weaving_machine.md
```

The scripts are `showcase/weaving/experiment.py` and
`showcase/weaving/report.py` (`python -m showcase weaving`, then
`python -m showcase weaving-report`).
