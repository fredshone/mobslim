# Performance

How fast does utility rise, and where does it level off, when the same
budget of processes runs one simulation per iteration or several strands?
Both experiments run an 80×80 grid (26,240 links) with 103,680 car
commuters leaving home between 06:00 and 08:00, with flow and storage
capacity at 0.3 of the links' own: medium congestion. In the first
iteration, car trips take about 50% longer than at full capacity
(1,651 s against 1,104 s), and nothing gets stuck. Every run does 100
iterations from the same routed plans with 10% plan innovation per strand
(5% reroute, 5% time mutation plus reroute, 90% ChangeExpBeta), on:

```{include} ../_generated/weaving_machine.md
```

The scripts are `showcase/weaving/experiment.py` and
`showcase/weaving/report.py` (`python -m showcase weaving`, then
`python -m showcase weaving-report`).

## Summary

* **Per iteration, `w` strands learn about `w` times faster.** With `w`
  strands, `w`×10% of the population innovates each iteration. To reach the
  Controller's final utility, 2 strands need 50 iterations, 4 strands 27 and
  8 strands 16, against 97.
* **In wall-clock time, weaving pays up to the physical cores.** On 4
  physical cores, 4 strands reach the Controller's final utility 1.26× sooner,
  and are higher at every time after the first 15 s. With 8 strands, the
  extra strands share cores, and an iteration costs 5× more than the
  Controller's. That cancels the gain in wall time, but 8 strands climb
  furthest in 100 iterations.
* **More strands settle higher in 100 iterations.** The Controller and a
  single strand are still climbing at the end (129.2). 8 strands are
  levelling off near 137.
* **Strands use cores better than PSim threads at this size.** PSim with 8
  threads is no faster per iteration than single-threaded HermesSim
  (0.52 s against 0.47 s), so 4 threads × 2 strands is the best split of the
  8-process budget.

## Experiment 1: strands

HermesSim (single-threaded), with 1, 2, 4 and 8 strands, and the plain
`Controller` for reference. One strand is the Controller's loop run through
`Weave`, and runs at the same speed.

```{image} ../_static/showcase/weaving_weaving_time_light.svg
:class: only-light
:alt: Utility against wall-clock time for the Controller and 1, 2, 4 and 8 strands
```
```{image} ../_static/showcase/weaving_weaving_time_dark.svg
:class: only-dark
:alt: Utility against wall-clock time for the Controller and 1, 2, 4 and 8 strands
```

Against iterations, each doubling of the strands roughly halves the
iterations needed:

```{image} ../_static/showcase/weaving_weaving_iteration_light.svg
:class: only-light
:alt: Utility against iteration for the Controller and 1, 2, 4 and 8 strands
```
```{image} ../_static/showcase/weaving_weaving_iteration_dark.svg
:class: only-dark
:alt: Utility against iteration for the Controller and 1, 2, 4 and 8 strands
```

```{include} ../_generated/weaving_weaving.md
```

An iteration with `w` strands costs more than one simulation. Its
replanning reroutes `w` times as many people, in the main process. Its `w`
simulations run side by side, so each runs slower once they outnumber the
physical cores.

## Experiment 2: threads × strands

PSim, with the budget of 8 split between threads per simulation `p` and
strands `w`, with `p × w = 8`.

```{image} ../_static/showcase/weaving_budget_time_light.svg
:class: only-light
:alt: Utility against wall-clock time for PSim with p threads and w strands, p times w equal to 8
```
```{image} ../_static/showcase/weaving_budget_time_dark.svg
:class: only-dark
:alt: Utility against wall-clock time for PSim with p threads and w strands, p times w equal to 8
```

```{image} ../_static/showcase/weaving_budget_iteration_light.svg
:class: only-light
:alt: Utility against iteration for PSim with p threads and w strands, p times w equal to 8
```
```{image} ../_static/showcase/weaving_budget_iteration_dark.svg
:class: only-dark
:alt: Utility against iteration for PSim with p threads and w strands, p times w equal to 8
```

```{include} ../_generated/weaving_budget.md
```

On this machine, 8 is twice the physical cores, so every split
oversubscribes them. Two or four strands still beat one 8-thread
simulation. Eight single-threaded strands are the slowest in wall time, but
reach the highest utility in 100 iterations.
