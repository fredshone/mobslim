# Highways

## I-395, a merge and a lane drop

Two hours of the morning peak in Washington, DC. An on-ramp joins as a
fourth lane that ends about 200 m later. That lane adds no capacity, so the
queue forms behind the lane drop, as it does on the real road.

```{raw} html
<div class="mobsim-anim replay-anim" data-key="tgsim-i395"></div>
<script src="../../_static/showcase/replay-tgsim-i395-sim.js"></script>
<script src="../../_static/showcase/replay-tgsim-i395-obs.js" onerror="void 0"></script>
```

## I-90/I-94, a diverge

The evening peak in Chicago. The three left lanes are queued by congestion
beyond the camera, while the three right lanes flow. The left lanes drain
at the rate observed leaving them, 1,000 veh/h per lane.

```{raw} html
<div class="mobsim-anim replay-anim" data-key="tgsim-i90-diverge"></div>
<script src="../../_static/showcase/replay-tgsim-i90-diverge-sim.js"></script>
<script src="../../_static/showcase/replay-tgsim-i90-diverge-obs.js" onerror="void 0"></script>
```

## I-90/I-94, a weave

Further along the same road, with the same rules.

```{raw} html
<div class="mobsim-anim replay-anim" data-key="tgsim-i90-weave"></div>
<script src="../../_static/showcase/replay-tgsim-i90-weave-sim.js"></script>
<script src="../../_static/showcase/replay-tgsim-i90-weave-obs.js" onerror="void 0"></script>
<script src="../../_static/showcase/replay-anim.js"></script>
```

## Notes

* **Queued and flowing lanes come out right together.** The diverge's
  right lanes are within 2 s of the recording, and the queued left lanes
  at both sites are within 5 s on average.
* **Single queued vehicles miss by more than the average.** A real queue
  crawls along the whole section, but the simulated one stands still at
  jam spacing in the last links. So a single queued vehicle can be 15–20 s
  off.
* **Weaving is not modelled.** At the weave, traffic crossing between lane
  groups slows the right lanes, and the simulation runs them a third too
  fast.
* **Data.** [TGSIM I-395](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-395-trajectories)
  and [TGSIM I-90/I-94 Stationary](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-90-i-94-stationary-trajectories)
  come from the Federal Highway Administration and the U.S. DOT ITS Joint
  Program Office. Their pages name no licence, so this build publishes only
  the simulated vehicles; the recorded tracks are built locally.

## Reproduce

Save the trajectory CSVs as `data/tgsim/I395_trajectories.csv` and
`data/tgsim_i90/I90_94_stationary_trajectories.csv`. Then run
`uv run python -m showcase tgsim` and `uv run python -m showcase tgsim-i90`.
The code is in `showcase/tgsim/` and `showcase/tgsim_i90/`.
