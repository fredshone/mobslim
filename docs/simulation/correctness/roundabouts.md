# Roundabouts

## rdb1

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb1"></div>
<script src="../../_static/showcase/replay-opendd-rdb1-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb1-obs.js" onerror="void 0"></script>
```

## rdb2

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb2"></div>
<script src="../../_static/showcase/replay-opendd-rdb2-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb2-obs.js" onerror="void 0"></script>
```

## rdb3

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb3"></div>
<script src="../../_static/showcase/replay-opendd-rdb3-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb3-obs.js" onerror="void 0"></script>
```

## rdb4

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb4"></div>
<script src="../../_static/showcase/replay-opendd-rdb4-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb4-obs.js" onerror="void 0"></script>
```

## rdb5

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb5"></div>
<script src="../../_static/showcase/replay-opendd-rdb5-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb5-obs.js" onerror="void 0"></script>
```

## rdb6

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb6"></div>
<script src="../../_static/showcase/replay-opendd-rdb6-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb6-obs.js" onerror="void 0"></script>
```

## rdb7

```{raw} html
<div class="mobsim-anim replay-anim" data-key="opendd-rdb7"></div>
<script src="../../_static/showcase/replay-opendd-rdb7-sim.js"></script>
<script src="../../_static/showcase/replay-opendd-rdb7-obs.js" onerror="void 0"></script>
<script src="../../_static/showcase/replay-anim.js"></script>
```

## Notes

* **Not modelled.** A queue model has no giving way. Entering vehicles
  merge with circulating ones by HermesSim's junction rule, and no vehicle
  slows down unless a queue holds it up. The 25 km/h stands in for the
  average slowing.
* **Data.** [openDD](http://www.l3pilot.eu/openDD) (Breuer et al., 2020)
  is licensed CC BY-ND 4.0. This build publishes only the simulated
  vehicles; the recorded tracks in the animations are built locally from the
  download.

## Reproduce

Unzip `opendd_v3-rdb1.zip` to `opendd_v3-rdb7.zip` into `data/opendd/`,
then run `uv run python -m showcase opendd`. The code is in
`showcase/opendd/`.
