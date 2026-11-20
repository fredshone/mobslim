# Configuration

Scoring is set in one of three places: a MATSim config, a MobSlim config,
or Python. All three give a `CharyparNagelScorer` with the same
parameters, and `python -m mobslim.config reference` lists every one.

## MATSim style

A MATSim config's `scoring` module (`planCalcScore` in older configs) is
read as MATSim reads it. MATSim's defaults fill whatever the config leaves
out: its six default modes, `waitingPt`, `relative` zero-utility durations,
and the stuck penalty.

```xml
<module name="scoring">
  <parameterset type="scoringParameters">
    <param name="performing" value="+6" />
    <param name="lateArrival" value="-18" />
    <parameterset type="modeParams">
      <param name="mode" value="car" />
      <param name="marginalUtilityOfTraveling_util_hr" value="-6.0" />
    </parameterset>
    <parameterset type="activityParams">
      <param name="activityType" value="w" />
      <param name="typicalDuration" value="08:00:00" />
      <param name="minimalDuration" value="06:00:00" />
      <param name="openingTime" value="07:00:00" />
      <param name="latestStartTime" value="09:00:00" />
      <param name="closingTime" value="18:00:00" />
    </parameterset>
  </parameterset>
</module>
```

```python
from mobslim import Controller
controller = Controller.from_matsim_config("config.xml")
```

A `scoringParameters` set with a `subpopulation` scores the persons whose
`subpopulation` attribute names it, as in MATSim. Parameters MobSlim cannot
honour are logged, or raise an error with `strict=True`. Money events
(tolls and fares) are not modelled.

## MobSlim style

The same parameters in a MobSlim config (YAML), with snake-case names and
MATSim's defaults. `mobslim from-matsim config.xml -o config.yaml`
translates a MATSim config into one.

```yaml
scoring:
  performing: 6.0
  modes:
    car: {marginal_utility_of_traveling: -6.0}
  activities:
    h: {typical_duration: "12:00:00", minimal_duration: "08:00:00"}
    w:
      typical_duration: "08:00:00"
      minimal_duration: "06:00:00"
      opening_time: "07:00:00"
      closing_time: "18:00:00"
      latest_start_time: "09:00:00"
```

Almost any scoring value can also be set **per person**. It takes a default
and rules on the person's attributes, and the last rule that matches wins.
Rules can do what subpopulations do in MATSim, and more:

```yaml
scoring:
  modes:
    walk:
      marginal_utility_of_traveling:
        default: -6
        rules:
          - {when: age > 50, set: -9}
          - {when: age > 70, set: -15}
  activities:
    w:
      typical_duration:
        default: "08:00:00"
        rules:
          - {when: employment == "part_time", set: "04:00:00"}
```

`mobslim check config.yaml` reports unknown activity types, modes and
attributes before a run. `python -m mobslim.config resolved config.yaml`
prints the config with every default filled in.

## Python

`ScoringParams` holds the same values as plain dataclasses (the example on
the [Scoring](index.md) page builds one). For something else entirely, any
object with a `score(plan) -> float` method is a `Scorer`:

```python
from mobslim import Controller

class TimeAtHome:
    """Utility = hours at home."""
    def score(self, plan):
        return sum((a.end or 24 * 3600) - (a.start or 0) for a in plan.activities
                   if a.type == "h") / 3600

controller = Controller.from_mobslim_config("config.yaml", scorer=TimeAtHome())
```

On equil, the commuters then score about 22.4: their hours at home.

A scorer with a `score_trajectories(trajectories)` method scores every
person at once from the mobsim's arrays, as `CharyparNagelScorer` does.
Without one, the controller builds an experienced plan per person and calls
`score` on each.
