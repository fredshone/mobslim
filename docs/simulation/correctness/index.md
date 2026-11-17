# Correctness

How close does MobSlim come to real traffic? We compare the Mobslim HermesSim to real vehicle movements data extracted from overhead film.

For each scenario we build a network and simulate each vehicle using their observed route through the infrastructure and observed first link entry time.

## Scenarios

* **[Roundabouts](roundabouts.md)**: seven German roundabouts, filmed by drone.
* **[Highways](highways.md)**: three US highway sections, filmed from overpasses and a helicopter.

## Link speeds and flow capacities

Every link takes its free speed and capacity from its type of road as follows:

| Road | Free speed | Capacity per lane |
|:--|--:|--:|
| Roundabouts | 25 km/h | 1,800 veh/h |
| Highways | 55 mph | 1,800 veh/h |

## Results summary

For each vehicle seen coming into view and leaving it, we compare its time
in view. The "typical error" is the median gap between a vehicle's
simulated and observed time in view.

```{include} ../../_generated/correctness_headline.md
```

```{toctree}
:hidden:

roundabouts
highways
```
