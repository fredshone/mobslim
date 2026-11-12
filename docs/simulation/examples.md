# Examples

Running the mobsims from Python and from a config file. Each example uses
MATSim's equil scenario from `scenarios/equil/`, and each one runs as shown.

## Run a mobsim once

A mobsim takes a scenario with routed plans, plus an `EventsManager`, and
returns `Trajectories`: arrays of times, with no event objects.

```python
from mobslim import (EventsManager, HermesSim, HermesSimParams, Scenario,
                     read_network_xml, read_population_xml, route_scenario)

network = read_network_xml("scenarios/equil/network.xml")
population = read_population_xml("scenarios/equil/plans100.xml", network)
scenario = route_scenario(Scenario(network, population))  # mobsims need routed plans

sim = HermesSim(HermesSimParams())
traj = sim.run(scenario, EventsManager())  # no handlers: no events are built

trip_times = traj.leg_arrival - traj.leg_departure  # s per leg, NaN if stuck
print(f"mean trip {trip_times.mean():.0f} s, stuck {traj.stuck.sum()}")
```

## Swap mobsims

All five share one interface, so they can be swapped freely.

```python
from mobslim import AgentSim, DSim, HermesSim, PSim, QSim

for sim in [QSim(), AgentSim(), HermesSim(), PSim(), DSim()]:
    traj = sim.run(scenario, EventsManager())
    t = traj.leg_arrival - traj.leg_departure
    print(f"{type(sim).__name__:10} mean trip {t.mean():6.0f} s")
```

```
QSim       mean trip   1695 s
AgentSim   mean trip   1695 s
HermesSim  mean trip   1696 s
PSim       mean trip   1696 s
DSim       mean trip   1694 s
```

## Common parameters

Every `*Params` class takes these parameters:

```python
from mobslim import QSim, QSimParams

sim = QSim(QSimParams(
    flow_capacity_factor=0.1,     # a 10% sample: links pass 10% of their capacity
    storage_capacity_factor=0.1,  # ... and hold 10% of their cars
    stuck_time=10.0,              # s before a blocked car is pushed on (inf: never)
    end_time=30 * 3600,           # stop the day here (None: when everyone is done)
    effective_cell_size=7.5,      # m of road per car
))
```

## The clock

Three parameters set how a mobsim keeps time (see
[design detail](design-detail.md#choosing-the-clock)):

```python
# MATSim's own clock: whole seconds, a second at each junction, times rounded down
matsim_clock = dict(continuous_time=False, junction_delays=True, rounding="down")
sim = QSim(QSimParams(**matsim_clock))

# the whole-second mobsims cannot run in exact time
HermesSimParams(continuous_time=True)
# ValueError: HermesSim runs in whole seconds only (continuous_time=False); ...
```

## Threads, kernels and seeds

```python
from mobslim import DSimParams, PSimParams

PSim(PSimParams(threads=8, partitioner="morton"))  # results do not depend on threads
DSim(DSimParams(threads=8,
                partitioner="metis",               # needs pymetis
                seed=4711,                         # the junction lottery
                buffer_in_storage=False))          # as matsim-rust
HermesSim(HermesSimParams(backend="python"))       # the Python reference kernel
                                                   # ("auto": Rust if installed)
```

## Events

Events are built only for the handlers that ask for them. `EventLog` keeps
them all, and `EventsXMLWriter` writes MATSim's format.

```python
from mobslim import EventLog
from mobslim.events import EventHandler, EventsXMLWriter, LinkEnter

class LinkVolumes(EventHandler):
    handles = (LinkEnter,)  # only these events are built and passed in

    def __init__(self):
        self.volumes = {}

    def handle(self, event):
        self.volumes[event.link] = self.volumes.get(event.link, 0) + 1

counter, log = LinkVolumes(), EventLog()
writer = EventsXMLWriter("events.xml.gz", network, population)
QSim().run(scenario, EventsManager([counter, log, writer]))
log.to_dataframe()  # one row per event
```

## Trajectories

```python
from mobslim.routing.traveltime import TimeBins

traj = HermesSim().run(scenario, EventsManager())

traj.link_enter                    # entry time per route link, in plan order
traj.link_leave()                  # leave time per route link
traj.leg_entered                   # links entered per leg
traj.stuck                         # per person
traj.travel_time_table(network, bin_size=900)  # mean link times per 15 min (uint16 s)
traj.travel_time_table(network, TimeBins.off_peak())  # 15 min 06-22 h, hourly otherwise
traj.experienced_plans()[0].legs[0]
# ExperiencedLeg(mode='car', departure_time=21600.0, arrival_time=22500.0,
#                distance=25000.0, links=(1, 10, 19), rides=None)

# what the controllers ask for outside their output iterations: the link
# travel-time sums built while simulating, and no per-link entry times
lean = HermesSim().run(scenario, EventsManager(), traversal_bins=(900, 30 * 3600),
                       link_times=False)
lean.link_enter                    # None (leg_entered, leg_distance etc. as above)
lean.traversal_aggregate(network)  # the sums the mobsim built
```

Link entry times are the largest output (one value per route link), and
nothing an iteration needs reads them, so the controllers record them in
the last iteration only, unless `ControllerParams(link_times_interval=n)`
asks for every `n`-th as well. See [Design: trajectories](../design.md).

## In the controller

The controller routes plans, runs the mobsim each iteration, then scores
and replans.

```python
from mobslim import Controller, ControllerParams

result = Controller(scenario, mobsim=PSim(PSimParams(threads=4)),
                    params=ControllerParams(iterations=3)).run()
result.metrics_frame()[["score_executed", "car_travel_time"]]
```

## In a config file

The `mobsim:` section of a mobslim config. Run
`mobslim reference` to list every parameter, and `mobslim resolved
config.yaml` to see the defaults filled in.

```yaml
mobsim:
  type: psim                 # qsim, agentsim, hermes, psim or dsim
  flow_capacity_factor: 0.1
  storage_capacity_factor: 0.1
  stuck_time: 11             # MATSim's 10 s plus its one-second step (the default)
  end_time: "30:00:00"
  continuous_time: false     # left out: the mobsim's own clock
  junction_delays: true
  rounding: down
  psim_threads: 8            # per-mobsim options carry its name:
                             # hermes_backend, agentsim_blocked, psim_backend,
                             # psim_partitioner, dsim_threads, dsim_partitioner,
                             # dsim_buffer_in_storage
```

Public transport vehicles drive in the traffic, or keep exactly to the
timetable:

```yaml
transit:
  schedule: transitSchedule.xml
  vehicles: transitVehicles.xml
  vehicles_on: network       # or: schedule
  schedule_modes: [rail]     # these keep to the timetable even so
```

```python
from mobslim import Controller

Controller.from_mobslim_config("config.yaml").run()
Controller.from_matsim_config("config.xml").run()  # a MATSim config, read as is
```
