# The mobsims compared

A *mobsim* is the part of MobSlim that plays out a day: people leave home,
drive or walk, queue at busy junctions and arrive. MobSlim has five mobsims.
All of them use the same basic picture of traffic, which comes from MATSim:

* a road has a **free-flow time** (its length divided by its speed limit),
  the fastest anyone can drive it;
* a road has a **capacity**, how many cars per hour can leave it;
* a road has **room** for a limited number of cars. When it is full, cars
  behind it wait on the road before, so queues can spread backwards;
* cars leave a road in the order they entered it;
* a car that has been stuck for a while (`stuck_time`, 10 seconds by
  default) is pushed on anyway, so the network never locks up for good.

They differ in the small rules around this picture: how they count time,
who goes first when two queues want the same gap, and when a road counts
as full. Those rules are what this page is about, because they are what
changes the results. How fast each mobsim runs is covered in
[the design notes](design.md).

## At a glance

| MobSlim mobsim | Closest MATSim equivalent | Clock | Who goes first at a busy junction | Random? |
|---|---|---|---|---|
| QSim | MATSim's QSim (its standard mobsim) | exact times, to fractions of a second | first come, first served | no |
| AgentSim | none (MobSlim's own) | exact times | the first to try again (blocked cars check about once a second) | no |
| HermesSim | MATSim's Hermes | whole seconds | first come, first served | no |
| ParallelSim | none (MobSlim's own) | whole seconds | whoever has waited longest | no |
| ShardSim | matsim-rust (an experimental Rust port of MATSim) | whole seconds | a lottery weighted by road size | yes, a fixed seed |

"Random" means that changing the random seed changes who wins at busy
junctions, and so changes the results slightly. With the same seed, every
MobSlim mobsim gives the same results every time, and the parallel ones give
the same results however many processor cores they use.

## How they keep time

This is the difference you are most likely to see.

**QSim and AgentSim** use exact times. If a road takes 10.5 seconds, a car
spends 10.5 seconds on it, and a departure at 08:00:00.2 happens at
08:00:00.2.

**HermesSim and ParallelSim** tick once per second. Everything happens on a
whole second: a road's free-flow time is rounded to the nearest second,
and a planned departure is rounded up to the next second. Capacity is kept
exact over time: a road that lets one car through every 1.8 seconds lets
cars go at 2, 2, 2, 2, 1, ... second gaps, which average 1.8.

**ShardSim** also ticks once per second, but it follows MATSim's and
matsim-rust's way of handling a second. A car first reaches the end of the
road, and only in the *next* second crosses the junction onto the next road.
So each road costs about one second more. Road times are rounded *down* first.
Also, an activity starts one second after the person arrives.

The same trip on every mobsim: a car leaves at 100 s and drives three roads
of equal length, then starts an activity.

| | enters road 2 | enters road 3 | arrives | activity starts |
|---|---|---|---|---|
| **Roads of exactly 10 s** | | | | |
| QSim, AgentSim | 110 | 120 | 130 | 130 |
| HermesSim, ParallelSim | 110 | 120 | 130 | 130 |
| ShardSim | 111 | 122 | 132 | 133 |
| **Roads of 10.5 s** | | | | |
| QSim, AgentSim | 110.5 | 121 | 131.5 | 131.5 |
| HermesSim, ParallelSim | 111 | 122 | 133 | 133 |
| ShardSim | 111 | 122 | 132 | 133 |

What this means for results:

* With whole-second road times, QSim, AgentSim, HermesSim and ParallelSim
  give the same free-flow times. Otherwise the whole-second sims are up to
  half a second per road off, in either direction. Over a trip of many roads
  the errors mostly cancel.
* ShardSim's trips are slower by up to a second per road, about 5% on
  MobSlim's test grids. That is MATSim's own behaviour, and it makes
  ShardSim's road times the closest to real MATSim runs (see below).

### Choosing the clock

The table above shows each mobsim's own way of keeping time, but the clock
is a setting. Every mobsim takes three timing parameters, also available in
the configuration file under `mobsim:`:

* `continuous_time`: exact times (`true`) or whole seconds (`false`).
* `junction_delays`: whether a car waits one second at each junction before
  crossing into the next road, and a person one second after arriving before
  starting the activity. This is how MATSim's one-second steps work out.
* `rounding`: with whole seconds, whether road and walking times are
  rounded to the `nearest` second or `down`. Planned activity ends always
  round up, so nobody leaves early.

| | `continuous_time` | `junction_delays` | `rounding` |
|---|---|---|---|
| QSim | true (default) or false | false (default) or true | nearest (default) or down |
| AgentSim | true (default) or false | false (default) or true | nearest (default) or down |
| HermesSim | false only | false (default) or true | nearest (default) or down |
| ParallelSim | false only | false (default) or true | nearest (default) or down |
| ShardSim | false only | true (default) or false | down (default) or nearest |

Asking a mobsim for something it cannot do (exact times in a whole-second
mobsim, or rounding with exact times) stops with an error that says so.
Every setting works with public transport too (see below).

With the same settings, the mobsims keep time the same way. The trip above,
on roads of 10.5 s:

| `continuous_time`, `junction_delays`, `rounding` | enters road 2 | enters road 3 | arrives | activity starts |
|---|---|---|---|---|
| true, false (QSim, AgentSim) | 110.5 | 121 | 131.5 | 131.5 |
| true, true | 111.5 | 123 | 133.5 | 134.5 |
| false, false, nearest (HermesSim, ParallelSim) | 111 | 122 | 133 | 133 |
| false, true, nearest | 112 | 124 | 135 | 136 |
| false, false, down | 110 | 120 | 130 | 130 |
| false, true, down (ShardSim, as matsim-rust) | 111 | 122 | 132 | 133 |

The last road has no junction delay: the car arrives as it reaches its end.
Every mobsim gives exactly these times (`tests/test_qsim.py`).

Some pairs go further than one trip. With whole seconds, QSim gives exactly
HermesSim's results, car for car, however congested; AgentSim does too until
queues spill back onto the roads behind. The settings only make the
*clocks* agree: the rules of the previous section (who gets a gap, when a
road is full, how arriving cars leave) still differ.

## Public transport

Every mobsim runs public transport, with every clock setting, by the same
rules: buses (or trains) run to the timetable, either exactly (`transit:
vehicles_on: schedule`) or as vehicles in the traffic (`vehicles_on:
network`), where they queue with cars, wait at stops for boarding and
alighting, and leave no earlier than the timetable says. Passengers walk to
the stop, wait in line, board the first vehicle with room, and walk on at
the end.

The two can be mixed. `schedule_modes` lists the modes whose vehicles keep
exactly to the timetable while the others drive in the traffic, typically
trains on their own tracks next to buses in the streets:

```yaml
transit:
  vehicles_on: network
  schedule_modes: [rail]
```

This is MATSim's "deterministic" public transport (SBB's
`deterministicServiceModes`, which a MATSim config converts to). Every
mobsim supports it, and passengers change between the two kinds of line as
usual.

The clock settings apply to public transport as they do to cars:

* in whole seconds, times from the timetable round *up* (nothing leaves
  early), and walks, door times and boarding times round like road times
  (`rounding`);
* with junction delays, a bus serves a stop when it reaches the end of the
  road, and crosses into the next road one second after it leaves the
  stop. That is how MATSim does it.

A bus on a 100 s road with stops at its end, timetabled to leave each stop
2 minutes after the last (times from its departure):

| | arrives at stop 2 | leaves stop 2 | arrives at stop 3 |
|---|---|---|---|
| without junction delays | 100 | 120 | 220 |
| with junction delays | 101 | 120 | 221 |

The mobsims differ only in their usual rules (who goes first when queues
merge; ShardSim's queue and holding area), plus two things in the parallel
ones, which keep their results independent of the number of processor
cores: a bus starts at the end of its first road without waiting for room
there, and when a bus and a car want the same gap in the same second,
ParallelSim's tie-breaking rule decides, not the order they arrived in.

Against a real MATSim run of its "pt-tutorial" (900 people, a grid of roads
and a train line), with the vehicles in the traffic:

| | average error in time spent on a road | average error in a train's stop arrival | error in boardings | error in average pt trip time |
|---|---|---|---|---|
| QSim, AgentSim | 0.79 s | 2.0 s | 0% | −0.05% |
| HermesSim, ParallelSim | 0.79 s | 2.1 s | 0% | −0.06% |
| ShardSim (MATSim's clock, its default) | 0.002 s | 0.76 s | 0.3% | −0.01% |
| QSim, HermesSim or ParallelSim on MATSim's clock | 0.015 s | 0.76 s | 0.3% | −0.01% |

On the same clock, all of them give practically the same results: what is
left is MATSim's one-second steps, which MATSim's clock (`continuous_time:
false`, `junction_delays: true`, `rounding: down`) reproduces. With the
vehicles on the timetable (`vehicles_on: schedule`) every mobsim is equally
far from MATSim (13 s per stop arrival), because MATSim's trains do run in
the traffic.

## Differences that matter under congestion

When traffic is light, all five mobsims agree closely. When roads fill up,
these rules start to matter.

**Who gets a gap when roads merge.** When two queues wait for space on the
same road, something must decide who goes first.

* QSim and HermesSim: first come, first served. Cars already waiting get
  gaps in the order they started waiting, except that a car reaching the
  junction at the very moment a gap opens can take it first. In HermesSim,
  "the same moment" means the same second, which happens often in heavy
  traffic and keeps streams of cars flowing.
* AgentSim: blocked cars check again about once a second, and the first to
  check after a gap opens gets it. That is close to first come, first served.
* ParallelSim: whoever has waited longest gets the gap, judged once per
  second. A car that only just reached the junction never jumps ahead of one
  that was already waiting, which breaks up streams of cars in heavy
  congestion.
* ShardSim (like MATSim and matsim-rust): a lottery, where bigger roads get
  more tickets. A main road wins more often than a side street, but not
  always.

**When a road counts as full.** Every mobsim limits how many cars a road
holds, and a full road holds back the roads behind it. In all of them, a car
counts towards its road until it has crossed the junction at the end.
matsim-rust differs: a car that has reached the end and is only waiting to
cross the junction moves into a small holding area there (about one
second's worth of the road's capacity, usually one car) and no longer
counts, so each road holds about one car more. ShardSim can do the same
(`buffer_in_storage=False`), but by default it counts those cars like the
other mobsims. In practice this changes results little (see below).

**Arriving at the destination.** In QSim, AgentSim, HermesSim and ParallelSim,
a car on the last road of its trip still has to wait its turn to leave it,
like through traffic. In ShardSim (following matsim-rust), arriving cars
simply leave, so a busy last road does not slow them down.

**Being stuck.** All mobsims push a car on after it has waited `stuck_time`.
QSim and AgentSim do it at exactly that time; HermesSim and ParallelSim at
the next whole second. ShardSim does it only once the car has waited
*longer* than `stuck_time`, in whole seconds, so with 10 seconds it waits 11.

How much this changes results, measured on MobSlim's test grids (commuters
on a square street grid) at 5% of normal road capacity, so that queues
spread across much of the network. The numbers are total travel time
compared with QSim's.

| | QSim's average trip | AgentSim | HermesSim | ParallelSim | ShardSim | ShardSim, matsim-rust's holding area |
|---|---|---|---|---|---|---|
| **stuck time 10 s (the default)** | | | | | | |
| 20 × 20 grid | 30 min | 0.0% | −0.1% | −0.2% | −7.1% | −7.1% |
| 40 × 40 grid | 103 min | (not run) | 0.0% | +0.1% | −3.4% | −3.4% |
| **stuck time switched off** | | | | | | |
| 20 × 20 grid | 39 min | +5.6% | −0.4% | +14.8% | −24.5% | −26.2% |
| 40 × 40 grid | 190 min | (not run) | −0.8% | +32.8% | −39.9% | −42.0% |

* **With the usual stuck time, the mobsims agree closely**, even this
  congested. Pushing stuck cars on after 10 seconds stops jams from
  building up, and that matters more than any of the rules above.
  ShardSim's trips are 3–7% shorter, probably mostly because its arriving
  cars do not wait for the last road's capacity (its rules' effects have not
  been measured one by one). In lighter traffic its trips are a few percent *longer*
  instead (3% at 20% capacity, 5% in free flow), from the extra second
  per road.
* **With the stuck time switched off, the rules decide everything.** Jams
  only clear as fast as the merging rules allow, and no two rule sets
  agree: ParallelSim's "longest wait first" breaks up streams of cars and
  is much slower, AgentSim's "first to check again" a little slower, and
  ShardSim much faster. The holding area accounts for only about 2 points
  of ShardSim's difference.

MATSim's own results in such jammed regimes also depend on its random seed.

## Against real MATSim runs

MobSlim keeps a few runs of MATSim 2025 on its "equil" test network, and
compares each mobsim with them (`tests/test_matsim_reference.py`).

| | average error in time spent on a road | error in average trip time |
|---|---|---|
| **equil, 100 people** | | |
| QSim | 0.48 s | 0.01% |
| HermesSim, ParallelSim | 0.47 s | 0.03% |
| ShardSim (either holding-area rule) | 0.21 s | 0.05% |
| **equil, 2000 people, half capacity** | | |
| QSim | 1.25 s | 0.00% |
| HermesSim | 1.18 s | 0.01% |
| ParallelSim | 1.36 s | 0.01% |
| ShardSim | 1.03 s | 0.02% |
| ShardSim, matsim-rust's holding area | 0.48 s | 0.02% |

All of them match MATSim's trip times to within a twentieth of a percent.
ShardSim is closest road by road, because it copies MATSim's extra second at
each junction; with matsim-rust's holding area it is closer still on the
busier run.

## Each mobsim against its MATSim equivalent

### QSim and MATSim's QSim

QSim plays the same queue model as MATSim's standard mobsim. The differences
that change results:

* **Exact times instead of one-second ticks.** Cars move at the exact
  moment the model allows. MATSim also handles roads and junctions in
  separate steps of each second, so its cars spend an extra second at each
  junction. QSim's do not, by default; `continuous_time: false` and
  `junction_delays: true` give it MATSim's clock.
* **First come, first served at junctions**, where MATSim draws lots weighted
  by road size. QSim has no randomness at all.
* **Arriving cars wait their turn** to leave the last road of their trip,
  like through traffic.
* The default `stuck_time` in MobSlim's configuration files is 11 seconds,
  MATSim's 10 plus its one-second step, so the push-on happens at about the
  same moment.

### HermesSim and MATSim's Hermes

Both tick once per second and are built for speed. HermesSim keeps QSim's
rules where Hermes simplifies them:

* **Blocked cars wait in line** for a gap and get it in the order they
  arrived. Hermes has blocked cars try again every second.
* **The stuck timer runs per car.** In Hermes it runs per road.
* **Capacity is exact over time.** A road with spare capacity in a quiet
  second keeps it for the next car, so a road meant to pass a car every 1.8
  seconds does so on average. Hermes does not carry spare capacity over a
  quiet second.

The result is that, with whole-second road times, HermesSim gives exactly
the same trips as QSim.

### ShardSim and matsim-rust

ShardSim copies matsim-rust's design: the network is split into regions
with one worker each, and its traffic rules (the junction lottery, the
extra second per junction, rounding road times down, arriving cars leaving
freely, activities starting one second after arrival). The extra seconds
and the rounding are its default timing settings (see "Choosing the
clock"); the rest is fixed. Its code is MobSlim's own. The differences:

* **A car waiting to cross a junction still fills its road**, as in the
  other MobSlim mobsims, so a road never holds more than its stated room.
  matsim-rust lets it move into a holding area that does not count;
  `buffer_in_storage=False` does the same.

* **People are at junctions, not on roads.** In MATSim and matsim-rust an
  activity is on a road, and a trip starts by joining the end of that road.
  In MobSlim it is at a junction, and a trip starts by entering its first
  road, waiting there if the road is full. This is the same for every
  MobSlim mobsim (see [section 5 of the design notes](design.md)).
* **The lottery uses different random numbers**, so with congestion
  individual trips differ from matsim-rust's even with the same seed. On
  average they behave the same.
* **Results never depend on the number of workers.** In matsim-rust the
  order of events within a second can change with how the network is split.
  ShardSim always gives the same trips and events.
* Teleported trips (walks, bikes) take at least one second.
* No "kinematic waves" option, and no rerouting during the day.

### AgentSim and ParallelSim

These have no MATSim equivalent. **AgentSim** is QSim written from each
traveller's point of view (each person decides their next move), with
blocked cars checking again about once a second. Without congestion that
spills backwards it gives exactly QSim's trips. **ParallelSim** is HermesSim
rebuilt to run on many cores, with rules chosen so that its results never
depend on how many cores it uses. Without congestion it gives HermesSim's
trips (except on roads that take less than a second, which take a full
second). When jams are left to build up (no stuck time), its "longest
wait goes first" rule makes it differ, as shown above.
