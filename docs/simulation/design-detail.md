# Design detail

## At a glance

| Mobsim | Language | Closest MATSim equivalent | Clock | Busy junction: who goes first | Random |
|---|---|---|---|---|---|
| QSim | Python | QSim | exact | first come, first served | no |
| AgentSim | Python | none | exact | first to retry (about once a second) | no |
| HermesSim | Rust (Python reference) | Hermes | whole seconds | first come, first served | no |
| PSim | Rust (Python reference), threads | none | whole seconds | longest wait | no |
| DSim | Rust, threads | matsim-rust | whole seconds, MATSim's | lottery weighted by capacity | fixed seed |

With the same inputs (and seed), every mobsim gives the same results every
run, whatever the number of threads.

## Each mobsim

### QSim

* MATSim's queue model, event-driven, in exact (fractional-second) time.
* Junctions: first come, first served. No randomness.
* Arriving cars wait their turn to leave their last link, like through
  traffic.
* Differs from MATSim's QSim: exact times, no extra second per junction (by
  default), no junction lottery.

### AgentSim

* The QSim written from each agent's view: each person decides their next
  move.
* Blocked cars retry every `retry_interval` (1 s). `blocked="chain"` skips
  retries that cannot succeed, with the same results.
* Gives exactly the QSim's trips until queues spill back.
* Under spillback its junction order differs, and jams clear more slowly.

### HermesSim

* MATSim's Hermes idea: whole-second ticks, built for speed.
* Blocked cars wait in line, where Hermes retries every second.
* Each car has its own stuck timer (in Hermes, each link has one).
* Capacity is exact over time: a link passing one car per 1.8 s gives gaps
  of 2, 2, 2, 2, 1 s.
* With whole-second link times it gives exactly the QSim's trips.

### PSim

* HermesSim rebuilt for many threads, with the network split into regions
  (`partitioner`: morton or metis).
* Junctions: the longest wait goes first, decided once per second, so
  results never depend on thread order.
* In free flow it gives HermesSim's trips. Links shorter than a second take a
  full second.
* With no stuck time, this rule breaks up streams of cars, so jams clear more
  slowly.

### DSim

* matsim-rust's design: one worker per network region (`partitioner`:
  metis).
* Junctions: a lottery weighted by link capacity (`seed`).
* MATSim's clock by default: a second at each junction, times rounded down,
  activities starting a second after arrival.
* Arriving cars leave freely.
* `buffer_in_storage=False`: cars waiting at a junction free their link's
  space, as in matsim-rust. By default they still fill it.
* Differs from matsim-rust: other random numbers, results independent of
  the split, teleports take at least 1 s, no kinematic waves, no
  within-day rerouting.

## Shared rules

* People are at nodes, not on links. A trip starts by entering its first
  link (see [section 5 of the design notes](../design.md)).
* A car fills its link until it crosses the junction at the end.
* `stuck_time` pushes a blocked car on: QSim and AgentSim exactly, HermesSim
  and PSim at the next second, DSim after it has *exceeded* it (10 s waits
  11). The config default is 11 s, MATSim's 10 plus its one-second step.

## Choosing the clock

Every mobsim takes three timing parameters (also under `mobsim:` in a
config):

* `continuous_time`: exact times, or whole seconds.
* `junction_delays`: one second at each junction, and one before an activity
  starts, as MATSim's steps work out.
* `rounding`: with whole seconds, link and walk times round to the `nearest`
  second, or `down`. Activity ends always round up.

| | `continuous_time` | `junction_delays` | `rounding` |
|---|---|---|---|
| QSim, AgentSim | **true** or false | **false** or true | **nearest** or down |
| HermesSim, PSim | false only | **false** or true | **nearest** or down |
| DSim | false only | **true** or false | **down** or nearest |

Defaults are in bold. Settings a mobsim cannot do raise an error. On the
same settings, every mobsim keeps time identically. Here is one car on three
links of 10.5 s each, leaving at 100 s:

| `continuous_time`, `junction_delays`, `rounding` | enters link 2 | enters link 3 | arrives | activity starts |
|---|---|---|---|---|
| true, false (QSim, AgentSim) | 110.5 | 121 | 131.5 | 131.5 |
| false, false, nearest (HermesSim, PSim) | 111 | 122 | 133 | 133 |
| false, true, down (DSim, MATSim) | 111 | 122 | 132 | 133 |

* With whole-second link times, the first two give the same times.
* MATSim's clock adds up to a second per link, about 5% on the test grids.

## Public transport

* Every mobsim, with every clock setting.
* `vehicles_on: network`: vehicles drive in the traffic, wait at stops, and
  never leave early. `vehicles_on: schedule`: they keep exactly to the
  timetable.
* `schedule_modes: [rail]` keeps some modes on the timetable while the
  rest drive (MATSim's deterministic pt).
* With junction delays, a vehicle serves a stop at the end of its link, and
  crosses into the next link one second after leaving the stop.
* Parallel mobsims: a vehicle starts at the end of its first link without
  waiting for space there.

## Measured differences

* **Congestion, stuck time 10 s** (grids at 5% capacity, total travel time
  against the QSim): AgentSim, HermesSim and PSim are within 0.2%. DSim is
  3–7% faster, mostly because its arriving cars leave freely.
* **Congestion, no stuck time:** the junction rules decide everything.
  HermesSim stays within 1%, AgentSim is +6% (20×20 grid), PSim +15 to +33%, and DSim
  −24 to −42%.
* **Against MATSim on equil** (frozen MATSim 2025 runs,
  `tests/test_matsim_reference.py`): every mobsim matches the mean trip time
  within 0.05%. DSim is closest link by link (0.2–1.0 s mean error, against
  0.5–1.4 s for the others).
* **Against MATSim on pt-tutorial** (vehicles in the traffic): on MATSim's
  clock every mobsim's mean link-time error is under 0.02 s, and its
  stop-arrival error 0.76 s. On their own clocks it is 0.8 s and 2 s.
