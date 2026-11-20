# Correctness

MobSlim's scorer is checked against Java MATSim in two ways. The first
uses the same days: MATSim's events from MATSim's own run, scored by both
scorers. The second uses whole runs: MobSlim's mobsims on MATSim's plans,
against MATSim's scores.

## The same days, scored by both

`tests/test_matsim_reference.py` holds three frozen runs of MATSim 2025.0
(iteration 0). For each one, `verification/experienced.py` turns MATSim's
events into MobSlim's experienced plans, and MobSlim's scorer scores them:

| run | agents | MATSim's mean score | largest difference |
|---|---|---|---|
| equil, 100 agents | 100 | 64.76 | 4 × 10⁻¹⁴ |
| equil, 2,000 agents at half capacity | 2,000 | 104.65 | 6 × 10⁻¹⁴ |
| pt-tutorial (car, pt, walk; work, shop and home) | 900 | 131.95 | 9 × 10⁻¹⁴ |

Every agent's score agrees to floating-point rounding, and the test fails
if any difference reaches 10⁻⁹. Two days from those runs:

**Equil, agent 999** drives to work for 06:15 and leaves at 14:15, the day
on the [Scoring](index.md) page:

| part | utils |
|---|---|
| work, 06:15–14:15 (open from 07:00) | 43.27 |
| home, 14:54 to 06:00 the next day | 88.54 |
| car, 15 min and 39 min | −1.50, −3.90 |
| **day: MobSlim / MATSim** | **126.42 / 126.42** |

**pt-tutorial, agent 525** takes pt to work, then pt to the shop, then
walks home:

| part | utils |
|---|---|
| work, 07:42–17:06 | 55.71 |
| shop, 17:24–17:53 | 2.69 |
| home, 18:17 to 07:00 the next day | 76.17 |
| pt (with its waits), pt, walk | −4.24, −1.84, −2.39 |
| **day: MobSlim / MATSim** | **126.11 / 126.11** |

MATSim writes only each day's total, so the parts shown are MobSlim's.

## Open Berlin

On [Open Berlin](../simulation/performance.md#open-berlin-1pct) at 1%
(52,263 agents, over 1,100 activity types, five subpopulations), the same check
found two rules MobSlim had wrong. Both are now as MATSim scores them:

* **A person who stays home all day scores 0**, because MATSim's scoring
  sees no activity events for them. MobSlim used to give them a full day at
  home (165.12 utils). 5,035 Berlin residents never leave home, so this
  alone lifted MobSlim's mean score by 15.9.
* **A person who is stuck at the end of the day has no evening
  activity.** Their first activity counts until they left it, and every
  activity they left counts as performed. MobSlim used to stretch the last
  activity they reached to the end of the day.

On MATSim Hermes's iteration-0 events, MATSim's mean score is 111.73 and
MobSlim's scorer gives 111.63. 145 of MATSim's 160 stuck agents agree within
0.01. The rest of the difference comes from how MATSim's events are read,
not from the scorer: MobSlim folds each trip's walks into its main leg, so
walking is charged at the main mode's rates. That matters most for long
walks, such as one Berlin agent's 8-hour walk from a ride drop-off.

## Whole runs

When MobSlim also runs the simulation, the scores differ as much as the
simulated days do. On Open Berlin's calibrated plans and routes, MobSlim's
HermesSim averages 110.74 at iteration 0, against MATSim Hermes's 111.73.
Commercial and goods traffic score 2.7 to 6.8 lower, and more agents get
stuck (see the [simulation performance](../simulation/performance.md) notes).
On the reference runs, `tests/test_matsim_reference.py` also checks that
each mobsim stays no further from MATSim than when the runs were frozen.
