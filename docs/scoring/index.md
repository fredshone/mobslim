# Scoring

A score is how much an agent's day was worth, in utils. After every
iteration, each agent scores the day it just experienced, and the score
decides which of its plans it is likely to keep (`ChangeExpBeta`, for
example, switches plans with a probability that grows with the score
difference). Scores only matter relative to each other: a plan that scores
2 utils more is chosen more often.

* **[Correctness](correctness.md)**: MobSlim's scores against MATSim's, on
  the same days.
* **[Configuration](configuration.md)**: scoring parameters in a MATSim
  config, a MobSlim config, or Python.

```{toctree}
:hidden:

correctness
configuration
```

## What a day is worth

`CharyparNagelScorer` is MATSim's default scoring function (Charypar and
Nagel, 2005):

* **Activities earn utility with diminishing returns.** An activity
  performed for `t` hours earns `performing · t_typ · ln(t / t_0)`.
  `t_typ` is the activity's typical duration, and `t_0` is the duration
  at which it is worth nothing (`t_typ / e` by default). Below `t_0`, the
  score keeps falling linearly, so very short stays cost utility.
* **Time counts only while the activity is open.** Arriving before
  `opening_time` is waiting, and so is staying after `closing_time`.
  Arriving after `latest_start_time` pays `late_arrival` per hour, and
  leaving before `minimal_duration` pays `early_departure` per hour.
* **Legs cost.** Each leg pays its mode's constant, plus utility per hour
  of travel and per metre (or per unit of money spent on distance).
* **The night wraps round.** The first and last activities of the day are
  scored as one overnight activity if they have the same type, for example
  home from the evening to the next morning.
* **A person who gets stuck pays `stuck_penalty`.** As in MATSim, the
  default is 24 hours at the worst of late arrival, early departure, and
  travelling or waiting instead of performing.

So each day balances one activity against another. In MATSim's *equil*
scenario, a commuter leaves home at 06:00, drives 15 minutes to work, and
drives 39 minutes home:

```python
from mobslim import ActivityParams, ModeParams, ScoringParams
from mobslim.experience import ExperiencedActivity, ExperiencedLeg, ExperiencedPlan
from mobslim.scoring import CharyparNagelScorer

H = 3600
scorer = CharyparNagelScorer(ScoringParams(
    activities={"h": ActivityParams(12 * H, minimal_duration=8 * H),
                "w": ActivityParams(8 * H, minimal_duration=6 * H, opening_time=7 * H,
                                    closing_time=18 * H, latest_start_time=9 * H)},
    modes={"car": ModeParams(marginal_utility_of_traveling=-6.0)},
))

def day(leave_work):
    """Leave home at 06:00, 15 min to work, 39 min home."""
    return ExperiencedPlan((
        ExperiencedActivity("h", 0, None, 6 * H),
        ExperiencedLeg("car", 6 * H, 6.25 * H, 25_000, ()),
        ExperiencedActivity("w", 1, 6.25 * H, leave_work),
        ExperiencedLeg("car", leave_work, leave_work + 0.65 * H, 65_000, ()),
        ExperiencedActivity("h", 0, leave_work + 0.65 * H, None),
    ))

for hours in (8, 11, 14.25, 16, 17, 19):
    print(f"leave work at {hours:5.2f} h: {scorer.score(day(hours * H)):7.2f}")
```

```text
leave work at  8.00 h:   69.02
leave work at 11.00 h:  111.91
leave work at 14.25 h:  126.42
leave work at 16.00 h:  127.93
leave work at 17.00 h:  127.38
leave work at 19.00 h:  119.24
```

Work opens at 07:00, so the 45 minutes before then earn nothing. Leaving at
08:00 leaves one hour of work, below the work activity's `t_0` of 2.9 hours.
The best time to leave is 16:00, when another hour at work is worth no more
than another hour at home. Work closes at 18:00, so staying until 19:00
only takes an hour from home.

Leaving at 14:15 is what one of equil's commuters does in MATSim's own run,
and MATSim scores that day 126.42 too (see [Correctness](correctness.md)).
