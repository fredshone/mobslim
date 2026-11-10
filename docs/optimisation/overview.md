# Overview

## The controller

`Controller` runs MATSim's co-evolutionary loop. Every person holds a few
plans with scores. Each iteration:

1. **Replan.** Each person draws a strategy by weight. Most *select* one of
   their plans (ChangeExpBeta). A share *innovate*: they copy a plan,
   change it (reroute, shift activity times, change mode) and execute the
   new plan.
2. **Simulate** everyone's selected plan once.
3. **Score** what happened, and store each executed plan's score.
4. **Learn travel times** from the simulation, for the next routing.

The innovation share stays small (MATSim's default is 10-20%). If too many
people change plan at once, the network they planned for no longer exists
and the run oscillates. The cost is speed: with 10% innovating, each person
gets a new plan about once every ten iterations.

## Weaving

`Weave` keeps each simulated world realistic but runs several of them per
iteration. Each world, or *strand*, has a different cohort innovating:

```{raw} html
<figure class="ov-diagram" aria-label="A ledger of everyone's plans and scores feeds K strands; each strand has a different cohort innovating and everyone else replaying a plan from the ledger; their scores and travel times merge back into the ledger">
<svg viewBox="0 0 760 250" role="img" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="wv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="ov-ink-fill"/>
    </marker>
  </defs>
  <rect x="10" y="85" width="140" height="80" rx="8" class="ov-box"/>
  <text x="80" y="115" class="ov-title">Ledger</text>
  <text x="80" y="135" class="ov-sub">everyone's plans,</text>
  <text x="80" y="150" class="ov-sub">scores, sample counts</text>
  <line x1="150" y1="125" x2="198" y2="125" class="ov-line" marker-end="url(#wv-arrow)"/>
  <rect x="200" y="85" width="130" height="80" rx="8" class="ov-box"/>
  <text x="265" y="115" class="ov-title">Innovate</text>
  <text x="265" y="135" class="ov-sub">K cohorts, routed</text>
  <text x="265" y="150" class="ov-sub">in one batch</text>
  <rect x="360" y="10" width="200" height="230" rx="10" class="ov-group"/>
  <text x="460" y="32" class="ov-title">Strands (in parallel)</text>
  <rect x="375" y="45" width="170" height="40" rx="6" class="ov-rs"/>
  <text x="460" y="63" class="ov-title">strand 1</text><text x="460" y="78" class="ov-sub">cohort 1 innovates</text>
  <rect x="375" y="95" width="170" height="40" rx="6" class="ov-rs"/>
  <text x="460" y="113" class="ov-title">strand 2</text><text x="460" y="128" class="ov-sub">cohort 2 innovates</text>
  <text x="460" y="160" class="ov-sub">…</text>
  <rect x="375" y="180" width="170" height="40" rx="6" class="ov-rs"/>
  <text x="460" y="198" class="ov-title">strand K</text><text x="460" y="213" class="ov-sub">cohort K innovates</text>
  <line x1="330" y1="125" x2="358" y2="125" class="ov-line" marker-end="url(#wv-arrow)"/>
  <line x1="560" y1="125" x2="588" y2="125" class="ov-line" marker-end="url(#wv-arrow)"/>
  <rect x="590" y="85" width="160" height="80" rx="8" class="ov-box"/>
  <text x="670" y="115" class="ov-title">Merge</text>
  <text x="670" y="135" class="ov-sub">scores per plan,</text>
  <text x="670" y="150" class="ov-sub">summed travel times</text>
</svg>
</figure>
```

1. **Allocate.** A scheduler picks `K` cohorts of persons who innovate, one
   per strand. People who have waited longest since their last innovation
   are more likely to be picked.
2. **Innovate** every cohort's new plans at once, routed in one batch.
3. **Build the strands.** In strand `j`, cohort `j` executes its new plans
   and everyone else executes a plan that ChangeExpBeta picks from the
   ledger, with strand `j`'s own random draws.
4. **Simulate and score** the strands in parallel worker processes.
5. **Merge** into the ledger. Each executed plan's score is the mean of its
   samples across strands, blended with its old score. New plans join as
   scored plans, and each person's worst plans are dropped back to
   `max_plans`. Travel times are learned from all strands' link
   traversals together.

With innovation rate `r` per strand and `K` strands, each person innovates
`r × K` times per iteration on average. That can exceed 1, yet each
simulated world only ever has a share `r` of people trying something new.
`Weave(strands=1)` behaves like `Controller`.
