# MobSlim simulations


```{raw} html
<figure class="ov-diagram" aria-label="Plans are encoded once into flat arrays, run by one of five engines, and come back as trajectory arrays from which travel times, scores and optional events are computed">
<svg viewBox="0 0 760 300" role="img" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ov-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="ov-ink-fill"/>
    </marker>
  </defs>
  <!-- inputs -->
  <rect x="10" y="110" width="120" height="80" rx="8" class="ov-box"/>
  <text x="70" y="140" class="ov-title">Scenario</text>
  <text x="70" y="160" class="ov-sub">network, plans,</text>
  <text x="70" y="175" class="ov-sub">transit</text>
  <line x1="130" y1="150" x2="168" y2="150" class="ov-line" marker-end="url(#ov-arrow)"/>
  <rect x="170" y="110" width="120" height="80" rx="8" class="ov-box"/>
  <text x="230" y="140" class="ov-title">PlanBatch</text>
  <text x="230" y="160" class="ov-sub">plans as flat</text>
  <text x="230" y="175" class="ov-sub">integer arrays</text>
  <!-- engines -->
  <rect x="320" y="10" width="220" height="280" rx="10" class="ov-group"/>
  <text x="430" y="32" class="ov-title">Simulations</text>
  <g>
    <rect x="335" y="45" width="190" height="40" rx="6" class="ov-py"/>
    <text x="430" y="63" class="ov-title">QSim</text><text x="430" y="78" class="ov-sub">Python · event-driven</text>
    <rect x="335" y="91" width="190" height="40" rx="6" class="ov-py"/>
    <text x="430" y="109" class="ov-title">AgentSim</text><text x="430" y="124" class="ov-sub">Python · agent-driven</text>
    <rect x="335" y="137" width="190" height="40" rx="6" class="ov-rs"/>
    <text x="430" y="155" class="ov-title">HermesSim</text><text x="430" y="170" class="ov-sub">Rust · whole seconds</text>
    <rect x="335" y="183" width="190" height="40" rx="6" class="ov-rs"/>
    <text x="430" y="201" class="ov-title">PSim</text><text x="430" y="216" class="ov-sub">Rust · threads, synchronous steps</text>
    <rect x="335" y="229" width="190" height="40" rx="6" class="ov-rs"/>
    <text x="430" y="247" class="ov-title">DSim</text><text x="430" y="262" class="ov-sub">Rust · threads, matsim-rust design</text>
  </g>
  <line x1="290" y1="150" x2="318" y2="150" class="ov-line" marker-end="url(#ov-arrow)"/>
  <line x1="540" y1="150" x2="568" y2="150" class="ov-line" marker-end="url(#ov-arrow)"/>
  <!-- outputs -->
  <rect x="570" y="110" width="180" height="80" rx="8" class="ov-box"/>
  <text x="660" y="140" class="ov-title">Trajectories</text>
  <text x="660" y="160" class="ov-sub">link entry/exit times,</text>
  <text x="660" y="175" class="ov-sub">activity times (arrays)</text>
</svg>
</figure>
```

| | QSim | AgentSim | HermesSim | PSim | DSim |
|---|---|---|---|---|---|
| **Language** | Python | Python | Rust (and a Python reference kernel) | Rust (and a Python reference kernel) | Rust |
| **Scheduling** | event-driven: one wake-up per link head | agent-driven: each agent schedules its own next move | per-second buckets, drained in order | one synchronous step per second | one step per second, neighbour sync |
| **Clock** | exact or whole seconds | exact or whole seconds | whole seconds | whole seconds | whole seconds |
| **Threads** | 1 | 1 | 1 | any number, same results | any number, same results |
| **Closest MATSim** | QSim | none | Hermes | none | matsim-rust |
| **Use it for** | the readable reference | experimenting with agent logic | fastest single-threaded runs | large networks on many cores | comparing designs with matsim-rust |

All five simulation use a meso-scale queue model for traffic simulation (as per MATSim): free-speed time, flow capacity with fractional headways, storage capacity and spillback, first-in first-out links, and a stuck time. Buses and trains drive in traffic or keep to the timetable, with boarding, alighting and door times. The full design notes are in [MobSlim Design](design.md#33-mobsim-mobsim).

## Core design ideas

Every mobsim implements the same one-method interface,
`run(scenario, events) -> Trajectories`.

Where possible data is held as python numpy arrays to allow easy parallelisation and cheap passing into fast rust kernels:

* **Plans go in as arrays.** `PlanBatch` encodes selected plans as activity ends, modes, and routes as packed link indices.
* **Trajectories come out as arrays.** Each vehicle's link entry and exit times, and each activity's start and end. Travel times, scores and experienced plans are numpy operations.
* **Events are optional.** MATSim-style events (`entered link`, `PersonEntersVehicle` and so on) are built only for handlers that ask for them, for example to write the `events.xml` used on the [correctness page](correctness.md). A normal iteration builds none.

## Verification

* **Against MATSim.** The [verification suite](https://github.com/fredshone/mobslim/blob/main/verification/README.md) compares every mobsim with Java MATSim 2025.0 on MATSim's example scenarios, and scoring against MATSim's scoring to floating-point precision.
* **Consistency.** With the same settings, our QSim equals our HermesSim move for move. PSim and DSim give the same results on any partition of the network.
* **Deterministic.** The same seed gives the same results, every time, on any number of threads.


