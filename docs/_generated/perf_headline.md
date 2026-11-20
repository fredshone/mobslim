| Scenario | Agents | MATSim Hermes | Fastest MobSlim | Speed-up | Peak memory: MATSim Hermes → that MobSlim |
|---|--:|---|---|---|---|
| grid8km | 26,240 | 934 ms | 136 ms (HermesSim) | **6.9×** | 1.59 GB → 227 MB |
| grid16km | 104,960 | 10.1 s | 1.13 s (HermesSim) | **8.9×** | 6.34 GB → 653 MB |
| grid32km | 419,840 | 73.7 s | 8.48 s (PSim, 16 threads) | **8.7×** | 38.12 GB → 3.28 GB |
| open-berlin-1pct | 52,263 | 9.54 s | 1.94 s (HermesSim) | **4.9×** | 9.97 GB → 3.55 GB |
| open-berlin-10pct | 526,111 | 91.8 s | 7.63 s (PSim, 16 threads) | **12.0×** | 49.27 GB → 7.26 GB |
