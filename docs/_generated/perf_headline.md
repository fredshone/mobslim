| Scenario | Agents | MATSim Hermes | Fastest MobSlim | Speed-up | Peak memory: MATSim Hermes → that MobSlim |
|---|--:|---|---|---|---|
| grid-40 | 26,240 | 1.08 s | 127 ms (HermesSim) | **8.5×** | 1.54 GB → 222 MB |
| grid-40-congested | 26,240 | 934 ms | 136 ms (HermesSim) | **6.9×** | 1.59 GB → 227 MB |
| grid-80 | 104,960 | 11.4 s | 906 ms (PSim, 16 threads) | **12.6×** | 6.63 GB → 727 MB |
| grid-80-congested | 104,960 | 10.1 s | 1.13 s (HermesSim) | **8.9×** | 6.34 GB → 653 MB |
| grid-160 | 419,840 | 109 s | 8.01 s (PSim, 16 threads) | **13.6×** | 40.68 GB → 3.18 GB |
| grid-160-congested | 419,840 | 73.7 s | 8.48 s (PSim, 16 threads) | **8.7×** | 38.12 GB → 3.28 GB |
| siouxfalls | 84,110 | 1.60 s | 185 ms (HermesSim) | **8.7×** | 3.58 GB → 553 MB |
