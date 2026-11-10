| run | s per iteration | final utility | reaches 129.19 at | wall time | vs Controller | utility after 47 s |
|---|---|---|---|---|---|---|
| Controller | 0.47 | 129.19 | iteration 97 | 46 s | reference | 129.26 |
| 1 strand | 0.45 | 129.12 | iteration 98 | 45 s | 1.02× | 129.21 |
| 2 strands | 0.78 | 132.09 | iteration 50 | 40 s | 1.15× | 129.83 |
| 4 strands | 1.29 | 135.56 | iteration 27 | 36 s | 1.26× | 130.22 |
| 8 strands | 2.53 | 136.97 | iteration 16 | 44 s | 1.03× | 129.25 |

Utility is the mean score of everything simulated in an iteration (all strands), as a 5-iteration rolling mean. *Final utility* is the mean over the last 10 iterations. *Reaches* is when a run first matches Controller's final utility; the last column is each run's utility once Controller has finished (47 s). Wall time starts after the one-off routing of the input plans (the same for every run) and includes starting the worker processes.
