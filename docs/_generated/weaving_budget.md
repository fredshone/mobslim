| run | s per iteration | final utility | reaches 129.16 at | wall time | vs p=8, w=1 | utility after 52 s |
|---|---|---|---|---|---|---|
| p=8, w=1 | 0.52 | 129.16 | iteration 96 | 51 s | reference | 129.25 |
| p=4, w=2 | 0.87 | 132.09 | iteration 49 | 43 s | 1.17× | 129.91 |
| p=2, w=4 | 1.69 | 135.54 | iteration 26 | 46 s | 1.10× | 129.58 |
| p=1, w=8 | 3.56 | 136.97 | iteration 16 | 62 s | 0.81× | 128.41 |

Utility is the mean score of everything simulated in an iteration (all strands), as a 5-iteration rolling mean. *Final utility* is the mean over the last 10 iterations. *Reaches* is when a run first matches p=8, w=1's final utility; the last column is each run's utility once p=8, w=1 has finished (52 s). Wall time starts after the one-off routing of the input plans (the same for every run) and includes starting the worker processes.
