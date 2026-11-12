# Details

**Selected plans.** After a merge, each person's selected plan is the one
they executed in strand 0. The next iteration's ChangeExpBeta starts from
it in every strand. Listeners see strand 0: its events, its trajectories
and its scenario (`state.strand == 0`). `state.strands` holds every
strand's result.

**Iteration 0** runs one strand. Nobody has innovated yet, so every strand
would simulate the same plans.

**Scores.** A plan may be executed in several strands in one iteration, or
in none. `MeanReducer` blends the mean of this iteration's samples into the
stored score with `learning_rate`. A plan with no samples keeps its score.
The ledger counts samples per plan (`score_n`), which `PrecisionReducer`
uses to weight old against new. New plans arrive already scored, so the
worst-score removal can drop a bad one at once.

**Travel times.** Each strand returns sums of link traversal times per time
bin (a `TraversalAggregate`, only the cells used), not trajectories; the
compiled mobsims build them while simulating. The merge adds them, so the
learned times average over all strands. `travel_time_smoothing < 1` also
averages across iterations.

**Determinism.** Random draws are keyed by `(seed, iteration, purpose,
strand)`. A seeded run gives the same result with any loom and any number
of workers.

**Core budget.** Strands times the mobsim's own threads should not exceed
the cores: 8 strands of HermesSim, or 4 strands of 2-thread PSim, on 8
cores. Hyperthreads count for less than cores (see
[Performance](performance.md)). Each strand mutates and routes its own
innovators before simulating them, so routing runs in parallel too. Each
strand routes on its share of the logical CPUs (logical CPUs divided by
workers plus one, for the main process; `ProcessLoom(router_threads=...)`
or `$MOBSLIM_ROUTER_THREADS` to override). Routing waits on memory, so
while the other strands simulate, a second routing thread on each core's
hyperthread routes about twice as fast. Only the choice of who innovates,
with which strategy and from which plan, is made in the main process.

**Pool size.** The pool's size is independent of the number of strands: jobs
queue when there are more strands than workers. The default is one worker
per strand after the first, at most one fewer than the physical cores.
Laptop CPUs slow down a lot with every core busy, so measure on the machine
you will use.

**Memory.** Each worker holds the network, the transit schedule and the
plans its iteration's strands use: each person's plan in every strand plus
the innovators' source plans, about one plan per person, not the whole plan
store. These are written once per iteration to shared memory (`/dev/shm`)
and read by each worker.

**Workers.** Strands run in `spawn`ed processes by default. The mobsim and
scorer must be picklable, and scripts need an `if __name__ == "__main__":`
guard. `ProcessLoom(start_method="fork")` avoids the guard on Linux.

**Limits.** Strategies run through the vectorised replanning path. Custom
innovative strategies without `mutate_batch` modules run per person, which
is slower. Each iteration waits for all its strands; there is no
asynchronous loop yet, but strands are built from the ledger alone and carry
the ledger version they were built from, ready for one.
