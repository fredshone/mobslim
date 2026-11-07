"""ShardSim: matsim-rust's parallel design, on mobslim's plans.

A second parallel mobsim, next to :mod:`mobslim.mobsim.parallel`, built to
compare the two designs. It follows matsim-rust
(https://github.com/matsim-vsp/matsim-rust): reimplemented from its
documentation and design, no code taken (that project is GPL-3.0).

**Design.** The nodes are split into partitions (by default with METIS, as
matsim-rust does), one worker thread each. A worker owns everything on its
partition: its nodes, the links into them, the entry side (storage,
departures) of the links out of them, and the persons there. Moves inside a
partition take effect at once. A link between two partitions is *split*:
vehicles entering it are sent to the downstream worker and the space they
free is sent back. Every second, each worker exchanges one message with
each *neighbour* (a partition it shares a link with) and waits for its
neighbours only, not for all workers. It runs every second from the first
activity end until everyone is done (or ``end_time``).

**Link model**, as matsim-rust's queue dynamics (and MATSim's QSim), not
HermesSim's: a link has a queue and a buffer. A vehicle moves from queue to
buffer once past its free-speed time (floored to whole seconds, at least
one) if the link has flow capacity, accumulated per second and consumed on
entering the buffer. A node picks its
non-empty in-link buffers at random, weighted by flow capacity (one seeded
stream per node), and empties each into the next links while they have
storage, or the front vehicle has waited longer than ``stuck_time``.
Departures wait at their first link until it has storage (or they are
stuck); vehicles at the end of their route leave without using flow
capacity. An arrival starts its activity one second later; teleports take
at least one second. The extra second at each node and before an activity,
and free-speed times rounded down, are the default timing
(``junction_delays=True``, ``rounding="down"``); with
``junction_delays=False`` a tick moves links before nodes, so a vehicle
crosses a node in the tick it reaches it, and activities start on arrival.

**One deviation from matsim-rust, by default:** a vehicle in a buffer
keeps its storage on the link until it crosses the node, as in mobslim's
other mobsims, so a link never holds more than its storage capacity (bar
vehicles forced on after ``stuck_time``).
matsim-rust frees the storage when the vehicle enters the buffer, so each
link holds about one second's flow more; ``buffer_in_storage=False`` does
that.

**Determinism.** Trajectories do not depend on the number of threads or
the partition (``tests/test_shard.py`` checks it); events come in a
canonical order (by second, person, step). They depend on ``seed``.

Only the compiled kernel exists (``rust/mobslim-shard``; ``uv sync --group
rust``).

**Transit** follows the shared rules (``tests/test_transit.py``) on the
same clock as the other whole-second mobsims (see
:func:`~mobslim.mobsim._transit_tables.transit_times`), in MATSim's way: a
transit vehicle serves a stop when it reaches the front of its link's
queue, before moving to the buffer (at a blocking stop it holds the queue;
at a non-blocking one it leaves it and rejoins at the front), so with
junction delays it crosses the node one tick after it leaves the stop. A
link runs its stops' tasks (doors, boarding, alighting, departures) in the
links step; a trip's state travels with its vehicle. A trip in network mode
appears at the end of its first link at its departure time, taking its
storage from the next tick. With transit, workers that are not neighbours
stay within one tick of each other (passengers can walk anywhere).
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mobslim.events import EventsManager
from mobslim.mobsim._common import (
    check_timing,
    link_parameters,
    round_steps,
    validate,
)
from mobslim.mobsim.hermes import _event_mask, kernel_results
from mobslim.mobsim.parallel import parallel_inputs
from mobslim.mobsim.partition import METHODS as PARTITIONERS
from mobslim.mobsim.partition import partition
from mobslim.planbatch import PlanBatch
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories

try:  # the compiled kernel (rust/mobslim-shard); optional
    import mobslim_shard as _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None


def rust_available() -> bool:
    """Whether the compiled kernel (``mobslim_shard``) is installed."""
    return _rust is not None


@dataclass(frozen=True)
class ShardSimParams:
    flow_capacity_factor: float = 1.0
    storage_capacity_factor: float = 1.0
    stuck_time: float = 10.0
    effective_cell_size: float = 7.5  # meters per vehicle (PCU)
    end_time: Optional[float] = None  # None: run until every plan is done
    validate_routes: bool = True
    threads: Optional[int] = None  # workers (= partitions); None: one per CPU
    partitioner: str = "metis"  # "metis" (as matsim-rust) or "morton"
    seed: int = 4711  # the nodes' random streams
    # True: a vehicle waiting in a link's buffer still takes storage there
    # until it crosses the node (as in mobslim's other mobsims); False: it
    # frees it on entering the buffer, as matsim-rust does
    buffer_in_storage: bool = True
    # timing (see mobslim.mobsim._common.check_timing): whole seconds only;
    # matsim-rust's: junction delays, road and teleport times rounded down
    continuous_time: bool = False
    junction_delays: bool = True
    rounding: str = "down"

    def __post_init__(self):
        if not self.stuck_time > 0:
            raise ValueError("stuck_time must be positive (inf disables it).")
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be at least 1.")
        if self.partitioner not in PARTITIONERS:
            raise ValueError(f"partitioner must be one of {PARTITIONERS}, not {self.partitioner!r}.")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be in [0, 2**64).")
        check_timing("ShardSim", self.continuous_time, self.junction_delays, self.rounding,
                     continuous=False, whole=True)


class ShardSim:
    """matsim-rust's parallel queue mobsim (see the module docstring).
    Stateless between runs."""

    def __init__(self, params: ShardSimParams = ShardSimParams()):
        self.params = params
        if _rust is None:
            raise ImportError(
                "ShardSim needs the mobslim_shard extension (rust/mobslim-shard; "
                "`uv sync --group rust` builds it when cargo is installed)."
            )

    @property
    def threads(self) -> int:
        """Worker threads (= network partitions)."""
        if self.params.threads is not None:
            return self.params.threads
        return os.cpu_count() or 1

    def run(self, scenario: Scenario, events: EventsManager) -> Trajectories:
        plans = scenario.population.selected_batch()
        return self.run_batch(scenario, plans, events)

    def run_batch(self, scenario: Scenario, plans: PlanBatch, events: EventsManager,
                  node_part: Optional[np.ndarray] = None) -> Trajectories:
        """Run already-encoded plans (``plans`` row ``i`` = person index ``i``).

        ``node_part`` assigns each node to a partition (default: by
        ``params.partitioner``); trajectories do not depend on it.
        """
        if plans.n_plans != len(scenario.population):
            raise ValueError("PlanBatch must have one plan per person.")
        cfg = self.params
        if cfg.validate_routes:
            validate(scenario, plans)
        inputs, tele_distance = shard_inputs(scenario, plans, cfg, _event_mask(events))
        net = scenario.network
        if node_part is None:
            node_part = partition(net, plans, self.threads, cfg.partitioner)
        node_part = np.asarray(node_part, dtype=np.int64)
        out = _rust.run(**inputs, node_part=node_part,
                        n_parts=int(np.max(node_part, initial=0)) + 1, seed=int(cfg.seed),
                        buffer_in_storage=bool(cfg.buffer_in_storage))
        return kernel_results(scenario, plans, events, inputs, tele_distance, out)


def shard_inputs(scenario: Scenario, plans: PlanBatch, cfg: ShardSimParams,
                 event_mask: int) -> tuple[dict, np.ndarray]:
    """The kernel's keyword arguments (without the partition and seed), plus
    each leg's teleport distance."""
    inputs, tele_distance = parallel_inputs(scenario, plans, cfg, event_mask)
    free_time, headway, _ = link_parameters(
        scenario.network, cfg.flow_capacity_factor, cfg.storage_capacity_factor,
        cfg.effective_cell_size)
    del inputs["free_steps"], inputs["headway"], inputs["rs_node"]
    if scenario.transit is None:
        inputs["rs_link"] = np.zeros(0, dtype=np.int64)
    else:
        s = scenario.transit.schedule
        inputs["rs_link"] = s.stop_link[s.rs_stop].astype(np.int64)
    inputs["queue_steps"] = np.maximum(round_steps(free_time, cfg.rounding), 1)
    headway = np.asarray(headway, dtype=float)
    with np.errstate(divide="ignore"):
        inputs["cap"] = np.where(np.isfinite(headway), 1.0 / headway, 0.0)
    return inputs, tele_distance
