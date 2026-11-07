"""ParallelSim: a whole-second queue mobsim that runs on several threads.

The network's nodes are split into partitions, one per thread. Each second
is one *synchronous step*, and the rules are written so that no decision
depends on how the nodes are partitioned. **Results are identical for any
number of threads**, including one, and between the Python reference kernel
in this module and the Rust kernel (``rust/mobslim-parallel``), which runs
the partitions in parallel.

The queue model is MATSim's (and the QSim's): free-speed time, flow capacity
with exact fractional headways (HermesSim's whole-second rule), storage
capacity and spillback, FIFO links, and stuck time. What makes it parallel is
*where* each decision is taken and *when* its effects become visible:

* **A node decides every move through it.** In each step, a node takes the
  vehicles that may leave its incoming links (past their free-speed time and
  flow slot) and the vehicles waiting to depart onto its outgoing links, and
  moves them, best first, while the target link has storage. Vehicles that
  reach the end of their route leave their last link here. A source (an
  incoming link, or a link's departure line) whose head is refused stops for
  this step: FIFO.
* **Who goes first** at a node: the vehicle that has waited longest: since
  it was first refused, if it was, else since it could first have left (or
  departed). Ties are broken by a hash of the second and the source, so no
  approach is systematically preferred.
* **Storage is owned by a link's from-node**, the only node that fills it.
  Space a vehicle frees by leaving a link reaches that node at the end of
  the step, so it can be reused from the next second.
* **Each link takes at least one second**, so a vehicle crosses at most one
  node per step. A person who arrives late for an activity leaves at once,
  as in MATSim, if its next leg departs from the node it arrived at (the
  usual case); otherwise one second later.
* **Every effect on another node or person is a message** delivered at the
  end of the step (a vehicle entering a link, space freed on a link, an
  activity or teleport ending later), even inside one partition. That is
  what makes the partitioning invisible: a step only reads state its own
  node owns, plus messages from earlier steps.

Compared with HermesSim (whose moves within one second chain through any
number of links and nodes): free-speed times shorter than 1 s take 1 s,
space freed on a link is reused one second later, teleports take at
least 1 s, and the order in which vehicles
from different approaches merge differs. Under free flow the trajectories
match HermesSim's to within these rules.

**Transit** follows the shared rules (``tests/test_transit.py``) on the
same whole-second clock as HermesSim (see
:func:`~mobslim.mobsim._transit_tables.transit_times`), with the same
ownership: a route stop (its waiting passengers and the trips there) belongs
to the to-node of its link, which runs the trips' stop tasks (doors,
boarding, alighting, departures) at the start of its node step, so a bus
that leaves a stop crosses the node in the same step. A trip in network
mode appears at the end of its first link at its departure time, taking its
storage from the next second (it is never refused there). Passengers and
trips moving to another node, or into a later second, are messages; a walk
of 0 s on the spot happens at once.

Events are recorded as the kernel runs and built afterwards, as in
HermesSim.
"""

import heapq
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mobslim.events import EventsManager
from mobslim.mobsim import hermes
from mobslim.mobsim._common import check_timing, validate
from mobslim.mobsim.hermes import (
    EV_ACT_END,
    EV_ACT_START,
    EV_ARRIVAL,
    EV_DEPARTURE,
    EV_DRIVER_START,
    EV_ENTER_VEH,
    EV_LEAVE_VEH,
    EV_LINK_ENTER,
    EV_LINK_LEAVE,
    EV_TELEPORT,
    EV_VEH_ARRIVE,
    EV_VEH_DEPART,
    EV_VEH_LINK_ENTER,
    EV_VEH_LINK_LEAVE,
    EV_WAITING,
    MAX_INDEX,
    N_EVENT_KINDS,
    _event_mask,
    ceil_step,
    kernel_inputs,
    kernel_results,
)
from mobslim.mobsim.partition import METHODS as PARTITIONERS
from mobslim.mobsim.partition import partition, partition_nodes  # noqa: F401
from mobslim.network import Network
from mobslim.planbatch import PlanBatch
from mobslim.scenario import Scenario
from mobslim.trajectories import Trajectories

try:  # the compiled kernel (rust/mobslim-parallel); optional
    import mobslim_parallel as _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None

BACKENDS = ("auto", "rust", "python")

# agenda entry kinds: persons (and trip starts, as persons n_plans + trip)
_ACT_END, _TELEPORT_ARRIVE, _ACT_START, _PAX_AT_STOP, _PAX_ARRIVE, _TRIP_START = range(6)
# trip tasks, run by the node where the trip is
_STOP_ARRIVE, _STOP_DEPART, _DOOR_OPEN, _ALIGHT_DONE, _BOARD_DONE, _DOOR_CLOSE = range(6)
# doors of a trip in network mode
_OPENING, _OPEN, _CLOSING, _CLOSED = 1, 2, 3, 4
# packed event codes, as HermesSim's
KIND_SHIFT, PERSON_SHIFT = hermes._KIND_SHIFT, hermes._PERSON_SHIFT
# event order within a second: person steps (activities, teleports) in person
# order, then node steps in node order
_PHASE_PERSON, _PHASE_NODE = 0, 1
_MASK64 = (1 << 64) - 1


def rust_available() -> bool:
    """Whether the compiled kernel (``mobslim_parallel``) is installed."""
    return _rust is not None


@dataclass(frozen=True)
class ParallelSimParams:
    flow_capacity_factor: float = 1.0
    storage_capacity_factor: float = 1.0
    stuck_time: float = 10.0
    effective_cell_size: float = 7.5  # meters per vehicle (PCU)
    end_time: Optional[float] = None  # None: run until every plan is done
    validate_routes: bool = True
    # threads (and network partitions) for the Rust kernel; None: one per
    # CPU. Results do not depend on it. The Python kernel runs on one thread.
    threads: Optional[int] = None
    # "rust": the compiled kernel (ImportError if missing); "python": the
    # reference kernel in this module; "auto": rust if installed
    backend: str = "auto"
    # how the Rust kernel splits the nodes: "morton" or "metis" (see
    # mobslim.mobsim.partition); only the speed depends on it
    partitioner: str = "morton"
    # timing (see mobslim.mobsim._common.check_timing): whole seconds only
    continuous_time: bool = False
    junction_delays: bool = False
    rounding: str = "nearest"

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {self.backend!r}.")
        if not self.stuck_time > 0:
            raise ValueError("stuck_time must be positive (inf disables it).")
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be at least 1.")
        if self.partitioner not in PARTITIONERS:
            raise ValueError(f"partitioner must be one of {PARTITIONERS}, not {self.partitioner!r}.")
        check_timing("ParallelSim", self.continuous_time, self.junction_delays, self.rounding,
                     continuous=False, whole=True)


class ParallelSim:
    """Whole-second queue mobsim on several threads, with results that do
    not depend on the number of threads. Stateless between runs."""

    def __init__(self, params: ParallelSimParams = ParallelSimParams()):
        self.params = params
        if params.backend == "rust" and _rust is None:
            raise ImportError(
                "ParallelSim backend='rust' needs the mobslim_parallel extension "
                "(rust/mobslim-parallel; `uv sync --group rust` builds it when cargo is installed)."
            )

    @property
    def backend(self) -> str:
        """The kernel that runs: ``"rust"`` or ``"python"``."""
        if self.params.backend == "auto":
            return "rust" if _rust is not None else "python"
        return self.params.backend

    @property
    def threads(self) -> int:
        """Threads (= network partitions) the Rust kernel uses."""
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
        ``params.partitioner``); results do not depend on it.
        """
        if plans.n_plans != len(scenario.population):
            raise ValueError("PlanBatch must have one plan per person.")
        cfg = self.params
        if cfg.validate_routes:
            validate(scenario, plans)
        inputs, tele_distance = parallel_inputs(scenario, plans, cfg, _event_mask(events))
        net = scenario.network
        if self.backend == "rust":
            if node_part is None:
                node_part = partition(net, plans, self.threads, cfg.partitioner)
            out = _rust.run(**inputs, node_part=np.asarray(node_part, dtype=np.int64),
                            n_parts=int(np.max(node_part, initial=0)) + 1)
        else:
            out = run_kernel(**inputs)
        return kernel_results(scenario, plans, events, inputs, tele_distance, out)


# ---------------------------------------------------------------- inputs

# the transit arguments of HermesSim's kernel, which ParallelSim's take too
TRANSIT_KEYS = (
    "network_vehicles", "leg_transit", "ride_offsets", "ride_board", "ride_alight",
    "walk_before_steps", "walk_after_steps", "rs_stop", "rs_link_pos", "rs_await",
    "stop_blocking", "trip_k0", "trip_k1", "trip_start_step", "trip_stop_offsets",
    "trip_stop_arr_step", "trip_stop_dep_step", "trip_link_offsets", "trip_links",
    "trip_link_steps", "trip_capacity", "trip_pce", "trip_access_steps", "trip_egress_steps",
    "trip_parallel", "door_open_steps", "door_close_steps", "trip_network", "rs_network")


def parallel_inputs(scenario: Scenario, plans: PlanBatch, cfg: ParallelSimParams,
                    event_mask: int) -> tuple[dict, np.ndarray]:
    """The kernel's keyword arguments (without the partition), plus each
    leg's teleport distance."""
    h, tele_distance = kernel_inputs(scenario, plans, cfg, event_mask)
    net = scenario.network
    inputs = {k: h[k] for k in (
        "free_steps", "headway", "storage", "act_offsets", "act_end_time", "act_duration",
        "leg_network", "leg_teleport_steps", "route_offsets", "route_links", "stuck_time",
        "end_time", "event_mask", "junction_delay") + TRANSIT_KEYS}
    inputs["link_from"] = net.from_node.astype(np.int64)
    inputs["link_to"] = net.to_node.astype(np.int64)
    inputs["n_nodes"] = int(net.n_nodes)
    inputs["act_owner"] = activity_owners(plans, net, h["leg_network"])
    inputs["rs_node"] = stop_nodes(scenario)
    if max(net.n_nodes, plans.n_acts) >= MAX_INDEX:
        raise ValueError(f"ParallelSim supports fewer than {MAX_INDEX} nodes and activities.")
    return inputs, tele_distance


def activity_owners(plans: PlanBatch, net: Network, leg_network: np.ndarray) -> np.ndarray:
    """Per activity, the node whose partition runs the person while it is
    there: where its next leg departs (the first route link's from-node, for
    a routed network leg), else the activity's node."""
    owner = plans.act_node.astype(np.int64).copy()
    origin = plans.leg_origin_act()
    r0, r1 = plans.route_offsets[:-1], plans.route_offsets[1:]
    routed = (np.asarray(leg_network) != 0) & (r1 > r0)
    if routed.any():
        first = plans.route_links[r0[routed]].astype(np.int64)
        owner[origin[routed]] = net.from_node[first]
    return owner


def stop_nodes(scenario: Scenario) -> np.ndarray:
    """Per route stop, the node that runs it (its waiting passengers and the
    trips there): the to-node of the stop's link, where vehicles serve it."""
    if scenario.transit is None:
        return np.zeros(0, dtype=np.int64)
    s = scenario.transit.schedule
    return scenario.network.to_node[s.stop_link[s.rs_stop]].astype(np.int64)


# ---------------------------------------------------------------- kernel


def tie_hash(t: int, source: int) -> int:
    """Tie-breaker between sources ready since the same second (splitmix64)."""
    z = (t * 0x9E3779B97F4A7C15 + source + 1) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def run_kernel(*, free_steps, headway, storage, act_offsets, act_end_time, act_duration,
               leg_network, leg_teleport_steps, route_offsets, route_links, stuck_time,
               end_time, event_mask, link_from, link_to, n_nodes, act_owner,
               network_vehicles, leg_transit, ride_offsets, ride_board, ride_alight,
               walk_before_steps, walk_after_steps, rs_stop, rs_link_pos, rs_await,
               stop_blocking, trip_k0, trip_k1, trip_start_step, trip_stop_offsets,
               trip_stop_arr_step, trip_stop_dep_step, trip_link_offsets, trip_links,
               trip_link_steps, trip_capacity, trip_pce, trip_access_steps,
               trip_egress_steps, trip_parallel, door_open_steps, door_close_steps,
               trip_network, rs_network,
               rs_node, junction_delay=0):
    """The reference kernel, on one thread. ``rust/mobslim-parallel/src/lib.rs``
    implements the same rules on partitions in parallel, with identical results.

    Arguments and results are HermesSim's (:func:`mobslim.mobsim.hermes.run_kernel`),
    plus the network's nodes (``link_from``, ``link_to``, ``n_nodes``) and
    who runs what: ``act_owner`` per activity, ``rs_node`` per route stop.
    Events are in canonical order: by second, then person steps by person
    (and trip starts, as persons ``n_plans + d``), then node steps by node.
    """
    def ints(a):
        return np.asarray(a).tolist()

    free_steps, headway, storage = ints(free_steps), ints(headway), ints(storage)
    act_offsets, act_end_time = ints(act_offsets), ints(act_end_time)
    act_duration, leg_network = ints(act_duration), ints(leg_network)
    leg_teleport_steps, route_offsets = ints(leg_teleport_steps), ints(route_offsets)
    route_links, link_from, link_to = ints(route_links), ints(link_from), ints(link_to)
    act_owner, rs_node = ints(act_owner), ints(rs_node)
    leg_transit, ride_offsets = ints(leg_transit), ints(ride_offsets)
    ride_board, ride_alight = ints(ride_board), ints(ride_alight)
    walk_before_steps, walk_after_steps = ints(walk_before_steps), ints(walk_after_steps)
    rs_stop, rs_link_pos, rs_await = ints(rs_stop), ints(rs_link_pos), ints(rs_await)
    stop_blocking, trip_k0, trip_k1 = ints(stop_blocking), ints(trip_k0), ints(trip_k1)
    trip_start_step, trip_stop_offsets = ints(trip_start_step), ints(trip_stop_offsets)
    trip_stop_arr_step, trip_stop_dep_step = ints(trip_stop_arr_step), ints(trip_stop_dep_step)
    trip_link_offsets, trip_links = ints(trip_link_offsets), ints(trip_links)
    trip_link_steps, trip_capacity = ints(trip_link_steps), ints(trip_capacity)
    trip_pce, trip_access_steps = ints(trip_pce), ints(trip_access_steps)
    trip_egress_steps, trip_parallel = ints(trip_egress_steps), ints(trip_parallel)
    finite_stuck = math.isfinite(stuck_time)
    net_veh = bool(network_vehicles)  # any trip on the network
    trip_network, rs_network = ints(trip_network), ints(rs_network)
    n_plans, n_links = len(act_offsets) - 1, len(free_steps)
    n_trips = len(trip_k0)
    n_plan_links = len(route_links)
    n_veh = n_plans + n_trips
    min_steps = [max(s, 1) for s in free_steps]
    tr_rstart = [0] * n_trips  # trip d's first route position (network mode)
    if net_veh:
        tr_rstart = [n_plan_links + o for o in trip_link_offsets[:-1]]
        route_links = route_links + trip_links

    in_links: list = [[] for _ in range(n_nodes)]
    out_links: list = [[] for _ in range(n_nodes)]
    for link in range(n_links):
        in_links[link_to[link]].append(link)
        out_links[link_from[link]].append(link)

    nan = math.nan
    t_act_start = [nan] * len(act_end_time)
    t_act_end = [nan] * len(act_end_time)
    t_dep = [nan] * len(leg_network)
    t_arr = [nan] * len(leg_network)
    t_enter = [nan] * len(route_links)

    # persons
    a_act = act_offsets[:-1]
    a_last = [o - 1 for o in act_offsets[1:]]
    a_leg = [-1] * n_plans
    a_done = [False] * n_plans
    a_ride = [0] * n_plans  # current pt ride
    a_ride_end = [0] * n_plans
    # vehicles: persons (each is also its car), then trips (network mode)
    a_rpos = [0] * n_veh
    a_rend = [0] * n_veh
    a_link = [-1] * n_veh
    v_pce = [1.0] * n_plans + trip_pce
    # links, exit side (owned by the to-node)
    l_queue = [deque() for _ in range(n_links)]  # (vehicle, next link or -1, earliest exit)
    l_nft = [0.0] * n_links  # next free flow slot (unrounded)
    # links, entry side (owned by the from-node)
    l_occ = [0.0] * n_links
    l_line = [deque() for _ in range(n_links)]  # persons waiting to depart onto the link
    l_blocked_on = [0] * n_links  # sources refused by this link, still waiting
    # sources: incoming link s < n_links, or the departure line of link s - n_links
    s_blocked: list = [None] * (2 * n_links)  # refused since
    s_target = [-1] * (2 * n_links)  # the link that refused it
    # rides
    n_rides = len(ride_board)
    t_wait = [nan] * n_rides
    t_board = [nan] * n_rides
    t_alight = [nan] * n_rides
    r_trip = [-1] * n_rides
    # route stops (owned by rs_node): waiting passengers and trips there
    waiting: list = [deque() for _ in rs_stop]
    at_stop: list = [[] for _ in rs_stop]
    # trips (owned by the node of the stop they are at, or of the link end
    # they are queued at; in messages between)
    tr_next = list(trip_k0)  # next route stop to serve
    tr_onboard: list = [[] for _ in range(n_trips)]
    tr_load = [0] * n_trips
    tr_visit = [-1] * n_trips  # route stop being served (network mode)
    tr_state = [0] * n_trips
    tr_token = [0] * n_trips
    tr_seq = [0] * n_trips  # orders a trip's tasks of one second
    tr_alight_q: list = [deque() for _ in range(n_trips)]
    tr_alight_busy = [False] * n_trips
    tr_board_busy = [False] * n_trips
    tr_boarding = [-1] * n_trips
    tr_pulled = [False] * n_trips
    t_stop_arr = [nan] * trip_stop_offsets[-1]
    t_stop_dep = [nan] * trip_stop_offsets[-1]

    rec = [bool(event_mask & (1 << k)) for k in range(N_EVENT_KINDS)]
    ev_time, ev_code = [], []
    now = 0

    def record(kind, p, ref):
        ev_time.append(now)
        ev_code.append(kind << KIND_SHIFT | p << PERSON_SHIFT | ref)

    agenda: dict[int, list] = {}  # second -> person entries (person, kind)
    wakes: dict[int, set] = {}  # second -> nodes
    tasks: dict[int, list] = {}  # node -> heap of trip tasks (t, late, trip, seq, kind, token)
    seconds: list[int] = []
    running = [-1]  # the node whose step is running (-1: the person phase)

    def at(t):
        if t not in agenda and t not in wakes:
            heapq.heappush(seconds, t)

    def schedule(t, kind, p):
        at(t)
        agenda.setdefault(t, []).append((p, kind))

    def wake(t, node):
        at(t)
        wakes.setdefault(t, set()).add(node)

    def task(t, late, d, kind, node, token=0):
        """A trip task at ``node`` (where trip ``d`` is) at second ``t``."""
        tr_seq[d] += 1
        heapq.heappush(tasks.setdefault(node, []), (t, late, d, tr_seq[d], kind, token))
        if not (t == now and running[0] == node):  # else the running step takes it
            wake(t, node)

    # messages, delivered at the end of the step
    m_enter, m_exit, m_person, m_task = [], [], [], []

    def planned_end(a, start):
        e = act_end_time[a]
        if e == e:
            return start if start > e else ceil_step(e)
        d = act_duration[a]
        if d == d:
            return max(start, ceil_step(start + d))
        return None

    # -- persons ----------------------------------------------------------
    # ``node``: the node the person is at, or None for the node running its
    # activity (in the person step)
    heap: list = []  # the running node step's sources

    def end_activity(p, node):
        a = a_act[p]
        t_act_end[a] = now
        if rec[EV_ACT_END]:
            record(EV_ACT_END, p, a)
        leg = a - p
        a_leg[p] = leg
        t_dep[leg] = now
        if rec[EV_DEPARTURE]:
            record(EV_DEPARTURE, p, leg)
        if leg_network[leg]:
            r0, r1 = route_offsets[leg], route_offsets[leg + 1]
            if r0 == r1:
                arrive(p, node)
                return
            a_rpos[p], a_rend[p] = r0, r1
            link = route_links[r0]
            line = l_line[link]
            line.append(p)
            if running[0] < 0:
                wake(now, link_from[link])
            elif len(line) == 1:  # a new source for the running node step
                push(now, n_links + link)
        elif leg_transit[leg]:
            here = act_owner[a] if node is None else node
            r0, r1 = ride_offsets[leg], ride_offsets[leg + 1]
            if r0 == r1:  # walk only
                walk_to_destination(p, here)
                return
            a_ride[p], a_ride_end[p] = r0, r1
            walk_to_stop(p, here)
        else:
            m_person.append((now + max(leg_teleport_steps[leg], 1), _TELEPORT_ARRIVE, p))

    def teleport_arrive(p):
        if rec[EV_TELEPORT]:
            record(EV_TELEPORT, p, a_leg[p])
        arrive(p, None)

    def arrive(p, node):
        leg = a_leg[p]
        t_arr[leg] = now
        if rec[EV_ARRIVAL]:
            record(EV_ARRIVAL, p, leg)
        a_leg[p] = -1
        a_link[p] = -1
        if junction_delay:  # the activity starts one second later
            m_person.append((now + 1, _ACT_START, p))
            return
        start_activity(p, node)

    def start_activity(p, node):
        a = a_act[p] = a_act[p] + 1
        t_act_start[a] = now
        if rec[EV_ACT_START]:
            record(EV_ACT_START, p, a)
        if a == a_last[p]:
            a_done[p] = True
            return
        t = planned_end(a, now)
        if t is None:
            return
        # ends now if it may (late arrival) and departs from here; another
        # node takes it over from the next second
        if t <= now and (node is None or act_owner[a] == node):
            end_activity(p, node)
        else:
            m_person.append((max(t, now + 1), _ACT_END, p))

    # -- passengers -------------------------------------------------------
    # ``here``: the node the person is at. Moves to another node, or into a
    # later second, are messages; a walk of 0 s on the spot happens at once.
    def walk_to_stop(p, here):
        r = a_ride[p]
        t = now + walk_before_steps[r]
        if t == now and rs_node[ride_board[r]] == here:
            pax_at_stop(p)
        else:
            m_person.append((max(t, now + 1), _PAX_AT_STOP, p))

    def walk_to_destination(p, here):
        t = now + walk_after_steps[a_leg[p]]
        if t == now:
            arrive(p, here)
        else:
            m_person.append((t, _PAX_ARRIVE, p))

    def pax_at_stop(p):
        r = a_ride[p]
        k = ride_board[r]
        t_wait[r] = now
        if rec[EV_WAITING]:
            record(EV_WAITING, p, k)
        trips = at_stop[k]
        if not rs_network[k]:  # vehicles on the timetable
            for d in trips:
                if tr_load[d] < trip_capacity[d]:
                    tr_load[d] += 1
                    board(d, p)
                    return
            waiting[k].append(p)
            return
        waiting[k].append(p)
        for d in trips:
            if serve_new_boarder(d):
                return

    def board(d, p):
        r = a_ride[p]
        t_board[r] = now
        r_trip[r] = d
        tr_onboard[d].append(p)
        if rec[EV_ENTER_VEH]:
            record(EV_ENTER_VEH, p, d)

    def alight(d, p, node):
        r = a_ride[p]
        t_alight[r] = now
        tr_onboard[d].remove(p)
        tr_load[d] -= 1
        if rec[EV_LEAVE_VEH]:
            record(EV_LEAVE_VEH, p, d)
        if r + 1 < a_ride_end[p]:
            a_ride[p] = r + 1
            walk_to_stop(p, node)
        else:
            walk_to_destination(p, node)

    def alighters(d, k):
        return [p for p in tr_onboard[d] if ride_alight[a_ride[p]] == k]

    # -- trips --------------------------------------------------------------
    def trip_start(d):
        if rec[EV_DRIVER_START]:
            record(EV_DRIVER_START, d, 0)
        if not trip_network[d]:
            k = trip_k0[d]
            task(max(now, trip_stop_arr_step[trip_stop_offsets[d]]), 0, d, _STOP_ARRIVE, rs_node[k])
            return
        # appears at the end of its first link, taking its storage from the
        # next second (a message to the link's from-node)
        v = n_plans + d
        rpos = a_rpos[v] = tr_rstart[d]
        a_rend[v] = rpos + trip_link_offsets[d + 1] - trip_link_offsets[d]
        link = a_link[v] = route_links[rpos]
        t_enter[rpos] = now
        if rec[EV_VEH_LINK_ENTER]:
            record(EV_VEH_LINK_ENTER, d, link)
        m_occupy.append((link, v_pce[v]))
        after = route_links[rpos + 1] if rpos + 1 < a_rend[v] else -1
        q = l_queue[link]
        q.append((v, after, now + trip_link_steps[rpos - n_plan_links] + jd(after)))
        if len(q) == 1:
            wake(max(candidate(link)[0], now), link_to[link])

    def jd(after):
        """The junction delay before ``after``, the next link (-1: none)."""
        return junction_delay if after >= 0 else 0

    def has_stop_here(d):
        k = tr_next[d]
        return k < trip_k1[d] and rs_link_pos[k] == a_rpos[n_plans + d] - tr_rstart[d]

    def record_arrival(d, k):
        t_stop_arr[trip_stop_offsets[d] + k - trip_k0[d]] = now
        if rec[EV_VEH_ARRIVE]:
            record(EV_VEH_ARRIVE, d, k)
        at_stop[k].append(d)

    def stop_depart(d, token, node):
        if token != tr_token[d]:
            return
        if trip_network[d] and tr_state[d] != _CLOSED:
            return
        k = tr_next[d]
        j = trip_stop_offsets[d] + k - trip_k0[d]
        t_stop_dep[j] = now
        if rec[EV_VEH_DEPART]:
            record(EV_VEH_DEPART, d, k)
        at_stop[k].remove(d)
        tr_next[d] = k + 1
        tr_visit[d] = -1
        tr_state[d] = 0
        if not trip_network[d]:
            if k + 1 < trip_k1[d]:  # on to the next stop: here, or a message
                t, there = max(now, trip_stop_arr_step[j + 1]), rs_node[k + 1]
                if there == node:
                    task(t, 0, d, _STOP_ARRIVE, node)
                else:
                    m_task.append((max(t, now + 1), d, _STOP_ARRIVE, there))
            return
        if has_stop_here(d):  # several stops on one link
            reach_stop(d, node)
            return
        v = n_plans + d
        rpos = a_rpos[v]
        link = route_links[rpos]
        after = route_links[rpos + 1] if rpos + 1 < a_rend[v] else -1
        # it leaves the link from now, crossing a junction one second later
        entry = (v, after, now + jd(after))
        if tr_pulled[d]:  # back from the stop into the lane, as its head
            tr_pulled[d] = False
            l_queue[link].appendleft(entry)
            unblock(link)  # the lane's head changed
        else:
            l_queue[link][0] = entry

    def sched_arrive(d, node):
        k = tr_next[d]
        record_arrival(d, k)
        for p in alighters(d, k):
            alight(d, p, node)
        q = waiting[k]
        while q and tr_load[d] < trip_capacity[d]:
            tr_load[d] += 1
            board(d, q.popleft())
        tr_token[d] += 1
        t = trip_stop_dep_step[trip_stop_offsets[d] + k - trip_k0[d]]
        task(max(now, t), 1, d, _STOP_DEPART, node, tr_token[d])

    def reach_stop(d, node):
        k = tr_next[d]
        tr_visit[d] = k
        record_arrival(d, k)
        tr_alight_q[d] = deque(alighters(d, k))
        if tr_alight_q[d] or boarders(d):
            open_doors(d, node)
        else:
            ready(d, node)

    def boarders(d):
        q = waiting[tr_visit[d]]
        return bool(q) and tr_load[d] - len(tr_alight_q[d]) < trip_capacity[d]

    def open_doors(d, node):
        tr_state[d] = _OPENING
        tr_token[d] += 1
        task(now + door_open_steps, 0, d, _DOOR_OPEN, node, tr_token[d])

    def door_open(d, token, node):
        if token != tr_token[d] or tr_state[d] != _OPENING:
            return
        tr_state[d] = _OPEN
        next_alight(d, node)
        if trip_parallel[d] or not tr_alight_busy[d]:
            next_board(d, node)
        maybe_close(d, node)

    def next_alight(d, node):
        busy = tr_alight_busy[d] = bool(tr_alight_q[d])
        if busy:
            task(now + trip_egress_steps[d], 0, d, _ALIGHT_DONE, node)

    def alight_done(d, node):
        alight(d, tr_alight_q[d].popleft(), node)
        next_alight(d, node)
        if not trip_parallel[d] and not tr_alight_busy[d] and not tr_board_busy[d]:
            next_board(d, node)
        maybe_close(d, node)

    def next_board(d, node):
        if boarders(d):
            p = waiting[tr_visit[d]].popleft()
            tr_load[d] += 1
            tr_board_busy[d] = True
            tr_boarding[d] = p
            task(now + trip_access_steps[d], 0, d, _BOARD_DONE, node)
        else:
            tr_board_busy[d] = False

    def board_done(d, node):
        p = tr_boarding[d]
        tr_boarding[d] = -1
        board(d, p)
        next_board(d, node)
        maybe_close(d, node)

    def maybe_close(d, node):
        if tr_state[d] == _OPEN and not tr_alight_busy[d] and not tr_board_busy[d]:
            tr_state[d] = _CLOSING
            task(now + door_close_steps, 0, d, _DOOR_CLOSE, node)

    def door_close(d, node):
        if tr_state[d] == _CLOSING:
            ready(d, node)

    def ready(d, node):
        """Doors closed and nobody to serve: depart (or await the departure)."""
        if boarders(d):
            open_doors(d, node)
            return
        tr_state[d] = _CLOSED
        k = tr_visit[d]
        t = now
        if rs_await[k]:
            t = max(now, trip_stop_dep_step[trip_stop_offsets[d] + k - trip_k0[d]])
        tr_token[d] += 1
        task(t, 1, d, _STOP_DEPART, node, tr_token[d])

    def serve_new_boarder(d):
        """A passenger joined the line at trip ``d``'s stop. ``True`` if it takes them."""
        if not boarders(d):
            return False
        node = rs_node[tr_visit[d]]
        state = tr_state[d]
        if state == _OPEN:
            if not tr_board_busy[d] and (trip_parallel[d] or not tr_alight_busy[d]):
                next_board(d, node)
            return True
        if state == _CLOSED:
            open_doors(d, node)
            return True
        return state == _OPENING or state == _CLOSING

    def run_tasks(node):
        """The node's trip tasks due now, in order (ordinary, then late; by trip)."""
        h = tasks.get(node)
        while h and h[0][0] == now:
            _, _, d, _, kind, token = heapq.heappop(h)
            if kind == _STOP_ARRIVE:
                sched_arrive(d, node)
            elif kind == _STOP_DEPART:
                stop_depart(d, token, node)
            elif kind == _DOOR_OPEN:
                door_open(d, token, node)
            elif kind == _ALIGHT_DONE:
                alight_done(d, node)
            elif kind == _BOARD_DONE:
                board_done(d, node)
            else:
                door_close(d, node)

    def tasks_due(node):
        h = tasks.get(node)
        return bool(h) and h[0][0] == now

    # -- nodes ------------------------------------------------------------
    def candidate(s):
        """``(ready, vehicle)`` of source ``s``'s head, or None if it has none
        (or it is a trip serving a stop)."""
        if s < n_links:
            q = l_queue[s]
            if not q:
                return None
            v, after, earliest = q[0]
            if v >= n_plans:
                d = v - n_plans
                if tr_visit[d] >= 0:
                    return None
                if has_stop_here(d):  # it serves the stop on reaching the link's end
                    return earliest - jd(after), v
            return max(earliest, ceil_step(l_nft[s])), v
        line = l_line[s - n_links]
        if not line:
            return None
        v = line[0]
        return int(t_dep[a_leg[v]]), v

    def push(ready, s):
        # a source refused before ranks by how long it has waited since
        since = s_blocked[s]
        heapq.heappush(heap, (ready if since is None else since, tie_hash(now, s), s))

    def unblock(s):
        if s_blocked[s] is not None:
            s_blocked[s] = None
            l_blocked_on[s_target[s]] -= 1

    def step_node(node):
        running[0] = node
        while True:
            run_tasks(node)
            move_node(node)
            if not tasks_due(node):
                break
        running[0] = -1

    def move_node(node):
        heap.clear()
        for s in in_links[node]:
            c = candidate(s)
            if c is not None and c[0] <= now:
                push(c[0], s)
        for link in out_links[node]:
            s = n_links + link
            c = candidate(s)
            if c is not None:  # departures are ready since they departed
                push(c[0], s)
        while heap:
            _, _, s = heapq.heappop(heap)
            ready, v = candidate(s)
            if s < n_links:
                _, nxt, earliest = l_queue[s][0]
                if v >= n_plans and has_stop_here(v - n_plans):  # serve the stop
                    d = v - n_plans
                    if not stop_blocking[rs_stop[tr_next[d]]]:
                        l_queue[s].popleft()  # into the stop: off the lane, still in storage
                        tr_pulled[d] = True
                    reach_stop(d, node)
                    c = candidate(s)
                    if c is not None and c[0] <= now:
                        push(c[0], s)
                    continue
            else:  # a departure: onto the first link of its route
                nxt, earliest = route_links[a_rpos[v]], 0
            pce = v_pce[v]
            if nxt >= 0:  # needs space on the next link
                since = s_blocked[s]
                forced = finite_stuck and since is not None and now >= ceil_step(since + stuck_time)
                if not forced and not l_occ[nxt] < storage[nxt]:
                    if since is None:
                        s_blocked[s] = now
                        s_target[s] = nxt
                        l_blocked_on[nxt] += 1
                        if finite_stuck:
                            wake(max(ceil_step(now + stuck_time), now + 1), node)
                    continue  # FIFO: nothing behind it moves this step
                unblock(s)
            if s < n_links:  # leave link s
                l_queue[s].popleft()
                nft = l_nft[s]
                base = max(earliest, nft) if now == ready else now
                l_nft[s] = base + headway[s] * pce
                if v < n_plans:
                    if rec[EV_LINK_LEAVE]:
                        record(EV_LINK_LEAVE, v, s)
                elif rec[EV_VEH_LINK_LEAVE]:
                    record(EV_VEH_LINK_LEAVE, v - n_plans, s)
                m_exit.append((s, pce))
                if nxt < 0:
                    if v < n_plans:
                        arrive(v, node)
                    else:  # a trip leaves the network
                        a_link[v] = -1
                else:
                    a_rpos[v] += 1
            else:
                l_line[s - n_links].popleft()
            if nxt >= 0:  # enter link nxt
                l_occ[nxt] += pce
                rpos = a_rpos[v]
                t_enter[rpos] = now
                a_link[v] = nxt
                after = route_links[rpos + 1] if rpos + 1 < a_rend[v] else -1
                if v < n_plans:
                    if rec[EV_LINK_ENTER]:
                        record(EV_LINK_ENTER, v, nxt)
                    steps = min_steps[nxt]
                else:
                    if rec[EV_VEH_LINK_ENTER]:
                        record(EV_VEH_LINK_ENTER, v - n_plans, nxt)
                    steps = max(trip_link_steps[rpos - n_plan_links], 1)
                # a junction delay when it crosses into a next link
                m_enter.append((nxt, v, after, now + steps + jd(after)))
            c = candidate(s)
            if c is not None and c[0] <= now:
                push(c[0], s)
        for s in in_links[node]:  # wake for heads that become ready later
            if s_blocked[s] is None:
                c = candidate(s)
                if c is not None:
                    wake(max(c[0], now + 1), node)

    m_occupy = []  # a trip appeared on a link: (link, pce) of storage taken

    def deliver():
        for nxt, v, after, earliest in m_enter:
            q = l_queue[nxt]
            q.append((v, after, earliest))
            if len(q) == 1:
                wake(max(candidate(nxt)[0], now + 1), link_to[nxt])
        for link, pce in m_exit:
            l_occ[link] -= pce
            if l_blocked_on[link]:
                wake(now + 1, link_from[link])
        for link, pce in m_occupy:
            l_occ[link] += pce
        for t, kind, p in m_person:
            schedule(t, kind, p)
        for t, d, kind, node in m_task:
            task(t, 0, d, kind, node)
        m_enter.clear()
        m_exit.clear()
        m_occupy.clear()
        m_person.clear()
        m_task.clear()

    # -- main loop --------------------------------------------------------
    for d in range(n_trips):
        schedule(max(0, trip_start_step[d]), _TRIP_START, n_plans + d)
    for p in range(n_plans):
        if a_act[p] == a_last[p]:
            a_done[p] = True
            continue
        t = planned_end(a_act[p], 0)
        if t is not None:
            schedule(t, _ACT_END, p)
    last = 0
    while seconds:
        t = heapq.heappop(seconds)
        if t not in agenda and t not in wakes:
            continue  # already run (a wake for the current second re-pushed it)
        if t > end_time:
            break
        now = last = t
        for p, kind in sorted(agenda.pop(t, ())):
            if kind == _ACT_END:
                end_activity(p, None)
            elif kind == _TELEPORT_ARRIVE:
                teleport_arrive(p)
            elif kind == _ACT_START:
                start_activity(p, None)
            elif kind == _PAX_AT_STOP:
                pax_at_stop(p)
            elif kind == _PAX_ARRIVE:
                arrive(p, None)
            else:
                trip_start(p - n_plans)
        for node in sorted(wakes.pop(t, ())):
            step_node(node)
        deliver()

    last_time = float(last)
    if math.isfinite(end_time):
        last_time = max(last_time, end_time)
    stuck = [not d for d in a_done]
    stuck_leg = [a_leg[p] for p in range(n_plans) if stuck[p]]
    stuck_link = [a_link[p] for p in range(n_plans) if stuck[p]]
    f8, i8 = np.float64, np.int64
    return (np.array(t_act_start, f8), np.array(t_act_end, f8), np.array(t_dep, f8),
            np.array(t_arr, f8), np.array(t_enter, f8), np.array(stuck, dtype=bool),
            np.array(ev_time, f8), np.array(ev_code, i8), np.array(stuck_leg, i8),
            np.array(stuck_link, i8), last_time, np.array(t_wait, f8), np.array(t_board, f8),
            np.array(t_alight, f8), np.array(r_trip, i8), np.array(t_stop_arr, f8),
            np.array(t_stop_dep, f8))
