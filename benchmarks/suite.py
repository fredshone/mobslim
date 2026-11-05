"""The benchmarks.

Each benchmark isolates one stage of the pipeline, so a design change can be
measured where it happens:

* ``load.*``       reading MATSim XML into the data model
* ``init.*``       one-off preparation before the first iteration
* ``mobsim.*``     the queue simulation, with and without the handlers the
                   controller attaches
* ``events.*``     replaying a recorded event stream through handlers
* ``trajectories.*`` deriving the iteration's outputs from the mobsim's arrays
* ``scoring.*``    scoring experienced plans / trajectories
* ``routing.*``    least-cost path search
* ``transit.*``    transit routing (RAPTOR) and the transit mobsims, on
                   scenarios with a schedule (``-s pt_tutorial``, ``-s siouxfalls``)
* ``replanning.*`` one replanning step over the whole population
* ``run.*``        end to end, as a user would run it

Inputs come from a :class:`Workload`, which builds each intermediate value
once (untimed) and caches it, so benchmarks only time their own stage.
"""

import logging
import time
from collections import defaultdict
from functools import cached_property
from random import Random

import numpy as np

from benchmarks.harness import Benchmark, SkipBenchmark
from benchmarks.scenarios import ScenarioSpec
from mobslim import (
    AgentSim,
    AgentSimParams,
    CharyparNagelScorer,
    Controller,
    ControllerParams,
    HermesSim,
    HermesSimParams,
    QSim,
    Scenario,
    route_scenario,
)
from mobslim.events import EventLog, EventsManager
from mobslim.experience import ExperiencedPlanCollector
from mobslim.listeners import ControllerListener
from mobslim.metrics import iteration_metrics
from mobslim.replanning import (
    ReplanningContext,
    Reroute,
    SerialRunner,
    StrategyManager,
    VectorRunner,
)
from mobslim.replanning.modules import (
    clear_routes_batch,
    route_batch,
    route_plans,
)
from mobslim.routing import (
    DijkstraRouter,
    RustRouter,
    TravelTimeCollector,
    TravelTimeTable,
    rust_router_available,
)
from mobslim.scoring import apply_scores
from mobslim.trajectories import Trajectories, TrajectoryCollector

SEED = 4711


class Workload:
    """Lazily built, cached inputs for one scenario.

    The chain follows one controller iteration: load, route at free speed,
    simulate, collect, score. Later benchmarks start from these values.
    """

    def __init__(self, spec: ScenarioSpec):
        self.spec = spec

    @cached_property
    def network(self):
        return self.spec.load_network()

    @cached_property
    def population(self):
        return self.spec.load_population(self.network)

    @cached_property
    def transit(self):
        load = self.spec.load_transit
        return None if load is None else load(self.network)

    @cached_property
    def scenario(self) -> Scenario:
        return Scenario(self.network, self.population, transit=self.transit)

    @cached_property
    def routed(self) -> Scenario:
        return route_scenario(self.scenario)

    @cached_property
    def events(self) -> list:
        log = EventLog()
        QSim().run(self.routed, EventsManager([log]))
        return log.events

    @cached_property
    def trajectories(self) -> Trajectories:
        return QSim().run(self.routed, EventsManager())

    @cached_property
    def experienced(self) -> list:
        return self.trajectories.experienced_plans()

    @cached_property
    def travel_time(self) -> TravelTimeTable:
        """Congested travel times learned from the first iteration."""
        return self.trajectories.travel_time_table(self.network)

    @cached_property
    def scores(self) -> list[float]:
        scorer = CharyparNagelScorer()
        return [scorer.score(p) for p in self.experienced]

    @cached_property
    def scored(self) -> Scenario:
        """The scenario after one iteration's scoring: ready to replan."""
        population = apply_scores(self.routed.population, self.scores)
        return self.routed.with_population(population)

    def context(self, iteration: int = 1) -> ReplanningContext:
        return ReplanningContext.from_scenario(self.scored, self.travel_time, iteration)

    def trips(self) -> list[tuple[int, int, float]]:
        """(origin, destination, departure) for every network leg of every
        selected plan, departing when the previous activity ends."""
        out, scenario = [], self.routed
        for plan in scenario.population.selected_plans():
            t = 0.0
            for i, leg, o, d in plan.trips():
                t = plan.elements[i - 1].planned_end(t) or t
                if scenario.modes.is_network_mode(leg.mode):
                    out.append((o, d, t))
                if leg.route is not None:
                    t += leg.route.travel_time
        return out

    def pt_trips(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(origin_xy, destination_xy, departure)`` of every transit leg of
        every selected plan, departing when the previous activity ends."""
        if self.transit is None:
            raise SkipBenchmark(f"scenario {self.spec.name!r} has no transit")
        scenario = self.routed
        o, d, t = [], [], []
        for plan in scenario.population.selected_plans():
            time = 0.0
            for i, leg, _, _ in plan.trips():
                prev, nxt = plan.elements[i - 1], plan.elements[i + 1]
                time = prev.planned_end(time) or time
                if scenario.modes.is_transit_mode(leg.mode):
                    o.append(prev.xy(scenario.network))
                    d.append(nxt.xy(scenario.network))
                    t.append(time)
                if leg.route is not None:
                    time += leg.route.travel_time
        if not o:
            raise SkipBenchmark(f"scenario {self.spec.name!r} has no transit legs")
        return np.array(o, dtype=float), np.array(d, dtype=float), np.array(t, dtype=float)

    def with_vehicles(self, vehicles: str) -> Scenario:
        """The routed scenario with transit vehicles in ``vehicles`` mode."""
        from dataclasses import replace

        if self.transit is None:
            raise SkipBenchmark(f"scenario {self.spec.name!r} has no transit")
        transit = self.routed.transit
        return replace(self.routed, transit=transit.with_params(
            replace(transit.params, vehicles=vehicles)))


def replay(events, *handlers) -> None:
    manager = EventsManager(list(handlers))
    manager.reset(0)
    manager.process_many(events)
    manager.finish()


# ------------------------------------------------------------- registry

BENCHMARKS: dict[str, Benchmark] = {}


def benchmark(name: str, info=None):
    """Register ``setup(workload) -> callable`` as a benchmark."""

    def wrap(setup):
        BENCHMARKS[name] = Benchmark(name, (setup.__doc__ or "").strip(), setup, info)
        return setup

    return wrap


# ----------------------------------------------------------------- load


@benchmark("load.network", info=lambda net: {"nodes": net.n_nodes, "links": net.n_links})
def _load_network(w: Workload):
    """Read the network XML."""
    return w.spec.load_network


@benchmark("load.population", info=lambda pop: {"persons": len(pop)})
def _load_population(w: Workload):
    """Read the population XML (needs a loaded network)."""
    network = w.network
    return lambda: w.spec.load_population(network)


# ----------------------------------------------------------------- init


@benchmark("init.free_speed_table")
def _free_speed_table(w: Workload):
    """Build the free-speed TravelTimeTable used for the first routing."""
    network = w.network
    return lambda: TravelTimeTable.free_speed(network)


def _route_info(scenario):
    routes = [
        leg.route
        for plan in scenario.population.selected_plans()
        for leg in plan.legs
        if leg.route is not None
    ]
    return {"routes": len(routes), "distance_m": round(sum(r.distance for r in routes), 1)}


@benchmark("init.route_scenario", info=_route_info)
def _route_scenario(w: Workload):
    """Validate modes and route every leg at free speed (Controller.prepare)."""
    scenario = w.scenario
    return lambda: route_scenario(scenario)


# --------------------------------------------------------------- mobsim


def _events_info(log):
    events = log.events
    return {"events": len(events), "last_time": round(events[-1].time, 3) if events else 0.0}


def _trajectories_info(tr):
    entered = int((tr.link_enter == tr.link_enter).sum())
    last = max(float(_nanmax(tr.act_start)), float(_nanmax(tr.leg_arrival)))
    return {"link_traversals": entered, "last_time": round(last, 3)}


def _nanmax(a):
    return np.nanmax(a) if len(a) else 0.0


@benchmark("mobsim.qsim_no_events", info=_trajectories_info)
def _qsim_no_events(w: Workload):
    """One QSim run with no event handlers: trajectories only (the default
    in the controller), so no event objects are built."""
    scenario, mobsim = w.routed, QSim()
    return lambda: mobsim.run(scenario, EventsManager())


@benchmark("mobsim.qsim", info=_events_info)
def _qsim(w: Workload):
    """One QSim run, events go to a single list (minimal dispatch cost)."""
    scenario, mobsim = w.routed, QSim()

    def run():
        log = EventLog()
        mobsim.run(scenario, EventsManager([log]))
        return log

    return run


@benchmark("mobsim.qsim_with_collectors")
def _qsim_with_collectors(w: Workload):
    """The iteration's whole mobsim stage as the controller runs it: one QSim
    run, then travel times and experienced plans from the trajectories."""
    scenario, mobsim = w.routed, QSim()

    def run():
        tr = mobsim.run(scenario, EventsManager())
        return tr.travel_time_table(scenario.network), tr.experienced_plans()

    return run


@benchmark("mobsim.agentsim_no_events", info=_trajectories_info)
def _agentsim_no_events(w: Workload):
    """One AgentSim run with no event handlers (compare with qsim_no_events)."""
    scenario, mobsim = w.routed, AgentSim()
    return lambda: mobsim.run(scenario, EventsManager())


# heavy spillback: links hold 0.2% of their vehicles and pass 1% of their flow
_CONGESTED = dict(storage_capacity_factor=0.002, flow_capacity_factor=0.01, stuck_time=120.0)


@benchmark("mobsim.agentsim_congested_poll", info=_trajectories_info)
def _agentsim_congested_poll(w: Workload):
    """AgentSim under heavy spillback, blocked agents polling every retry tick."""
    scenario, mobsim = w.routed, AgentSim(AgentSimParams(**_CONGESTED))
    return lambda: mobsim.run(scenario, EventsManager())


@benchmark("mobsim.agentsim_congested_chain", info=_trajectories_info)
def _agentsim_congested_chain(w: Workload):
    """The same run with blocked="chain" (identical results, fewer retries)."""
    scenario, mobsim = w.routed, AgentSim(AgentSimParams(blocked="chain", **_CONGESTED))
    return lambda: mobsim.run(scenario, EventsManager())


@benchmark("mobsim.agentsim", info=_events_info)
def _agentsim(w: Workload):
    """One AgentSim run, events go to a single list (compare with qsim)."""
    scenario, mobsim = w.routed, AgentSim()

    def run():
        log = EventLog()
        mobsim.run(scenario, EventsManager([log]))
        return log

    return run


@benchmark("mobsim.hermes_no_events", info=_trajectories_info)
def _hermes_no_events(w: Workload):
    """One HermesSim run (compiled kernel if built) with no event handlers."""
    scenario, mobsim = w.routed, HermesSim()
    return lambda: mobsim.run(scenario, EventsManager())


@benchmark("mobsim.hermes", info=_events_info)
def _hermes(w: Workload):
    """One HermesSim run, events go to a single list (compare with qsim)."""
    scenario, mobsim = w.routed, HermesSim()

    def run():
        log = EventLog()
        mobsim.run(scenario, EventsManager([log]))
        return log

    return run


@benchmark("mobsim.hermes_congested", info=_trajectories_info)
def _hermes_congested(w: Workload):
    """HermesSim under the same heavy spillback as agentsim_congested_*."""
    scenario, mobsim = w.routed, HermesSim(HermesSimParams(**_CONGESTED))
    return lambda: mobsim.run(scenario, EventsManager())


@benchmark("mobsim.hermes_py_no_events", info=_trajectories_info)
def _hermes_py_no_events(w: Workload):
    """HermesSim's Python reference kernel, no event handlers."""
    scenario, mobsim = w.routed, HermesSim(HermesSimParams(backend="python"))
    return lambda: mobsim.run(scenario, EventsManager())


# --------------------------------------------------------------- events


@benchmark("events.travel_times")
def _events_travel_times(w: Workload):
    """Replay recorded events into a TravelTimeCollector and build the table."""
    events, network = w.events, w.network

    def run():
        collector = TravelTimeCollector(network)
        replay(events, collector)
        return collector.table()

    return run


@benchmark("events.experienced_plans", info=lambda plans: {
    "plans": len(plans), "stuck": sum(p.stuck for p in plans)})
def _events_experienced(w: Workload):
    """Replay recorded events into an ExperiencedPlanCollector."""
    events, network, population = w.events, w.network, w.routed.population

    def run():
        collector = ExperiencedPlanCollector(network, population)
        replay(events, collector)
        return collector.plans()

    return run


@benchmark("events.trajectories")
def _events_trajectories(w: Workload):
    """Replay recorded events into a TrajectoryCollector (the path for
    mobsims that only emit events)."""
    events, network = w.events, w.network
    plans = w.trajectories.plans

    def run():
        collector = TrajectoryCollector(network, plans)
        replay(events, collector)
        return collector.trajectories()

    return run


# --------------------------------------------------------- trajectories
@benchmark("trajectories.travel_times")
def _trajectories_travel_times(w: Workload):
    """Travel-time table from the trajectory arrays (numpy)."""
    tr, network = w.trajectories, w.network
    return lambda: tr.travel_time_table(network)


@benchmark("trajectories.experienced_plans")
def _trajectories_experienced(w: Workload):
    """Build ExperiencedPlan objects from the trajectory arrays."""
    tr = w.trajectories
    return lambda: Trajectories(tr.plans, tr.act_start, tr.act_end, tr.leg_departure,
                                tr.leg_arrival, tr.leg_distance, tr.link_enter,
                                tr.stuck).experienced_plans()


# -------------------------------------------------------------- scoring


@benchmark("scoring.charypar_nagel", info=lambda s: {"mean_score": round(sum(s) / len(s), 6)})
def _scoring(w: Workload):
    """Score every experienced plan."""
    plans, scorer = w.experienced, CharyparNagelScorer()
    return lambda: [scorer.score(p) for p in plans]


@benchmark("scoring.charypar_nagel_vectorised",
           info=lambda s: {"mean_score": round(float(s.mean()), 6)})
def _scoring_vectorised(w: Workload):
    """Score every person at once from the trajectory arrays (numpy)."""
    tr, scorer = w.trajectories, CharyparNagelScorer()
    return lambda: scorer.score_trajectories(tr)


def _scoring_attributes(n: int):
    """Synthetic person attributes: 16 subpopulations, incomes, car constants."""
    from mobslim.attributes import PersonAttributes

    rng = np.random.default_rng(0)
    return PersonAttributes(n, {"subpopulation": [f"s{i % 16}" for i in range(n)],
                                "income": rng.uniform(1000, 5000, n),
                                "asc_car": rng.normal(0, 1, n)})


@benchmark("scoring.vectorised_subpops16",
           info=lambda s: {"mean_score": round(float(s.mean()), 6)})
def _scoring_subpops(w: Workload):
    """Vectorised scoring with 16 subpopulations, spread over the persons."""
    from mobslim.scoring import ScoringParams

    tr = w.trajectories
    subs = {f"s{i}": ScoringParams(performing=4.0 + i / 4) for i in range(16)}
    scorer = CharyparNagelScorer(subpopulations=subs)
    attrs = _scoring_attributes(tr.plans.n_plans)
    return lambda: scorer.score_trajectories(tr, attrs)


@benchmark("scoring.person_specific",
           info=lambda s: {"mean_score": round(float(s.mean()), 6)})
def _scoring_person_specific(w: Workload):
    """Vectorised scoring with income-dependent money and per-person car
    constants (rules resolved once per run, as in the controller)."""
    from mobslim.scoring import (
        IncomeDependentMoney,
        ModeConstants,
        ModeParams,
        ScoringParams,
    )

    tr = w.trajectories
    params = ScoringParams(modes={"car": ModeParams(monetary_distance_rate=-1e-4)})
    scorer = CharyparNagelScorer(params, person_rules=[IncomeDependentMoney(), ModeConstants()])
    attrs = _scoring_attributes(tr.plans.n_plans).without("subpopulation")
    return lambda: scorer.score_trajectories(tr, attrs)


@benchmark("scoring.apply_and_metrics")
def _apply_and_metrics(w: Workload):
    """Write scores onto plans and compute the iteration metrics row."""
    population, scores, tr = w.routed.population, w.scores, w.trajectories

    def run():
        scored = apply_scores(population, scores)
        return iteration_metrics(0, scored, tr)

    return run


# -------------------------------------------------------------- routing


def _routes_info(routes):
    return {"routes": len(routes), "travel_time_s": round(sum(r.travel_time for r in routes), 1)}


def _rust_router(w: Workload, **kwargs):
    if not rust_router_available():
        raise SkipBenchmark("mobslim_router (Rust router) not built")
    return RustRouter(w.network, "car", w.travel_time, **kwargs)


@benchmark("routing.dijkstra", info=_routes_info)
def _dijkstra(w: Workload):
    """Route every network trip of every selected plan on congested travel
    times (time-dependent Dijkstra in Python, one query per trip)."""
    trips = w.trips()
    router = DijkstraRouter(w.network, "car", w.travel_time)
    return lambda: [router.route(o, d, t) for o, d, t in trips]


@benchmark("routing.dijkstra_many", info=_routes_info)
def _dijkstra_many(w: Workload):
    """The same trips with DijkstraRouter.route_many (one search per shared
    origin and departure time; same routes)."""
    o, d, t = zip(*w.trips())
    router = DijkstraRouter(w.network, "car", w.travel_time)
    return lambda: router.route_many(o, d, t)


@benchmark("routing.rust_1thread", info=_routes_info)
def _rust_1thread(w: Workload):
    """The same trips with RustRouter.route_many on one thread (same routes)."""
    o, d, t = zip(*w.trips())
    router = _rust_router(w, threads=1)
    return lambda: router.route_many(o, d, t)


@benchmark("routing.rust", info=_routes_info)
def _rust(w: Workload):
    """The same trips with RustRouter.route_many on every CPU (same routes)."""
    o, d, t = zip(*w.trips())
    router = _rust_router(w)
    return lambda: router.route_many(o, d, t)


@benchmark("routing.rust_alt_1thread", info=_routes_info)
def _rust_alt(w: Workload):
    """The same trips with RustRouter's ALT (A*, 16 landmarks) on one thread:
    same travel times, possibly other routes where several tie."""
    o, d, t = zip(*w.trips())
    router = _rust_router(w, algorithm="alt", threads=1)
    return lambda: router.route_many(o, d, t)


# -------------------------------------------------------------- transit


def _pt_routes_info(routes):
    return {"routes": len(routes), "with_rides": sum(r.n_rides > 0 for r in routes),
            "travel_time_s": round(sum(r.travel_time for r in routes), 1)}


@benchmark("transit.raptor", info=_pt_routes_info)
def _raptor(w: Workload):
    """Route every transit leg of every selected plan with the Python RAPTOR
    (RaptorRouter.route_many_xy: one search per shared origin and time)."""
    from mobslim.transit import RaptorRouter

    o, d, t = w.pt_trips()
    router = RaptorRouter(w.network, w.transit)
    o, d, t = [tuple(p) for p in o.tolist()], [tuple(p) for p in d.tolist()], t.tolist()
    return lambda: router.route_many_xy(o, d, t)


def _rust_raptor(w: Workload, threads=None):
    from mobslim.transit import RustRaptorRouter, rust_raptor_available

    if not rust_raptor_available():
        raise SkipBenchmark("mobslim_router (Rust router) not built")
    o, d, t = w.pt_trips()
    router = RustRaptorRouter(w.network, w.transit, threads=threads)
    return lambda: router.route_many_xy(o, d, t)


@benchmark("transit.raptor_rust_1thread", info=_pt_routes_info)
def _raptor_rust_1thread(w: Workload):
    """The same legs with RustRaptorRouter on one thread (same routes)."""
    return _rust_raptor(w, threads=1)


@benchmark("transit.raptor_rust", info=_pt_routes_info)
def _raptor_rust(w: Workload):
    """The same legs with RustRaptorRouter on every CPU (same routes)."""
    return _rust_raptor(w)


def _pt_mobsim_info(tr):
    return {"rides": int(np.isfinite(tr.ride_board).sum()), "stuck": int(tr.stuck.sum()),
            "max_load": int(tr.pt_load().max()) if len(tr.ride_board) else 0}


def _pt_mobsim(w: Workload, vehicles: str, mobsim):
    scenario = w.with_vehicles(vehicles)
    return lambda: mobsim.run(scenario, EventsManager())


@benchmark("transit.qsim_schedule", info=_pt_mobsim_info)
def _qsim_schedule(w: Workload):
    """One QSim run, transit vehicles on the timetable, no event handlers."""
    return _pt_mobsim(w, "schedule", QSim())


@benchmark("transit.qsim_network", info=_pt_mobsim_info)
def _qsim_network(w: Workload):
    """One QSim run, transit vehicles in the queue model, no event handlers."""
    return _pt_mobsim(w, "network", QSim())


@benchmark("transit.agentsim_schedule", info=_pt_mobsim_info)
def _agentsim_schedule(w: Workload):
    """One AgentSim run, transit vehicles on the timetable, no event handlers."""
    return _pt_mobsim(w, "schedule", AgentSim())


@benchmark("transit.agentsim_network", info=_pt_mobsim_info)
def _agentsim_network(w: Workload):
    """One AgentSim run, transit vehicles in the queue model, no event handlers."""
    return _pt_mobsim(w, "network", AgentSim())


@benchmark("transit.hermes_schedule", info=_pt_mobsim_info)
def _hermes_schedule(w: Workload):
    """One HermesSim run (Rust kernel if built), timetable vehicles."""
    return _pt_mobsim(w, "schedule", HermesSim())


@benchmark("transit.hermes_network", info=_pt_mobsim_info)
def _hermes_network(w: Workload):
    """One HermesSim run (Rust kernel if built), queue-model vehicles."""
    return _pt_mobsim(w, "network", HermesSim())


@benchmark("transit.hermes_py_network", info=_pt_mobsim_info)
def _hermes_py_network(w: Workload):
    """HermesSim's Python reference kernel, queue-model vehicles."""
    return _pt_mobsim(w, "network", HermesSim(HermesSimParams(backend="python")))


# ----------------------------------------------------------- replanning


def _replan_info(population):
    return {"plans": sum(len(p.plans) for p in population)}


@benchmark("replanning.default_strategies", info=_replan_info)
def _replan_default(w: Workload):
    """One replanning step with the MATSim-like default strategies
    (80% ChangeExpBeta, 10% Reroute, 10% TimeAllocationMutator+Reroute)."""
    population, manager, runner = w.scored.population, StrategyManager.default(), SerialRunner()
    ctx = w.context()
    return lambda: runner.run(population, manager, ctx, SEED)


@benchmark("replanning.default_strategies_vectorised", info=_replan_info)
def _replan_default_vectorised(w: Workload):
    """The same step with the VectorRunner (the controller's default): the
    same rules on the population's tables, numpy random numbers."""
    population, manager, runner = w.scored.population, StrategyManager.default(), VectorRunner()
    ctx = w.context()
    return lambda: runner.run(population, manager, ctx, SEED)


@benchmark("replanning.reroute_all")
def _reroute_all(w: Workload):
    """Reroute every selected plan, one plan at a time, with the Python
    router: the worst case for routing in replanning."""
    plans = list(w.scored.population.selected_plans())
    ctx = ReplanningContext.from_scenario(w.scored, w.travel_time, 1, DijkstraRouter)
    module, rng = Reroute(), Random(SEED)
    return lambda: [module.mutate(p, ctx, rng) for p in plans]


@benchmark("replanning.reroute_all_batched")
def _reroute_all_batched(w: Workload):
    """The same, batched as the replanning runners do it (clear routes, then
    route_plans), with the default router (Rust if built). Same routes."""
    plans = list(w.scored.population.selected_plans())
    batched, module = w.context().batched(), Reroute()

    def run():
        ctx = w.context()  # fresh routers, as each iteration gets
        return route_plans([module.mutate(p, batched, None) for p in plans], ctx)

    return run


@benchmark("replanning.reroute_all_vectorised")
def _reroute_all_vectorised(w: Workload):
    """Every selected plan rerouted on columns (clear_routes_batch, then
    route_batch), as the VectorRunner does it. Same routes."""
    batch = w.scored.population.selected_batch()
    ctx = ReplanningContext.from_scenario(w.scored, w.travel_time, 1)
    return lambda: route_batch(clear_routes_batch(batch, ctx, keep_network=False), ctx)


# ------------------------------------------------------------------ run


def _run_info(result):
    last = result.metrics[-1]
    return {
        "iterations": len(result.metrics),
        "score_executed": round(last["score_executed"], 6),
        "score_best": round(last["score_best"], 6),
        "stuck": last["stuck"],
    }


def _controller(scenario: Scenario, iterations: int) -> Controller:
    logging.getLogger("mobslim").setLevel(logging.WARNING)
    return Controller(scenario, params=ControllerParams(iterations=iterations, seed=SEED))


@benchmark("run.controller", info=_run_info)
def _run_controller(w: Workload):
    """Controller.run for the scenario's iterations, starting from loaded data
    (prepare + iterations; no file reading)."""
    scenario, n = w.scenario, w.spec.iterations
    return lambda: _controller(scenario, n).run()


@benchmark("run.end_to_end", info=_run_info)
def _run_end_to_end(w: Workload):
    """What a user runs: load XML, build the scenario, run all iterations."""
    spec = w.spec

    def run():
        network = spec.load_network()
        transit = None if spec.load_transit is None else spec.load_transit(network)
        scenario = Scenario(network, spec.load_population(network), transit=transit)
        return _controller(scenario, spec.iterations).run()

    return run


# ------------------------------------------------------ stage breakdown


class StageTimer(ControllerListener):
    """Splits a Controller.run into stages using the listener hooks.

    Each hook closes the stage that ran since the previous hook, so the
    stages add up to the whole run.
    """

    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self._t = time.perf_counter()

    def _lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.totals[stage] += now - self._t
        self._t = now

    def on_startup(self, controller):
        self._lap("prepare (route at free speed)")

    def on_iteration_start(self, state):
        self._lap("other")

    def on_before_mobsim(self, state):
        self._lap("replanning")

    def on_after_mobsim(self, state):
        self._lap("mobsim + event handlers")

    def on_scoring(self, state):
        self._lap("scoring")

    def on_iteration_end(self, state):
        self._lap("travel times + metrics")

    def on_shutdown(self, controller):
        self._lap("other")


def stage_breakdown(w: Workload, repeats: int = 3) -> dict[str, float]:
    """Seconds per stage of ``run.controller`` (best of ``repeats``)."""
    best: dict[str, float] = {}
    for _ in range(repeats):
        controller = _controller(w.scenario, w.spec.iterations)
        timer = controller.add_listener(StageTimer())
        controller.run()
        if not best or sum(timer.totals.values()) < sum(best.values()):
            best = dict(timer.totals)
    return best
