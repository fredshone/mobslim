"""HermesSim-specific behaviour.

The shared queue-model spec runs against HermesSim (both backends) in
``test_qsim.py`` and ``test_trajectories.py``. This module covers what is
specific to it:

* the Rust kernel and the Python reference kernel agree exactly, on
  trajectories and on events, across congestion levels, stuck settings,
  cut-off runs, teleports and far-future times;
* with whole-second parameters HermesSim reproduces the QSim exactly, even
  under spillback;
* how times are discretised (free-speed rounding, activity ends, teleports);
* with fractional parameters it stays close to the QSim;
* the controller learns with it;
* backend selection.
"""

import math
from dataclasses import astuple
from random import Random

import numpy as np
import pytest

from mobslim import (
    Controller,
    ControllerConfig,
    HermesSim,
    HermesSimConfig,
    NetworkBuilder,
    Plan,
    Population,
    QSim,
    QSimConfig,
    Scenario,
    grid_network,
    route_scenario,
)
from mobslim.events import (
    ActivityEnd,
    EventLog,
    EventsManager,
    LinkEnter,
    LinkLeave,
    PersonArrival,
)
from mobslim.mobsim import hermes
from tests.conftest import chain_network, commuters, routed

ARRAYS = ("act_start", "act_end", "leg_departure", "leg_arrival", "leg_distance",
          "link_enter", "stuck")

needs_rust = pytest.mark.skipif(not hermes.rust_available(),
                                reason="mobslim_hermes (Rust kernel) not built")


def run(mobsim, scenario):
    log = EventLog()
    tr = mobsim.run(scenario, EventsManager([log]))
    return tr, log.events


def assert_same(a, b, label=""):
    (ta, ea), (tb, eb) = a, b
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(ta, name), getattr(tb, name),
                                      err_msg=f"{label} {name}")
    assert ea == eb, label


def random_grid(seed, size=5, persons=400, capacity=360.0, integer_times=True, walkers=0.0):
    """Random home-work-home commuters on a grid; optionally some walk."""
    net = grid_network(size, spacing=200.0, capacity=capacity)
    rng = Random(seed)
    n = net.n_nodes
    plans = {}
    for i in range(persons):
        home = rng.randrange(n)
        work = rng.randrange(n - 1)
        work += work >= home
        start = rng.randrange(0, 1800) if integer_times else rng.uniform(0, 1800)
        dur = rng.randrange(60, 900) if integer_times else rng.uniform(60, 900)
        mode = "walk" if rng.random() < walkers else "car"
        plans[i] = (Plan.builder().act("home", home, end_time=start).leg(mode)
                    .act("work", work, duration=dur).leg("car").act("home", home).build())
    return route_scenario(Scenario(net, Population.from_plans(plans)))


# (flow factor, storage factor): free flow, flow-bound, heavy spillback
CONGESTION = [(1.0, 1.0), (0.2, 0.05), (0.1, 0.03), (0.5, 0.01)]


def assert_same_up_to_float_noise(a, b, label=""):
    """Equal, except for float noise in continuous times (e.g. QSim
    accumulating a 100.00000000000001 s headway)."""
    (ta, ea), (tb, eb) = a, b
    for name in ARRAYS:
        np.testing.assert_allclose(getattr(ta, name), getattr(tb, name), rtol=0, atol=1e-6,
                                   err_msg=f"{label} {name}")
    if all(e.time == round(e.time) for e in ea):
        assert ea == eb, label  # no noise: same events in the same order
        return
    # noise can reorder events within a second (119.99999999999999 < 120)

    def key(e):
        return (round(e.time, 6), type(e).__name__, *astuple(e)[1:])

    assert sorted(map(key, ea)) == sorted(map(key, eb)), label


TRANSIT_ARRAYS = ("ride_wait_start", "ride_board", "ride_alight", "ride_trip")


def assert_same_transit(a, b, label=""):
    assert_same(a, b, label)
    (ta, _), (tb, _) = a, b
    for name in TRANSIT_ARRAYS:
        np.testing.assert_array_equal(getattr(ta, name), getattr(tb, name),
                                      err_msg=f"{label} {name}")
    for name in ("stop_arrival", "stop_departure", "link_enter"):
        np.testing.assert_array_equal(getattr(ta.transit, name), getattr(tb.transit, name),
                                      err_msg=f"{label} transit.{name}")


def random_corridor(seed, vehicles, blocking, persons=120):
    """Riders between random stops of ``test_transit``'s corridor, small
    buses with fractional door and passenger times, and cars sharing the road."""
    from tests.test_transit import (
        WALK,
        corridor_network,
        corridor_transit,
        node,
    )

    rng = Random(seed)
    net = corridor_network()
    transit = corridor_transit(
        net, vehicles=vehicles, capacity=3, access=1.4, egress=2.6, door_open=0.5,
        door_close=1.5, blocking=blocking, doors=rng.choice(["serial", "parallel"]),
        departures=[25200.0 + 150.0 * i for i in range(12)])
    plans = {}
    for i in range(persons):
        a, b = sorted(rng.sample(range(5), 2))
        mode = "car" if rng.random() < 0.3 else "pt"
        plans[i] = (Plan.builder().act("home", node(net, a), end_time=rng.uniform(24800, 26500))
                    .leg(mode).act("work", node(net, b)).build())
    return route_scenario(Scenario(net, Population.from_plans(plans),
                                   teleported_modes={"walk": WALK}, transit=transit))


# ------------------------------------------------------- rust == python


@needs_rust
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("flow,storage", CONGESTION)
@pytest.mark.parametrize("stuck_time", [math.inf, 30.0])
def test_rust_matches_python(seed, flow, storage, stuck_time):
    scenario = random_grid(seed, integer_times=False, walkers=0.2)
    cfg = dict(flow_capacity_factor=flow, storage_capacity_factor=storage, stuck_time=stuck_time)
    py = run(HermesSim(HermesSimConfig(backend="python", **cfg)), scenario)
    rs = run(HermesSim(HermesSimConfig(backend="rust", **cfg)), scenario)
    assert_same(py, rs, f"seed={seed} flow={flow} storage={storage} stuck={stuck_time}")


@needs_rust
@pytest.mark.parametrize("end_time", [600.0, 1234.5, 5000.0])
def test_rust_matches_python_when_cut_off(end_time):
    scenario = random_grid(3, integer_times=False, walkers=0.2)
    cfg = dict(flow_capacity_factor=0.1, storage_capacity_factor=0.03, end_time=end_time)
    py = run(HermesSim(HermesSimConfig(backend="python", **cfg)), scenario)
    rs = run(HermesSim(HermesSimConfig(backend="rust", **cfg)), scenario)
    assert_same(py, rs, f"end_time={end_time}")
    assert py[0].stuck.any()


@needs_rust
def test_rust_matches_python_far_in_the_future():
    """Times past the Rust kernel's dense agenda (~24 days) use its overflow
    buckets, including entries pushed for the second being drained."""
    late = 40 * 86400.0
    net = chain_network([100, 100], capacity=360)
    plans = {i: Plan.builder().act("h", 0, end_time=late + (i // 3)).leg("car")
             .act("w", 2, duration=7).leg("walk").act("h", 0).build() for i in range(12)}
    scenario = routed(Scenario(net, Population.from_plans(plans)))
    cfg = dict(storage_capacity_factor=0.01, stuck_time=2e6)
    py = run(HermesSim(HermesSimConfig(backend="python", **cfg)), scenario)
    rs = run(HermesSim(HermesSimConfig(backend="rust", **cfg)), scenario)
    assert_same(py, rs)
    assert not py[0].stuck.any() and np.nanmin(py[0].leg_departure) == late


@needs_rust
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("vehicles,blocking", [("schedule", False), ("network", False),
                                               ("network", True)])
def test_rust_matches_python_with_transit(seed, vehicles, blocking):
    scenario = random_corridor(seed, vehicles, blocking)
    cfg = dict(storage_capacity_factor=0.05, flow_capacity_factor=0.3)
    py = run(HermesSim(HermesSimConfig(backend="python", **cfg)), scenario)
    rs = run(HermesSim(HermesSimConfig(backend="rust", **cfg)), scenario)
    assert_same_transit(py, rs, f"seed={seed} vehicles={vehicles} blocking={blocking}")
    assert np.isfinite(py[0].ride_board).sum() > 50  # the riders rode
    assert (py[0].denied_boardings() > 0).any()  # and some were left behind


@needs_rust
@pytest.mark.parametrize("vehicles", ["schedule", "network"])
def test_rust_matches_python_on_pt_tutorial(vehicles):
    from tests.test_transit_scenarios import load

    scenario = route_scenario(load("pt_tutorial", vehicles))
    py = run(HermesSim(HermesSimConfig(backend="python")), scenario)
    rs = run(HermesSim(HermesSimConfig(backend="rust")), scenario)
    assert_same_transit(py, rs, vehicles)
    assert np.isfinite(py[0].ride_board).any()



# ------------------------------------------------- == QSim on whole seconds


@pytest.mark.parametrize("flow,storage", CONGESTION)
@pytest.mark.parametrize("stuck_time", [math.inf, 30.0])
def test_matches_qsim_with_whole_second_parameters(flow, storage, stuck_time):
    """200 m at 10 m/s (20 s), 360 veh/h (10 s headway, 50 s at 0.2 flow, ...)
    and whole-second activity times: the same moves in the same order."""
    scenario = random_grid(7, integer_times=True)
    cfg = dict(flow_capacity_factor=flow, storage_capacity_factor=storage, stuck_time=stuck_time)
    q = run(QSim(QSimConfig(**cfg)), scenario)
    h = run(HermesSim(HermesSimConfig(**cfg)), scenario)
    assert_same_up_to_float_noise(q, h, f"flow={flow} storage={storage} stuck={stuck_time}")


def test_close_to_qsim_with_fractional_parameters():
    """Fractional link times, headways and departures: congested travel times
    stay within a few percent of the QSim's. (Up to 7%: with storage x0.05
    and no stuck time, rounding reorders who gets freed space, and the
    one-at-a-time departure lines amplify that.)"""
    scenario = random_grid(5, size=6, persons=900, capacity=600.0, integer_times=False)
    for flow, storage in CONGESTION:
        cfg = dict(flow_capacity_factor=flow, storage_capacity_factor=storage, stuck_time=math.inf)
        q = QSim(QSimConfig(**cfg)).run(scenario, EventsManager())
        h = HermesSim(HermesSimConfig(**cfg)).run(scenario, EventsManager())
        tq = np.nansum(q.leg_arrival - q.leg_departure)
        th = np.nansum(h.leg_arrival - h.leg_departure)
        assert th == pytest.approx(tq, rel=0.07), (flow, storage)
        assert q.stuck.sum() == h.stuck.sum() == 0


# ----------------------------------------------------- discretisation


@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=needs_rust)])
def test_free_speed_time_rounds_half_up(backend):
    net = chain_network([104, 105, 106], freespeed=10)  # 10.4, 10.5, 10.6 s
    tr, events = run(HermesSim(HermesSimConfig(backend=backend)),
                     routed(Scenario(net, commuters(1, 0, 3))))
    assert [e.time for e in events if isinstance(e, LinkEnter)] == [0, 10, 21]
    assert [e.time for e in events if isinstance(e, PersonArrival)] == [32]


@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=needs_rust)])
def test_activity_ends_round_up(backend):
    net = chain_network([100], freespeed=10)
    plan = (Plan.builder().act("h", 0, end_time=100.2).leg("car")
            .act("w", 1, duration=4.5).leg("car").act("x", 1).build())
    _, events = run(HermesSim(HermesSimConfig(backend=backend)),
                    routed(Scenario(net, Population.from_plans({"a": plan}))))
    assert [e.time for e in events if isinstance(e, ActivityEnd)] == [101, 111 + 5]


@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=needs_rust)])
def test_teleport_time_rounds_half_up(backend):
    net = chain_network([1000])
    plan = Plan.builder().act("h", 0, end_time=0).leg("walk").act("w", 1).build()
    tr, events = run(HermesSim(HermesSimConfig(backend=backend)),
                     Scenario(net, Population.from_plans({"a": plan})))
    exact = 1300 / (3 / 3.6)  # beeline 1000 m * 1.3 at 3 km/h
    assert [e.time for e in events if isinstance(e, PersonArrival)] == [math.floor(exact + 0.5)]
    assert tr.leg_distance[0] == pytest.approx(1300)


def test_several_vehicles_leave_in_one_second():
    net = chain_network([100], freespeed=10, capacity=3 * 3600)  # 3 veh/s
    tr, _ = run(HermesSim(HermesSimConfig()), routed(Scenario(net, commuters(9, 0, 1))))
    # exact slots 10, 10.33, 10.67, 11, ...; each leaves at the first whole second at or after its slot
    assert tr.leg_arrival.tolist() == [10, 11, 11, 11, 12, 12, 12, 13, 13]


# ----------------------------------------------------- integration


def test_controller_learns_with_hermes(equil_scenario):
    result = Controller(equil_scenario, mobsim=HermesSim(),
                        config=ControllerConfig(iterations=10)).run()
    m = result.metrics
    assert m[-1]["score_best"] > m[0]["score_executed"]
    assert m[-1]["car_travel_time"] < m[0]["car_travel_time"]
    assert all(row["stuck"] == 0 for row in m)


def test_backend_selection(monkeypatch):
    assert HermesSim(HermesSimConfig(backend="python")).backend == "python"
    expected = "rust" if hermes.rust_available() else "python"
    assert HermesSim().backend == expected
    with pytest.raises(ValueError, match="backend"):
        HermesSimConfig(backend="java")
    monkeypatch.setattr(hermes, "_rust", None)
    assert HermesSim().backend == "python"
    with pytest.raises(ImportError, match="mobslim_hermes"):
        HermesSim(HermesSimConfig(backend="rust"))


def test_storage_never_exceeded_without_stuck_time():
    """With stuck_time=inf no vehicle is ever pushed onto a full link."""
    b = NetworkBuilder()
    for i in range(4):
        b.add_node(i, 100.0 * i, 0)
    for i in range(3):
        b.add_link(i, i, i + 1, 100, 10, 3600 if i < 2 else 60)  # last link: 1 veh/min
    net = b.build()
    scenario = routed(Scenario(net, commuters(40, 0, 3)))
    cfg = HermesSimConfig(stuck_time=math.inf, storage_capacity_factor=0.3)
    _, events = run(HermesSim(cfg), scenario)
    storage = hermes.kernel_inputs(scenario, hermes.PlanBatch.from_plans(
        scenario.population.selected_plans()), cfg, 0)[0]["storage"]
    occupancy = [0] * net.n_links
    for e in events:
        if isinstance(e, LinkEnter):
            occupancy[e.link] += 1
            assert occupancy[e.link] <= math.ceil(storage[e.link])
        elif isinstance(e, LinkLeave):
            occupancy[e.link] -= 1
    assert len([e for e in events if isinstance(e, PersonArrival)]) == 40
