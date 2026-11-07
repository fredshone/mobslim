"""ParallelSim: the Rust kernel equals the Python reference, and results do
not depend on the number of threads or how the nodes are partitioned.

Its queue-model behaviour is checked by the shared mobsim spec
(``tests/test_qsim.py``, which runs against ``parallel_py`` and, when built,
``parallel``).
"""

import math

import numpy as np
import pytest

from mobslim import (
    grid_network,
    read_network_xml,
    read_population_xml,
    route_scenario,
)
from mobslim.events import EventLog, EventsManager
from mobslim.mobsim import (
    HermesSim,
    HermesSimParams,
    ParallelSim,
    ParallelSimParams,
    SimulationError,
    parallel,
)
from mobslim.mobsim.parallel import partition_nodes
from mobslim.population import Plan, Population
from mobslim.scenario import Scenario
from tests.conftest import FIXTURES, commuters

needs_rust = pytest.mark.skipif(not parallel.rust_available(),
                                reason="mobslim_parallel extension not built")

FIELDS = ("act_start", "act_end", "leg_departure", "leg_arrival", "link_enter", "stuck")


@pytest.fixture(scope="module")
def equil():
    net = read_network_xml(FIXTURES / "equil" / "network.xml")
    return route_scenario(Scenario(net, read_population_xml(FIXTURES / "equil" / "plans100.xml", net)))


@pytest.fixture(scope="module")
def grid():
    """Commuters on a 6x6 grid, enough to fill links at low capacity."""
    net = grid_network(6, spacing=200.0, capacity=600.0)
    rng = np.random.default_rng(3)
    plans = {}
    for i in range(400):
        home, work = rng.choice(net.n_nodes, 2, replace=False).tolist()
        plans[i] = (Plan.builder().act("h", home, end_time=6 * 3600 + float(rng.uniform(0, 3600)))
                    .leg("car").act("w", work, duration=8 * 3600).leg("car").act("h", home).build())
    return route_scenario(Scenario(net, Population.from_plans(plans)))


def run(scenario, node_part=None, **params):
    log = EventLog()
    sim = ParallelSim(ParallelSimParams(**params))
    tr = sim.run_batch(scenario, scenario.population.selected_batch(), EventsManager([log]),
                       node_part=node_part)
    return tr, [repr(e) for e in log.events]


def assert_same(a, b):
    (ta, ea), (tb, eb) = a, b
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(ta, f), getattr(tb, f), err_msg=f)
    assert ea == eb


CASES = {
    "free": {},
    "congested": dict(flow_capacity_factor=0.05, storage_capacity_factor=0.05),
    "gridlock": dict(flow_capacity_factor=0.03, storage_capacity_factor=0.03, stuck_time=math.inf),
    "short stuck": dict(flow_capacity_factor=0.05, storage_capacity_factor=0.05, stuck_time=2.0),
    "end time": dict(flow_capacity_factor=0.1, storage_capacity_factor=0.1, end_time=8 * 3600),
}


@needs_rust
@pytest.mark.parametrize("case", sorted(CASES))
def test_rust_matches_python_on_any_threads(grid, case):
    ref = run(grid, backend="python", **CASES[case])
    for threads in (1, 2, 3, 8):
        assert_same(ref, run(grid, backend="rust", threads=threads, **CASES[case]))


@needs_rust
@pytest.mark.parametrize("junction_delays", [False, True])
@pytest.mark.parametrize("rounding", ["nearest", "down"])
@pytest.mark.parametrize("case", ["congested", "short stuck"])
def test_rust_matches_python_with_timing_options(grid, case, junction_delays, rounding):
    params = dict(CASES[case], junction_delays=junction_delays, rounding=rounding)
    ref = run(grid, backend="python", **params)
    for threads in (1, 3):
        assert_same(ref, run(grid, backend="rust", threads=threads, **params))


@needs_rust
def test_rust_matches_python_on_equil(equil):
    ref = run(equil, backend="python")
    assert_same(ref, run(equil, backend="rust", threads=4))


@needs_rust
@pytest.mark.parametrize("seed", range(3))
def test_any_partition_gives_the_same_result(grid, seed):
    """Random (non-contiguous) partitions: every move crosses partitions."""
    params = CASES["congested"]
    ref = run(grid, backend="rust", threads=1, **params)
    part = np.random.default_rng(seed).integers(0, 7, grid.network.n_nodes)
    assert_same(ref, run(grid, node_part=part, backend="rust", **params))


def test_python_kernel_ignores_partition(grid):
    part = np.arange(grid.network.n_nodes) % 3
    assert_same(run(grid, backend="python"), run(grid, node_part=part, backend="python"))


def test_free_flow_matches_hermes(grid):
    """Without congestion the rules only differ in corner cases (links under
    1 s, teleports under 1 s), so the trajectories match HermesSim's."""
    ours, _ = run(grid, backend="python")
    hermes = HermesSim(HermesSimParams(backend="python")).run(grid, EventsManager([]))
    np.testing.assert_array_equal(ours.leg_arrival, hermes.leg_arrival)
    np.testing.assert_array_equal(ours.link_enter, hermes.link_enter)


def test_space_is_reused_the_next_second():
    """One-vehicle link with a 10 s free-speed time: the second car can only
    enter one second after the first one left (Hermes: the same second)."""
    from tests.conftest import chain_network

    net = chain_network([7.5, 1000], freespeed=0.75, capacity=360)  # storage 1
    tr, _ = run(route_scenario(Scenario(net, commuters(2, 0, 2))), backend="python", stuck_time=1e9)
    first, second = tr.link_enter.reshape(2, 2)  # two route links per car
    assert first.tolist() == [0.0, 10.0]  # the first car leaves link 0 at 10 s
    assert second[0] == 11.0  # the second enters it at 11 s


def test_partition_nodes_is_balanced_and_complete(grid):
    plans = grid.population.selected_batch()
    for k in (1, 2, 4, 7):
        part = partition_nodes(grid.network, plans, k)
        assert part.shape == (grid.network.n_nodes,)
        assert set(part.tolist()) == set(range(k))
        weight = 1 + np.bincount(grid.network.to_node[plans.route_links.astype(np.int64)],
                                 minlength=grid.network.n_nodes)
        load = np.bincount(part, weights=weight, minlength=k)
        assert load.max() <= weight.sum() / k + weight.max()


TRANSIT_FIELDS = ("ride_wait_start", "ride_board", "ride_alight", "ride_trip")


def assert_same_transit(a, b):
    assert_same(a, b)
    (ta, _), (tb, _) = a, b
    for f in TRANSIT_FIELDS:
        np.testing.assert_array_equal(getattr(ta, f), getattr(tb, f), err_msg=f)
    for f in ("stop_arrival", "stop_departure", "link_enter"):
        np.testing.assert_array_equal(getattr(ta.transit, f), getattr(tb.transit, f),
                                      err_msg=f"transit.{f}")


TRANSIT_CASES = {
    "free": {},
    "congested": dict(flow_capacity_factor=0.1, storage_capacity_factor=0.1),
    "junctions": dict(junction_delays=True, rounding="down"),
}


@pytest.fixture(scope="module", params=[("schedule", False, False), ("network", False, False),
                                        ("network", True, False), ("network", False, True)],
                ids=lambda p: "-".join(map(str, p[:2])) + ("-rail" if p[2] else ""))
def transit_grid(request):
    """Bus lines (and, in the last case, rail lines on the timetable)."""
    from tests.conftest import transit_grid

    vehicles, blocking, rail = request.param
    return transit_grid(1, vehicles, blocking, rail=rail)


@needs_rust
@pytest.mark.parametrize("case", sorted(TRANSIT_CASES))
def test_transit_rust_matches_python_on_any_threads(transit_grid, case):
    """Buses, passengers and cars: identical to the Python kernel on any
    number of threads, so on any partition, even with lines crossing it."""
    ref = run(transit_grid, backend="python", **TRANSIT_CASES[case])
    tr = ref[0]
    assert np.isfinite(tr.ride_board).sum() > 400 and (tr.denied_boardings() > 0).any()
    for threads in (1, 2, 3, 8):
        assert_same_transit(ref, run(transit_grid, backend="rust", threads=threads,
                                     **TRANSIT_CASES[case]))


@needs_rust
@pytest.mark.parametrize("seed", range(2))
def test_transit_on_any_partition(transit_grid, seed):
    params = TRANSIT_CASES["congested"]
    ref = run(transit_grid, backend="rust", threads=1, **params)
    part = np.random.default_rng(seed).integers(0, 7, transit_grid.network.n_nodes)
    assert_same_transit(ref, run(transit_grid, node_part=part, backend="rust", **params))


def test_transit_close_to_hermes(transit_grid):
    """Without congestion only same-second merges differ (who goes first,
    and a bus leaving a stop crosses the node in the same step): trip times
    agree to within a second on average, and the same persons get stuck."""
    ours, _ = run(transit_grid, backend="python")
    hermes = HermesSim(HermesSimParams(backend="python")).run(transit_grid, EventsManager([]))
    ours_t, h_t = ours.leg_arrival - ours.leg_departure, hermes.leg_arrival - hermes.leg_departure
    assert abs(np.nanmean(ours_t - h_t)) < 1.0
    np.testing.assert_array_equal(ours.stuck, hermes.stuck)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        ParallelSimParams(threads=0)
    with pytest.raises(ValueError):
        ParallelSimParams(backend="gpu")
    with pytest.raises(ValueError):
        ParallelSimParams(stuck_time=0)


def test_unrouted_leg_raises(grid):
    plan = Plan.builder().act("h", 0, end_time=0).leg("car").act("w", 5).build()
    with pytest.raises(SimulationError):
        ParallelSim(ParallelSimParams(backend="python")).run(
            Scenario(grid.network, Population.from_plans({"x": plan})), EventsManager([]))
