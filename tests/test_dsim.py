"""DSim (matsim-rust's parallel design): trajectories and events do not
depend on the number of workers or the partition, and its link model
behaves as matsim-rust's.

The shared mobsim spec (``tests/test_qsim.py``) also runs it, with its
documented deviations as strict xfails.
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
from mobslim.events import EventLog, EventsManager, LinkEnter, PersonArrival
from mobslim.mobsim import (
    DSim,
    DSimParams,
    HermesSim,
    HermesSimParams,
    PSim,
    PSimParams,
    SimulationError,
    dsim,
    psim,
)
from mobslim.mobsim.partition import metis_partition, partition
from mobslim.population import Plan, Population
from mobslim.scenario import Scenario
from tests.conftest import FIXTURES, chain_network, commuters

pytestmark = pytest.mark.skipif(not dsim.rust_available(),
                                reason="mobslim_dsim extension not built")

FIELDS = ("act_start", "act_end", "leg_departure", "leg_arrival", "link_enter", "stuck")


def _has_metis() -> bool:
    try:
        import pymetis  # noqa: F401
    except ImportError:
        return False
    return True


metis = pytest.mark.skipif(not _has_metis(), reason="pymetis not installed")


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
    params.setdefault("partitioner", "morton")
    log = EventLog()
    tr = DSim(DSimParams(**params)).run_batch(
        scenario, scenario.population.selected_batch(), EventsManager([log]), node_part=node_part)
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


@pytest.mark.parametrize("junction_delays", [True, False])
@pytest.mark.parametrize("buffer_in_storage", [True, False])
@pytest.mark.parametrize("case", sorted(CASES))
def test_same_on_any_number_of_workers(grid, case, buffer_in_storage, junction_delays):
    params = dict(CASES[case], buffer_in_storage=buffer_in_storage,
                  junction_delays=junction_delays)
    ref = run(grid, threads=1, **params)
    for threads in (2, 3, 8):
        assert_same(ref, run(grid, threads=threads, **params))


@pytest.mark.parametrize("junction_delays", [True, False])
@pytest.mark.parametrize("buffer_in_storage", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_same_on_any_partition(grid, seed, buffer_in_storage, junction_delays):
    """Random (non-contiguous) partitions: most links are split, and many
    partitions are not neighbours."""
    params = dict(CASES["congested"], buffer_in_storage=buffer_in_storage,
                  junction_delays=junction_delays, rounding="nearest")
    ref = run(grid, threads=1, **params)
    part = np.random.default_rng(seed).integers(0, 7, grid.network.n_nodes)
    assert_same(ref, run(grid, node_part=part, **params))


@metis
def test_same_with_metis_on_equil(equil):
    ref = run(equil, threads=1)
    assert_same(ref, run(equil, threads=4, partitioner="metis"))


def test_seed_changes_merges_only_under_congestion(grid):
    free = [run(grid, seed=s)[0] for s in (1, 2)]
    np.testing.assert_array_equal(free[0].leg_arrival, free[1].leg_arrival)
    busy = [run(grid, seed=s, **CASES["congested"])[0] for s in (1, 2)]
    assert not np.array_equal(busy[0].leg_arrival, busy[1].leg_arrival)


def test_link_model_as_matsim_rust():
    """10 s links: a vehicle leaves the queue after 10 s and crosses the
    node one second later; it arrives when it leaves its last link."""
    net = chain_network([100, 100, 100], freespeed=10)
    log = EventLog()
    DSim(DSimParams(threads=2, partitioner="morton")).run(
        route_scenario(Scenario(net, commuters(1, 0, 3, dep=100))), EventsManager([log]))
    assert [e.time for e in log.events if isinstance(e, LinkEnter)] == [100, 111, 122]
    assert [e.time for e in log.events if isinstance(e, PersonArrival)] == [132]


def _two_link_loop():
    """Two 1-vehicle links forming a loop, with one vehicle on each wanting the other."""
    from tests.test_qsim import _gridlock_scenario

    return _gridlock_scenario()


def test_buffered_vehicles_keep_their_storage():
    """By default a vehicle waiting to cross a node still fills its link,
    so the loop locks (and stays locked without a stuck time)."""
    tr = DSim(DSimParams(threads=2, partitioner="morton", stuck_time=math.inf)).run(
        _two_link_loop(), EventsManager([]))
    assert tr.stuck.all()


def test_matsim_rust_frees_storage_on_entering_the_buffer():
    """buffer_in_storage=False: the buffered vehicles free their links, and
    each one crosses into the other's."""
    tr = DSim(DSimParams(threads=2, partitioner="morton", stuck_time=math.inf,
                                 buffer_in_storage=False)).run(_two_link_loop(), EventsManager([]))
    assert not tr.stuck.any()


def test_flow_capacity_limits_crossing_a_node():
    """Through traffic uses flow capacity (1 veh / 10 s on the first link);
    arrivals at the end of a route do not."""
    net = chain_network([100, 100], freespeed=10, capacity=360)
    tr = DSim(DSimParams(partitioner="morton")).run(
        route_scenario(Scenario(net, commuters(4, 0, 2))), EventsManager([]))
    second_link = tr.link_enter.reshape(4, 2)[:, 1]
    assert np.diff(second_link).tolist() == [10.0, 10.0, 10.0]
    assert np.diff(tr.leg_arrival).tolist() == [10.0, 10.0, 10.0]


def test_free_flow_close_to_hermes(grid):
    """Without congestion, a leg takes HermesSim's time plus at most one
    second per link (the step from buffer over the node, less flooring)."""
    lots = dict(flow_capacity_factor=100.0, storage_capacity_factor=100.0)
    ours, _ = run(grid, **lots)
    hermes = HermesSim(HermesSimParams(backend="python", **lots)).run(grid, EventsManager([]))
    links = np.diff(grid.population.selected_batch().route_offsets)
    extra = (ours.leg_arrival - ours.leg_departure) - (hermes.leg_arrival - hermes.leg_departure)
    assert (extra >= 0).all() and (extra <= links).all()


@metis
def test_metis_partition_is_balanced_and_contiguous(grid):
    net = grid.network
    for k in (2, 4, 7):
        part = metis_partition(net, k)
        assert part.shape == (net.n_nodes,) and set(part.tolist()) == set(range(k))
        assert np.bincount(part).max() <= math.ceil(net.n_nodes / k * 1.03) + 1
        # fewer links cut than a random split
        cut = np.mean(part[net.from_node] != part[net.to_node])
        rand = np.random.default_rng(0).integers(0, k, net.n_nodes)
        assert cut < np.mean(rand[net.from_node] != rand[net.to_node]) / 2


@metis
def test_psim_results_do_not_depend_on_the_partitioner(grid):
    if not psim.rust_available():
        pytest.skip("mobslim_psim extension not built")
    params = dict(backend="rust", threads=3, flow_capacity_factor=0.05, storage_capacity_factor=0.05)
    a = PSim(PSimParams(**params, partitioner="morton")).run(grid, EventsManager([]))
    b = PSim(PSimParams(**params, partitioner="metis")).run(grid, EventsManager([]))
    np.testing.assert_array_equal(a.leg_arrival, b.leg_arrival)


def test_unknown_partitioner_rejected(grid):
    with pytest.raises(ValueError):
        partition(grid.network, None, 2, "random")
    with pytest.raises(ValueError):
        DSimParams(partitioner="random")
    with pytest.raises(ValueError):
        PSimParams(partitioner="random")


TRANSIT_FIELDS = ("ride_wait_start", "ride_board", "ride_alight", "ride_trip")


def assert_same_transit(a, b):
    assert_same(a, b)
    (ta, _), (tb, _) = a, b
    for f in TRANSIT_FIELDS:
        np.testing.assert_array_equal(getattr(ta, f), getattr(tb, f), err_msg=f)
    for f in ("stop_arrival", "stop_departure", "link_enter"):
        np.testing.assert_array_equal(getattr(ta.transit, f), getattr(tb.transit, f),
                                      err_msg=f"transit.{f}")


@pytest.fixture(scope="module", params=[("schedule", False, False), ("network", False, False),
                                        ("network", True, False), ("network", False, True)],
                ids=lambda p: "-".join(map(str, p[:2])) + ("-rail" if p[2] else ""))
def transit_grid(request):
    """Bus lines (and, in the last case, rail lines on the timetable)."""
    from tests.conftest import transit_grid

    vehicles, blocking, rail = request.param
    return transit_grid(1, vehicles, blocking, rail=rail)


TRANSIT_CASES = {
    "free": {},
    "congested": dict(flow_capacity_factor=0.1, storage_capacity_factor=0.1),
    "no junction delays": dict(junction_delays=False, rounding="nearest"),
}


@pytest.mark.parametrize("case", sorted(TRANSIT_CASES))
def test_transit_same_on_any_number_of_workers(transit_grid, case):
    """Buses, passengers and cars give identical trajectories and events on
    1, 2, 3 and 8 workers, with lines crossing the partitions."""
    ref = run(transit_grid, threads=1, **TRANSIT_CASES[case])
    tr = ref[0]
    assert np.isfinite(tr.ride_board).sum() > 400 and (tr.denied_boardings() > 0).any()
    for threads in (2, 3, 8):
        assert_same_transit(ref, run(transit_grid, threads=threads, **TRANSIT_CASES[case]))


@pytest.mark.parametrize("seed", range(2))
def test_transit_same_on_any_partition(transit_grid, seed):
    params = TRANSIT_CASES["congested"]
    ref = run(transit_grid, threads=1, **params)
    part = np.random.default_rng(seed).integers(0, 7, transit_grid.network.n_nodes)
    assert_same_transit(ref, run(transit_grid, node_part=part, **params))


def test_transit_close_to_hermes(transit_grid):
    """With HermesSim's timing (no junction delays, nearest rounding), pt and
    car trips take about as long as in HermesSim (a few seconds of
    matsim-rust's dynamics per trip), and the same persons are stuck."""
    ours, _ = run(transit_grid, junction_delays=False, rounding="nearest")
    hermes = HermesSim(HermesSimParams(backend="python")).run(transit_grid, EventsManager([]))
    diff = (ours.leg_arrival - ours.leg_departure) - (hermes.leg_arrival - hermes.leg_departure)
    assert abs(np.nanmean(diff)) < 5.0
    np.testing.assert_array_equal(ours.stuck, hermes.stuck)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        DSimParams(threads=0)
    with pytest.raises(ValueError):
        DSimParams(stuck_time=0)
    with pytest.raises(ValueError):
        DSimParams(seed=-1)


def test_unrouted_leg_raises(grid):
    plan = Plan.builder().act("h", 0, end_time=0).leg("car").act("w", 5).build()
    with pytest.raises(SimulationError):
        DSim().run(Scenario(grid.network, Population.from_plans({"x": plan})), EventsManager([]))
