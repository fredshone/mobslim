"""Link traversal sums built while simulating (``traversal_bins``, see
``mobslim.mobsim.base.run_mobsim``) equal the sums of the trajectories."""

import dataclasses
import math

import numpy as np
import pytest

from mobslim import Plan, Population, Scenario, grid_network, route_scenario
from mobslim.events import EventsManager
from mobslim.mobsim import run_mobsim
from mobslim.routing.traveltime import DEFAULT_HORIZON, SparseAggregate
from tests.conftest import MOBSIMS, make_mobsim, transit_grid

COMPILED = [k for k in ("hermes", "psim", "dsim") if k in MOBSIMS]
BINS = (900.0, DEFAULT_HORIZON)

CASES = {
    "free": {},
    "congested": dict(flow_capacity_factor=0.05, storage_capacity_factor=0.05),
    "gridlock": dict(flow_capacity_factor=0.03, storage_capacity_factor=0.03, stuck_time=math.inf),
    "short stuck": dict(flow_capacity_factor=0.05, storage_capacity_factor=0.05, stuck_time=2.0),
    "end time": dict(flow_capacity_factor=0.1, storage_capacity_factor=0.1, end_time=8 * 3600),
    "junction delays": dict(flow_capacity_factor=0.05, storage_capacity_factor=0.05,
                            junction_delays=True),
}


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


def summed(tr, network, bin_size, horizon):
    """The sums of ``tr``'s trajectories (ignoring any built in the mobsim)."""
    return dataclasses.replace(tr, aggregate=None).traversal_aggregate(network, bin_size, horizon)


def assert_same(a, b):
    for name in ("total", "count", "sum_sq"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.bin_size == b.bin_size


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
@pytest.mark.parametrize("kind", COMPILED)
@pytest.mark.parametrize("case", sorted(CASES))
def test_in_sim_sums_equal_the_trajectories(grid, kind, case):
    tr = make_mobsim(kind, **CASES[case]).run(grid, EventsManager(), traversal_bins=BINS)
    assert isinstance(tr.aggregate, SparseAggregate)
    assert len(np.unique(tr.aggregate.cells)) == len(tr.aggregate.cells)
    got = tr.traversal_aggregate(grid.network, *BINS)
    assert_same(got, summed(tr, grid.network, *BINS))
    assert got.count.sum() > 0


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
@pytest.mark.parametrize("kind", COMPILED)
@pytest.mark.parametrize("bins", [(450.5, DEFAULT_HORIZON), (60.0, 7.5 * 3600), (7200.0, 3600.0)],
                         ids=["fractional bins", "clamped late", "one bin"])
def test_in_sim_sums_with_other_bins(grid, kind, bins):
    """Bins that are not whole seconds, and traversals past the table's
    end (clamped into its last bin)."""
    tr = make_mobsim(kind, **CASES["congested"]).run(grid, EventsManager(), traversal_bins=bins)
    assert tr.aggregate is not None
    assert_same(tr.traversal_aggregate(grid.network, *bins), summed(tr, grid.network, *bins))


@pytest.mark.parametrize("kind", [k for k in ("psim", "dsim") if k in MOBSIMS])
def test_in_sim_sums_do_not_depend_on_the_threads(grid, kind):
    ref = None
    for threads in (1, 2, 5):
        tr = make_mobsim(kind, threads=threads, **CASES["congested"]).run(
            grid, EventsManager(), traversal_bins=BINS)
        agg = tr.traversal_aggregate(grid.network, *BINS)
        if ref is None:
            ref = agg
        else:
            assert_same(agg, ref)


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
@pytest.mark.parametrize("kind", COMPILED)
@pytest.mark.parametrize("learn", [False, True])
def test_in_sim_sums_with_transit(kind, learn):
    """Buses in traffic: the sums are the persons'; with
    ``learn_from_transit`` the buses' traversals are added to them."""
    sc = transit_grid(seed=2)
    sc = dataclasses.replace(sc, transit=dataclasses.replace(
        sc.transit, params=dataclasses.replace(sc.transit.params, learn_from_transit=learn)))
    tr = make_mobsim(kind).run(sc, EventsManager(), traversal_bins=BINS)
    assert tr.aggregate is not None
    got = tr.traversal_aggregate(sc.network, *BINS)
    assert_same(got, summed(tr, sc.network, *BINS))
    sparse = tr.traversal_aggregate(sc.network, *BINS, sparse=True)
    assert_same(sparse.to_dense(), got)


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
def test_other_bins_than_built_are_summed_from_the_trajectories(grid):
    tr = make_mobsim(COMPILED[0]).run(grid, EventsManager(), traversal_bins=BINS)
    for bins in [(600.0, DEFAULT_HORIZON), (900.0, 24 * 3600.0)]:
        assert_same(tr.traversal_aggregate(grid.network, *bins), summed(tr, grid.network, *bins))
    assert tr.traversal_aggregate(grid.network, *BINS, sparse=True) is tr.aggregate


@pytest.mark.parametrize("kind", sorted(MOBSIMS))
def test_every_mobsim_takes_traversal_bins(grid, kind):
    """The common interface: every built-in mobsim takes the request (the
    Python ones leave the sums to the trajectories), with the same sums."""
    mobsim = make_mobsim(kind, **CASES["congested"])
    assert mobsim.sums_traversals
    plain = mobsim.run(grid, EventsManager())
    asked = run_mobsim(mobsim, grid, EventsManager(), BINS)
    assert plain.aggregate is None
    assert_same(asked.traversal_aggregate(grid.network, *BINS),
                plain.traversal_aggregate(grid.network, *BINS))


def test_run_mobsim_does_not_ask_mobsims_that_do_not_take_it(grid):
    class Plain:
        def __init__(self):
            self.inner = make_mobsim()

        def run(self, scenario, events):
            return self.inner.run(scenario, events)

    tr = run_mobsim(Plain(), grid, EventsManager(), BINS)
    assert tr.aggregate is None


# ------------------------------------------------------------------ link times

SAME = ("act_start", "act_end", "leg_departure", "leg_arrival", "leg_distance", "stuck",
        "leg_entered", "ride_wait_start", "ride_board", "ride_alight", "ride_trip")


def assert_same_without_link_times(full, lean):
    assert full.link_enter is not None and lean.link_enter is None
    for name in SAME:
        np.testing.assert_array_equal(getattr(full, name), getattr(lean, name), err_msg=name)
    assert full.experienced_plans() == lean.experienced_plans()
    if full.transit is not None:
        np.testing.assert_array_equal(full.transit.link_enter, lean.transit.link_enter)


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
@pytest.mark.parametrize("kind", COMPILED)
@pytest.mark.parametrize("case", sorted(CASES))
def test_without_link_times_everything_else_is_the_same(grid, kind, case):
    """Without the per-link entry times, the kernels report how many links
    each leg entered: the same as counting the recorded entries, so leg
    distances and experienced routes do not change."""
    mobsim = make_mobsim(kind, **CASES[case])
    full = mobsim.run(grid, EventsManager(), traversal_bins=BINS)
    lean = mobsim.run(grid, EventsManager(), traversal_bins=BINS, link_times=False)
    assert_same_without_link_times(full, lean)
    from mobslim.trajectories import entered_per_leg
    np.testing.assert_array_equal(full.leg_entered, entered_per_leg(full.plans, full.link_enter))
    assert_same(lean.traversal_aggregate(grid.network, *BINS),
                full.traversal_aggregate(grid.network, *BINS))
    if case == "end time":
        assert lean.stuck.any()  # legs under way at the end are covered


@pytest.mark.parametrize("kind", [k for k in ("psim", "dsim") if k in MOBSIMS])
def test_without_link_times_on_any_threads(grid, kind):
    ref = make_mobsim(kind, threads=1, **CASES["end time"]).run(grid, EventsManager())
    for threads in (2, 5):
        lean = make_mobsim(kind, threads=threads, **CASES["end time"]).run(
            grid, EventsManager(), link_times=False)
        assert_same_without_link_times(ref, lean)


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
@pytest.mark.parametrize("kind", COMPILED)
def test_without_link_times_with_transit(kind):
    """The buses' link entries are still recorded."""
    sc = transit_grid(seed=2)
    mobsim = make_mobsim(kind)
    full = mobsim.run(sc, EventsManager())
    lean = mobsim.run(sc, EventsManager(), link_times=False)
    assert_same_without_link_times(full, lean)
    assert np.isfinite(lean.transit.link_enter).any()


def test_link_times_are_asked_for_where_they_are_needed():
    from mobslim import ControllerParams

    p = ControllerParams(iterations=10)
    assert [i for i in range(10) if p.records_link_times(i)] == [9]
    p = ControllerParams(iterations=10, link_times_interval=4)
    assert [i for i in range(10) if p.records_link_times(i)] == [0, 4, 8, 9]


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
def test_without_link_times_link_level_views_say_why(grid):
    lean = make_mobsim(COMPILED[0]).run(grid, EventsManager(), link_times=False)
    with pytest.raises(ValueError, match="were not recorded"):
        lean.link_leave()
    with pytest.raises(ValueError, match="were not recorded"):
        lean.traversal_aggregate(grid.network)  # no in-sim sums to fall back on


def test_controllers_record_link_times_in_output_iterations_only(grid):
    """The listeners see link times in the last iteration only (by default)."""
    from mobslim import Controller, ControllerListener, ControllerParams, Weave
    from mobslim.weave import SerialLoom

    kind = COMPILED[0] if COMPILED else None

    class Seen(ControllerListener):
        def __init__(self):
            self.recorded = []

        def on_after_mobsim(self, state):
            self.recorded.append(state.trajectories.link_enter is not None)

    for make in (lambda s: Controller(grid, mobsim=make_mobsim(kind), listeners=[s],
                                      params=ControllerParams(iterations=3)),
                 lambda s: Weave(grid, strands=2, loom=SerialLoom(), mobsim=make_mobsim(kind),
                                 listeners=[s], params=ControllerParams(iterations=3))):
        seen = Seen()
        make(seen).run()
        expected = [False, False, True] if kind else [True, True, True]  # Python mobsims: always
        assert seen.recorded == expected


# ------------------------------------------------------------------ uneven bins

def test_off_peak_bins():
    from mobslim.routing.traveltime import TimeBins

    b = TimeBins.off_peak()  # 15 min from 06:00 to 22:00, hourly before and after
    assert b.n_bins == 6 + 64 + 8 and b.horizon == DEFAULT_HORIZON
    np.testing.assert_array_equal(b.starts[:8] / 3600, [0, 1, 2, 3, 4, 5, 6, 6.25])
    np.testing.assert_array_equal(b.starts[-9:] / 3600, [21.75, 22, 23, 24, 25, 26, 27, 28, 29])
    times = np.array([-5.0, 0, 3599, 3600, 21599, 21600, 22499, 22500, 79199, 79200, 1e9, np.nan])
    np.testing.assert_array_equal(b.bin_of(times), [0, 0, 0, 1, 5, 6, 6, 7, 69, 70, 77, 0])
    with pytest.raises(ValueError, match="multiple"):
        TimeBins.off_peak(900, 1000)
    assert TimeBins.uniform(900, 3600) == TimeBins(900, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="runs of slots"):
        TimeBins(900, [0, 2])


def uneven():
    from mobslim.routing.traveltime import TimeBins

    return {"off peak": TimeBins.off_peak(),
            "irregular": TimeBins(450.0, [0, 0, 0, 1, 2, 2, 3, 3, 3, 3] + [4] * 300)}


@pytest.mark.skipif(not COMPILED, reason="no compiled mobsim built")
@pytest.mark.parametrize("name", ["off peak", "irregular"])
def test_uneven_bins_sum_like_their_slots(grid, name):
    """Summing in uneven bins equals summing in the uniform slots and then
    adding up each bin's slots (numpy and compiled paths, and the mobsims;
    whole-second times, so the sums are exact)."""
    import mobslim.routing.traveltime as tt
    from mobslim.routing.traveltime import TraversalAggregate

    bins = uneven()[name]
    for kind in COMPILED:
        tr = make_mobsim(kind, **CASES["congested"]).run(
            grid, EventsManager(), traversal_bins=(bins, DEFAULT_HORIZON))
        fine = summed(tr, grid.network, bins.slot, bins.horizon)
        assert fine.n_bins == bins.n_slots
        coarse = summed(tr, grid.network, bins, DEFAULT_HORIZON)
        assert coarse.n_bins == bins.n_bins and coarse.bin_size == bins
        for field in ("total", "count", "sum_sq"):
            a = getattr(fine, field)
            expect = np.zeros((a.shape[0], bins.n_bins), dtype=a.dtype)
            np.add.at(expect.T, bins.slot_bin, a.T)
            np.testing.assert_array_equal(getattr(coarse, field), expect, err_msg=field)
        # the numpy path (without the compiled accumulate) gives the same
        agg = TraversalAggregate.empty(grid.network.n_links, bins.n_bins, bins)
        saved, tt._accumulate = tt._accumulate, None
        try:
            agg.accumulate(tr.plans.route_links, tr.link_enter, tr.link_leave())
        finally:
            tt._accumulate = saved
        assert_same(agg, coarse)
        # and so do the sums built while simulating
        assert tr.aggregate is not None
        assert_same(tr.traversal_aggregate(grid.network, bins), coarse)


def test_uneven_bins_route_like_their_slots(grid):
    """A table in uneven bins routes as the same table spread over its slots
    (the compiled and Python routers)."""
    from mobslim.routing import DijkstraRouter
    from mobslim.routing.compiled import RustRouter, rust_router_available
    from mobslim.routing.traveltime import TravelTimeTable

    bins = uneven()["off peak"]
    tr = make_mobsim(**CASES["congested"]).run(grid, EventsManager())
    table = tr.travel_time_table(grid.network, bins)
    spread = TravelTimeTable(table.times[:, bins.slot_bin], bins.slot)
    for t in (0.0, 3 * 3600.0, 6 * 3600.0 + 1, 7.5 * 3600, 21.9 * 3600, 23 * 3600, 1e6):
        assert table.link_travel_times(t) == spread.link_travel_times(t)
    routers = [DijkstraRouter]
    if rust_router_available():
        routers.append(RustRouter)
    rng = np.random.default_rng(0)
    queries = [(int(o), int(d), float(t)) for o, d, t in zip(
        rng.integers(0, grid.network.n_nodes, 40), rng.integers(0, grid.network.n_nodes, 40),
        rng.uniform(0, 26 * 3600, 40)) if o != d]
    for make in routers:
        a, b = make(grid.network, "car", table), make(grid.network, "car", spread)
        for o, d, t in queries:
            ra, rb = a.route(o, d, t), b.route(o, d, t)
            assert (ra.links, ra.travel_time) == (rb.links, rb.travel_time)


def test_controllers_learn_in_off_peak_bins_by_default(tmp_path):
    from mobslim import ControllerParams
    from mobslim.routing.traveltime import TimeBins

    assert ControllerParams().travel_time_bins() == TimeBins.off_peak()
    assert ControllerParams(travel_time_off_peak_bin_size=900).travel_time_bins() == 900
    with pytest.raises(ValueError, match="multiple"):
        ControllerParams(travel_time_off_peak_bin_size=1000)


# ------------------------------------------------------------------ compact tables

def test_learned_tables_are_whole_seconds_in_two_bytes(grid, caplog):
    from mobslim.routing.traveltime import compact_times

    tr = make_mobsim(**CASES["congested"]).run(grid, EventsManager())
    agg = tr.traversal_aggregate(grid.network)
    exact, table = agg.table(grid.network, compact=False), agg.table(grid.network)
    assert table.times.dtype == np.uint16
    np.testing.assert_array_equal(table.times, np.ceil(exact.times))
    assert (table.times >= grid.network.free_speed_travel_times()[:, None]).all()
    assert table.link_travel_time(0, 7 * 3600) == float(table.times[0, table.time_bin(7 * 3600)])
    with caplog.at_level("WARNING"):
        long = compact_times(np.array([[10.2, 70000.0]]))
    assert long.dtype == np.float32 and long.tolist() == [[11.0, 70000.0]]
    assert "float32" in caplog.text


def test_compact_tables_keep_the_landmark_bounds(grid):
    """Rounded up, a learned table is never below free speed, so the compiled
    router still uses its landmarks (ALT), and finds the same routes as on
    the same times in float64."""
    from mobslim.routing.compiled import RustRouter, rust_router_available
    from mobslim.routing.traveltime import TravelTimeTable

    if not rust_router_available():
        pytest.skip("mobslim_router not built")
    tr = make_mobsim(**CASES["congested"]).run(grid, EventsManager())
    table = tr.travel_time_table(grid.network)
    router = RustRouter(grid.network, "car", table)
    assert router.algorithm == "alt"
    wide = RustRouter(grid.network, "car",
                      TravelTimeTable(table.times.astype(np.float64), table.bin_size))
    for o, d, t in [(0, 35, 7 * 3600.0), (5, 30, 6.5 * 3600), (12, 1, 2 * 3600.0)]:
        assert router.route(o, d, t).links == wide.route(o, d, t).links
