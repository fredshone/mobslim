"""AgentSim-specific behaviour. The shared queue-model spec (free speed, flow,
storage, FIFO, stuck time, validation, ...) runs against AgentSim in
``test_qsim.py``; this module covers where AgentSim deliberately differs
from (or must exactly agree with) the QSim."""

import numpy as np
import pytest

from mobslim import (
    AgentSim,
    AgentSimParams,
    NetworkBuilder,
    Plan,
    Population,
    QSim,
    QSimParams,
    Scenario,
)
from mobslim.events import (
    EventLog,
    EventsManager,
    LinkEnter,
    PersonArrival,
    PersonStuck,
)
from tests.conftest import chain_network, commuters, routed, simulate


def arrivals(events):
    return [e.time for e in events if isinstance(e, PersonArrival)]


def _assert_same_trajectories(a, b):
    for name in ("act_start", "act_end", "leg_departure", "leg_arrival", "link_enter"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    np.testing.assert_array_equal(a.stuck, b.stuck)


def test_matches_qsim_without_spillback(equil_scenario):
    """With storage never binding, polling never happens: same trajectories."""
    scenario = routed(equil_scenario)
    q = QSim(QSimParams(storage_capacity_factor=1e6)).run(scenario, EventsManager())
    a = AgentSim(AgentSimParams(storage_capacity_factor=1e6)).run(scenario, EventsManager())
    _assert_same_trajectories(q, a)


def test_flow_headway_below_retry_interval_is_exact():
    """FIFO successors are scheduled at their exact exit time, never polled,
    so a 0.5 s headway is honoured with a 1 s retry interval."""
    net = chain_network([100], freespeed=10, capacity=7200)
    times = arrivals(simulate(routed(Scenario(net, commuters(6, 0, 1))), "agentsim", retry_interval=1.0))
    assert times == [10.0 + 0.5 * k for k in range(6)]


def _spillback_scenario():
    # up: 100 m, 10 s free time; down: holds 1 vehicle, 1 s free time, 1 veh/min
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 100, 0)
    b.add_node(2, 107.5, 0)
    b.add_link("up", 0, 1, 100, 10, 3600)
    b.add_link("down", 1, 2, 7.5, 7.5, 60)
    return routed(Scenario(b.build(), commuters(2, 0, 2)))


def _second_entry_into_down(events):
    return [e.time for e in events if isinstance(e, LinkEnter) and e.link == 1][1]


@pytest.mark.parametrize("retry, expected", [(1.0, 12.0), (0.25, 11.25)])
def test_blocked_vehicle_polls_for_space(retry, expected):
    """The second vehicle is blocked at 11 s by the full downstream link,
    which empties at 11 s. QSim wakes it immediately; AgentSim finds the
    space on its next retry."""
    scenario = _spillback_scenario()
    assert _second_entry_into_down(simulate(scenario, "qsim")) == 11.0
    events = simulate(scenario, "agentsim", retry_interval=retry)
    assert _second_entry_into_down(events) == pytest.approx(expected)


def test_stuck_time_caps_polling():
    """A blocked vehicle is forced on at exactly blocked_since + stuck_time,
    even when that falls between retries."""
    # downstream link holds 1 vehicle (100 s free time x 30 veh/h < 1) for 100 s
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 100, 0)
    b.add_node(2, 107.5, 0)
    b.add_link("up", 0, 1, 100, 10, 3600)
    b.add_link("down", 1, 2, 7.5, 0.075, 30)
    scenario = routed(Scenario(b.build(), commuters(2, 0, 2)))
    events = simulate(scenario, "agentsim", retry_interval=3.0, stuck_time=10.0)
    assert _second_entry_into_down(events) == pytest.approx(21.0)  # blocked at 11; retries at 14, 17, 20, forced at 21


def _departure_line_scenario(departures):
    """One link that holds a single vehicle for 10 s (7.5 m at 0.75 m/s,
    360 veh/h); one car trip across it per departure time."""
    b = NetworkBuilder()
    b.add_node(0, 0, 0)
    b.add_node(1, 7.5, 0)
    b.add_link("a", 0, 1, 7.5, 0.75, 360)
    plans = {i: Plan.builder().act("home", 0, end_time=t).leg("car").act("work", 1).build()
             for i, t in enumerate(departures)}
    return routed(Scenario(b.build(), Population.from_plans(plans)))


def _entries(events):
    return [(e.person, e.time) for e in events if isinstance(e, LinkEnter)]


def test_departures_onto_a_full_link_are_fifo():
    """Person 1 is blocked first (0.75 s), person 2 later (1.5 s). The link
    frees at 10 s. Polling alone would let person 2 in first (its retry at
    10.5 s comes before person 1's at 10.75 s); the departure line keeps
    them in order, and person 2 only starts trying once person 1 is in.
    Its wait (and its retry grid) starts when it reaches the front at
    10.75 s; the link frees again at 20.75 s, a retry tick, so it gets in then."""
    events = simulate(_departure_line_scenario([0.0, 0.75, 1.5]), "agentsim",
                      retry_interval=1.0, stuck_time=float("inf"))
    assert _entries(events) == [(0, 0.0), (1, 10.75), (2, 20.75)]


def test_queued_departers_count_stuck_time_from_reaching_the_front():
    """Person 1 (blocked at 0.75 s) is forced on at 5.75 s. Person 2 then
    heads the line; its stuck time runs from then, so it is forced on 5 s
    later, at 10.75 s (MATSim's buffer: one forced departure per stuck time)."""
    events = simulate(_departure_line_scenario([0.0, 0.75, 1.5]), "agentsim",
                      retry_interval=1.0, stuck_time=5.0)
    assert _entries(events) == [(0, 0.0), (1, 5.75), (2, 10.75)]


def _ring_gridlock():
    """Four links in a ring, each holding one vehicle; four cars each start
    on a different link and drive two links on. Every head wants the next
    link, which is full: a gridlock. A fifth car, on a separate link an hour
    later, keeps the run going meanwhile."""
    b = NetworkBuilder()
    for i in range(6):
        b.add_node(i, float(i), 0.0)
    for i in range(4):
        b.add_link(f"{i}", i, (i + 1) % 4, 7.5, 7.5, 3600)
    b.add_link("x", 4, 5, 100, 10, 3600)
    plans = {i: Plan.builder().act("home", i, end_time=0.0).leg("car").act("work", (i + 2) % 4).build()
             for i in range(4)}
    plans[4] = Plan.builder().act("home", 4, end_time=3600.0).leg("car").act("work", 5).build()
    return routed(Scenario(b.build(), Population.from_plans(plans)))


def _run_counting_retries(scenario, monkeypatch, **config):
    from mobslim.mobsim import agentsim
    calls = []
    retry = agentsim._AgentSimRun.retry
    monkeypatch.setattr(agentsim._AgentSimRun, "retry", lambda self, a: (calls.append(a), retry(self, a)))
    log = EventLog()
    trajectories = AgentSim(AgentSimParams(**config)).run(scenario, EventsManager([log]))
    return trajectories, log.events, len(calls)


_CONGESTED = [
    dict(storage_capacity_factor=0.002, flow_capacity_factor=0.01),
    dict(storage_capacity_factor=0.002, flow_capacity_factor=0.01, stuck_time=float("inf")),
    dict(storage_capacity_factor=0.002, flow_capacity_factor=0.01, stuck_time=300, retry_interval=0.3),
    dict(storage_capacity_factor=0.01, flow_capacity_factor=0.05, stuck_time=30, end_time=8 * 3600),
]


@pytest.mark.parametrize("config", _CONGESTED, ids=["stuck10", "no_stuck", "stuck300", "cut_off"])
def test_chain_mode_matches_polling_exactly(equil_scenario, monkeypatch, config):
    """Chain mode only skips retries that are certain to fail, and retries
    neither depend on when they were pushed nor change anything when they
    fail: the run is identical, with far fewer retries."""
    scenario = routed(equil_scenario)
    poll, poll_events, poll_retries = _run_counting_retries(scenario, monkeypatch, **config)
    chain, chain_events, chain_retries = _run_counting_retries(
        scenario, monkeypatch, blocked="chain", **config)
    _assert_same_trajectories(poll, chain)
    assert chain_events == poll_events
    assert poll_retries > 0 and chain_retries * 5 < poll_retries


def test_chain_mode_parks_a_gridlock(monkeypatch):
    """With stuck_time=inf, a blocking chain that loops back on itself can
    never move: chain mode stops retrying at once instead of polling until
    the rest of the traffic is done and the gridlock is detected.
    Trajectories are the same."""
    scenario = _ring_gridlock()
    poll, poll_events, poll_retries = _run_counting_retries(
        scenario, monkeypatch, stuck_time=float("inf"))
    chain, chain_events, chain_retries = _run_counting_retries(
        scenario, monkeypatch, stuck_time=float("inf"), blocked="chain")
    _assert_same_trajectories(poll, chain)
    assert chain.stuck.tolist() == [True] * 4 + [False]
    assert {(e.person, e.link) for e in chain_events if isinstance(e, PersonStuck)} == \
        {(e.person, e.link) for e in poll_events if isinstance(e, PersonStuck)}
    # each car fails once at 1 s; the first three still see an unblocked head
    # ahead, retry once at 2 s and find the cycle; then all four are parked
    assert chain_retries == 7
    assert poll_retries > 4 * 3000  # polled every second for the hour


def test_chain_mode_breaks_a_gridlock_at_the_stuck_deadline(monkeypatch):
    """With a finite stuck time the cycle is broken by the first deadline,
    exactly as when polling."""
    scenario = _ring_gridlock()
    poll, poll_events, _ = _run_counting_retries(scenario, monkeypatch, stuck_time=5.0, retry_interval=0.5)
    chain, chain_events, chain_retries = _run_counting_retries(
        scenario, monkeypatch, stuck_time=5.0, retry_interval=0.5, blocked="chain")
    _assert_same_trajectories(poll, chain)
    assert chain_events == poll_events
    assert not chain.stuck.any()
    assert chain_retries < 60


def test_blocked_must_be_poll_or_chain():
    with pytest.raises(ValueError, match="blocked"):
        AgentSimParams(blocked="wait")


@pytest.mark.parametrize("retry", [0.0, -1.0, float("inf"), float("nan")])
def test_retry_interval_must_be_positive_and_finite(retry):
    with pytest.raises(ValueError, match="retry_interval"):
        AgentSimParams(retry_interval=retry)
