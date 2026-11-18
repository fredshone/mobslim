"""MATSim's pt routes as mobslim rides (``verification.pt_routes``)."""

from types import SimpleNamespace

import numpy as np

from mobslim.ids import IdMap
from verification.pt_routes import _route_stop_after


def _schedule():
    """One route serving stop A twice (A, B, A, C), with trips at 1000 and 2000 s."""
    stops = IdMap()
    for s in ("A", "B", "C"):
        stops.add(s)
    return SimpleNamespace(
        route_index=lambda line, route: 0,
        stop_ids=stops,
        route_stop_offsets=np.array([0, 4]),
        rs_stop=np.array([0, 1, 0, 2]),
        rs_arrival=np.array([0.0, 60.0, 120.0, 180.0]),
        rs_departure=np.array([30.0, 90.0, 150.0, 210.0]),
        route_dep_offsets=np.array([0, 2]),
        dep_time=np.array([1000.0, 2000.0]),
    )


def test_boarding_visit_is_the_one_matsim_boarded_at():
    s = _schedule()
    assert _route_stop_after(s, "l", "r", "A") == 0  # no time: the first visit
    # MATSim's boardingTime is seconds off a trip's arrival or departure there
    assert _route_stop_after(s, "l", "r", "A", departure=2000.0 + 150.0 - 8) == 2
    assert _route_stop_after(s, "l", "r", "A", departure=1000.0 + 120.0 + 3) == 2
    assert _route_stop_after(s, "l", "r", "A", departure=1000.0 + 30.0) == 0
    assert _route_stop_after(s, "l", "r", "A", departure=9000.0) == 0  # near no trip
    assert _route_stop_after(s, "l", "r", "A", after=0) == 2
    assert _route_stop_after(s, "l", "r", "C", after=2) == 3
