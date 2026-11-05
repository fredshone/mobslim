"""The walks implied by pt legs, computed for a whole PlanBatch with numpy.

A pt leg with rides ``r0 .. r1-1`` walks: to ride ``r0``'s board stop from
the origin activity, between rides (alight stop of ``r - 1`` to board stop of
``r``), and from the last alight stop to the destination activity. A pt leg
without rides walks from origin to destination. Walks are beeline between
coordinates x ``TransitConfig.walk.beeline_distance_factor`` /
``speed``; distances include the factor, as for teleported legs.
"""

from dataclasses import dataclass

import numpy as np

from mobslim.planbatch import PlanBatch
from mobslim.routing.teleport import teleport_many


@dataclass(frozen=True)
class PtWalks:
    before_time: np.ndarray  # per ride: walk to its board stop [s]
    before_distance: np.ndarray  # per ride [m]
    after_time: np.ndarray  # per leg: walk to the destination (pt legs; 0 otherwise) [s]
    after_distance: np.ndarray  # per leg [m]


def pt_walks(plans: PlanBatch, act_xy: np.ndarray, transit) -> PtWalks:
    """Walk times and distances of every pt leg's walks (``act_xy`` resolved)."""
    s, walk = transit.schedule, transit.config.walk
    stop_xy = s.stop_xy
    is_pt = plans.mode_legs(transit.config.passenger_modes)
    origin = plans.leg_origin_act()
    ro = plans.ride_offsets
    board = plans.ride_board_stop.astype(np.int64)
    alight = plans.ride_alight_stop.astype(np.int64)
    n = plans.n_rides
    leg_of_ride = plans.leg_of_ride()
    frm = np.empty((n, 2))
    if n:
        first = np.zeros(n, dtype=bool)
        first[ro[:-1][np.diff(ro) > 0]] = True
        frm[first] = act_xy[origin[leg_of_ride[first]]]
        prev = np.flatnonzero(~first)
        frm[prev] = stop_xy[s.rs_stop[alight[prev - 1]]]
    to = stop_xy[s.rs_stop[board]] if n else np.empty((0, 2))
    before_t, before_d = teleport_many(frm, to, walk)

    legs = np.flatnonzero(is_pt)
    has_rides = np.diff(ro)[legs] > 0
    start = act_xy[origin[legs]].copy()
    with_rides = legs[has_rides]
    if len(with_rides):
        start[has_rides] = stop_xy[s.rs_stop[alight[ro[with_rides + 1] - 1]]]
    after_t, after_d = np.zeros(plans.n_legs), np.zeros(plans.n_legs)
    after_t[legs], after_d[legs] = teleport_many(start, act_xy[origin[legs] + 1], walk)
    return PtWalks(before_t, before_d, after_t, after_d)
