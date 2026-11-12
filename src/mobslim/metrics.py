"""Per-iteration summary metrics computed from scores and trajectories."""

import numpy as np

from mobslim.population import Population
from mobslim.trajectories import Trajectories


def iteration_metrics(iteration: int, population: Population, trajectories: Trajectories) -> dict:
    """Score statistics (MATSim's ``scorestats``) plus per-mode leg statistics."""
    row = score_metrics(iteration, population)
    row["stuck"] = int(trajectories.stuck.sum())

    # per mode, over completed legs
    b = trajectories.plans
    travel = trajectories.leg_arrival - trajectories.leg_departure
    done = ~np.isnan(travel)
    codes = b.leg_mode[done]
    count = np.bincount(codes, minlength=len(b.modes))
    total_time = np.bincount(codes, weights=travel[done], minlength=len(b.modes))
    total_dist = np.bincount(codes, weights=trajectories.leg_distance[done], minlength=len(b.modes))
    for code in sorted(range(len(b.modes)), key=b.modes.__getitem__):
        if count[code]:
            mode = b.modes[code]
            row[f"{mode}_legs"] = int(count[code])
            row[f"{mode}_travel_time"] = float(total_time[code] / count[code])
            row[f"{mode}_distance"] = float(total_dist[code] / count[code])
    if trajectories.transit is not None:
        row.update(pt_metrics(trajectories))
    return row


def score_metrics(iteration: int, population: Population) -> dict:
    """Score statistics (MATSim's ``scorestats``) alone."""
    scores = population.scores  # (n_persons, n_slots), NaN = unscored / empty
    scored = ~np.isnan(scores).all(axis=1)  # persons with at least one score
    rows = scores[scored]
    best, worst, average = (np.nanmax(rows, axis=1), np.nanmin(rows, axis=1),
                            np.nanmean(rows, axis=1))
    return {
        "iteration": iteration,
        "score_executed": _mean(population.selected_scores()),
        "score_best": _mean(best),
        "score_worst": _mean(worst),
        "score_average": _mean(average),
        "plans_per_person": float(population.n_plans.mean()) if len(population) else 0.0,
    }


def pt_metrics(tr: Trajectories) -> dict:
    """Transit metrics: boardings, denied boardings, waits, loads, delays."""
    boarded = ~np.isnan(tr.ride_board)
    rode = ~np.isnan(tr.ride_alight)
    b = tr.plans
    pt_legs = b.mode_legs(tr.transit_modes) & ~np.isnan(tr.leg_arrival)
    n_rides = np.diff(b.ride_offsets)[pt_legs]
    load = tr.pt_load()
    tt = tr.transit
    capacity = np.repeat(tt.transit.trip_capacity, np.diff(tt.trip_stop_offsets))
    delay = tt.departure_delay()
    delay = delay[~np.isnan(delay)]
    return {
        "pt_boardings": int(boarded.sum()),
        "pt_denied_boardings": int(tr.denied_boardings().sum()),
        "pt_wait_time": _mean((tr.ride_board - tr.ride_wait_start)[boarded]),
        "pt_in_vehicle_time": _mean((tr.ride_alight - tr.ride_board)[rode]),
        "pt_transfers": _mean(np.maximum(n_rides - 1, 0)),
        "pt_max_load": int(load.max()) if len(load) else 0,
        "pt_max_load_factor": float((load / capacity).max()) if len(load) else 0.0,
        "pt_trip_delay": _mean(delay),
    }


def _mean(values) -> float:
    """Mean ignoring NaN; NaN if there is nothing to average."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else float("nan")
