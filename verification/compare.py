"""Layer-2 metrics: how close one run's traffic is to a reference run's.

Every function takes tables from :mod:`verification.events_io` for the
reference (MATSim) and the test (mobslim) side and returns a flat dict of
numbers. ``band`` grades a metric green / amber / red against ``BANDS``.
The thresholds are a first guess, to be tuned once we've seen the gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from verification import events_io

HOUR = 3600.0


def geh(m: np.ndarray, c: np.ndarray) -> np.ndarray:
    """GEH statistic for hourly flows (0 where both are 0)."""
    m, c = np.asarray(m, float), np.asarray(c, float)
    s = m + c
    return np.sqrt(np.divide(2 * (m - c) ** 2, s, out=np.zeros_like(s), where=s > 0))


def hourly_volumes(trav: pd.DataFrame) -> pd.Series:
    """Vehicles entering each link per hour: Series indexed by (link, hour)."""
    hour = (trav["enter"] // HOUR).astype(int)
    return trav.groupby([trav["link"], hour]).size().rename("volume")


def compare_volumes(ref: pd.DataFrame, test: pd.DataFrame) -> dict:
    r, t = hourly_volumes(ref), hourly_volumes(test)
    both = pd.concat([r.rename("ref"), t.rename("test")], axis=1).fillna(0.0)
    g = geh(both["test"], both["ref"])
    diff = both["test"] - both["ref"]
    ss_tot = ((both["ref"] - both["ref"].mean()) ** 2).sum()
    daily = both.groupby(level=0).sum()
    return {
        "link_hours": int(len(both)),
        "geh_lt5_share": float((g < 5).mean()),
        "geh_max": float(g.max()),
        "volume_rmse": float(np.sqrt((diff ** 2).mean())),
        "volume_r2": float(1 - (diff ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "volume_abs_diff_rel": float(diff.abs().sum() / both["ref"].sum()),
        "daily_volume_max_abs_diff": float((daily["test"] - daily["ref"]).abs().max()),
        "traversals_ref": int(len(ref)),
        "traversals_test": int(len(test)),
    }


def compare_link_times(ref: pd.DataFrame, test: pd.DataFrame, bin_size: float = 900.0) -> dict:
    """Mean traversal time per (link, entry bin) on both sides, over the bins both have."""
    def table(trav):
        done = trav.dropna(subset=["leave"])
        b = (done["enter"] // bin_size).astype(int)
        return (done["leave"] - done["enter"]).groupby([done["link"], b]).mean()

    both = pd.concat([table(ref).rename("ref"), table(test).rename("test")], axis=1).dropna()
    d = both["test"] - both["ref"]
    per_traversal = {}
    for name, trav in (("ref", ref), ("test", test)):
        done = trav.dropna(subset=["leave"])
        per_traversal[name] = float((done["leave"] - done["enter"]).mean())
    return {
        "link_bins": int(len(both)),
        "link_time_mae_s": float(d.abs().mean()),
        "link_time_max_abs_s": float(d.abs().max()),
        "link_time_mean_signed_s": float(d.mean()),
        "traversal_time_mean_ref_s": per_traversal["ref"],
        "traversal_time_mean_test_s": per_traversal["test"],
    }


def _quantile_distance(a: np.ndarray, b: np.ndarray, n: int = 101) -> float:
    """1-D Wasserstein distance approximated on ``n`` quantiles."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0, 1, n)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def compare_legs(ref: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Per-trip (matched by person and trip index) and distribution-level
    door-to-door travel times (``events_io.trips``; metrics keep the ``leg_``
    names, a trip is one leg in mobslim)."""
    tt_r = ref["arrival"] - ref["departure"]
    tt_t = test["arrival"] - test["departure"]
    m = ref.assign(tt=tt_r).merge(test.assign(tt=tt_t), on=["person", "k"],
                                  suffixes=("_ref", "_test"), how="outer")
    both = m.dropna(subset=["tt_ref", "tt_test"])
    d = (both["tt_test"] - both["tt_ref"]).abs()
    a = (both["arrival_test"] - both["arrival_ref"]).abs()
    mean_r, mean_t = float(tt_r.mean()), float(tt_t.mean())
    return {
        "legs_ref": int(len(ref)),
        "legs_test": int(len(test)),
        "legs_unfinished_ref": int(tt_r.isna().sum()),
        "legs_unfinished_test": int(tt_t.isna().sum()),
        "leg_tt_mean_ref_s": mean_r,
        "leg_tt_mean_test_s": mean_t,
        "leg_tt_mean_rel_diff": (mean_t - mean_r) / mean_r if mean_r > 0 else float("nan"),
        "leg_tt_quantile_dist_s": _quantile_distance(tt_r.dropna().to_numpy(), tt_t.dropna().to_numpy()),
        "leg_tt_abs_diff_median_s": float(d.median()),
        "leg_tt_abs_diff_p95_s": float(d.quantile(0.95)),
        "leg_tt_abs_diff_max_s": float(d.max()),
        "arrival_abs_diff_median_s": float(a.median()),
        "arrival_abs_diff_p95_s": float(a.quantile(0.95)),
    }


def compare_modes(ref: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Mean trip travel time per mode (relative difference)."""
    out = {}
    for mode in sorted(set(ref["mode"]) | set(test["mode"])):
        r = (ref["arrival"] - ref["departure"])[ref["mode"] == mode].mean()
        t = (test["arrival"] - test["departure"])[test["mode"] == mode].mean()
        out[f"trip_tt_mean_rel_diff_{mode}"] = float((t - r) / r) if r else float("nan")
    return out


def compare_network(ref: pd.DataFrame, test: pd.DataFrame, link_length: dict) -> dict:
    def stats(trav):
        done = trav.dropna(subset=["leave"])
        vh = float((done["leave"] - done["enter"]).sum() / HOUR)
        vkm = float(trav["link"].map(link_length).sum() / 1000.0)
        return vh, vkm, float(done["leave"].max()) if len(done) else float("nan")

    (vh_r, vkm_r, last_r), (vh_t, vkm_t, last_t) = stats(ref), stats(test)
    return {
        "veh_hours_ref": vh_r, "veh_hours_test": vh_t,
        "veh_hours_rel_diff": (vh_t - vh_r) / vh_r if vh_r else float("nan"),
        "veh_km_ref": vkm_r, "veh_km_test": vkm_t,
        "veh_km_rel_diff": (vkm_t - vkm_r) / vkm_r if vkm_r else float("nan"),
        "last_link_exit_diff_s": last_t - last_r,
    }


def compare_transit(ref: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Stop calls matched by (line, route, departure, stop, visit)."""
    if ref.empty and test.empty:
        return {}
    key = ["line", "route", "departure", "stop", "visit"]
    m = ref.merge(test, on=key, how="outer", suffixes=("_ref", "_test"), indicator=True)
    both = m[m["_merge"] == "both"]
    arr = (both["arrival_test"] - both["arrival_ref"]).abs()
    dep = (both["departure_time_test"] - both["departure_time_ref"]).abs()
    board = (both["boardings_test"] - both["boardings_ref"]).abs()
    total = both["boardings_ref"].sum()
    return {
        "stop_calls_ref": int(len(ref)), "stop_calls_test": int(len(test)),
        "stop_calls_unmatched": int((m["_merge"] != "both").sum()),
        "stop_arrival_abs_diff_mean_s": float(arr.mean()),
        "stop_arrival_abs_diff_max_s": float(arr.max()),
        "stop_departure_abs_diff_mean_s": float(dep.mean()),
        "stop_departure_abs_diff_max_s": float(dep.max()),
        "boardings_ref": int(ref["boardings"].sum()), "boardings_test": int(test["boardings"].sum()),
        "boardings_abs_diff_rel": float(board.sum() / total) if total else float("nan"),
        "boardings_abs_diff_max": int(board.max()) if len(board) else 0,
    }


def compare_runs(ref_events: pd.DataFrame, test_events: pd.DataFrame, link_length: dict) -> dict:
    """All layer-2 metrics for one (reference, test) pair of event tables."""
    ref_trav, test_trav = events_io.link_traversals(ref_events), events_io.link_traversals(test_events)
    out = {}
    out.update(compare_volumes(ref_trav, test_trav))
    out.update(compare_link_times(ref_trav, test_trav))
    out.update(compare_legs(events_io.trips(ref_events), events_io.trips(test_events)))
    out.update(compare_modes(events_io.trips(ref_events), events_io.trips(test_events)))
    out.update(compare_network(ref_trav, test_trav, link_length))
    out.update(compare_transit(events_io.transit_stops(ref_events),
                               events_io.transit_stops(test_events)))
    out["stuck_ref"] = len(events_io.stuck_persons(ref_events))
    out["stuck_test"] = len(events_io.stuck_persons(test_events))
    return out


def compare_scores(ref: dict, test: dict, stuck: set, n_worst: int = 5) -> tuple[dict, list]:
    """Per-person scores (MATSim ``ref`` vs mobslim ``test``), by person id."""
    ids = sorted(set(ref) | set(test))
    r = np.array([ref.get(i, np.nan) for i in ids])
    t = np.array([test.get(i, np.nan) for i in ids])
    both = ~np.isnan(r) & ~np.isnan(t)
    d = t[both] - r[both]
    ad = np.abs(d)
    order = np.argsort(-np.where(both, np.abs(t - r), -1.0))
    worst = [ids[i] for i in order[:n_worst] if both[i] and abs(t[i] - r[i]) > 1e-9]
    return {
        "persons": len(ids),
        "persons_compared": int(both.sum()),
        "persons_missing": int((~both).sum()),
        "stuck": len(stuck),
        "score_mean_ref": float(np.mean(r[both])),
        "score_mean_test": float(np.mean(t[both])),
        "score_mean_diff": float(d.mean()),
        "score_abs_diff_mean": float(ad.mean()),
        "score_abs_diff_max": float(ad.max()) if len(ad) else 0.0,
        "share_within_1e-6": float((ad <= 1e-6).mean()),
        "share_within_0.1": float((ad <= 0.1).mean()),
    }, worst


# metric -> (green, amber, higher_is_better); compared on abs(value) unless higher_is_better.
BANDS = {
    "geh_lt5_share": (0.95, 0.85, True),
    "volume_r2": (0.95, 0.85, True),
    "volume_abs_diff_rel": (0.05, 0.15, False),
    "link_time_mae_s": (2.0, 10.0, False),
    "leg_tt_mean_rel_diff": (0.02, 0.05, False),
    "leg_tt_quantile_dist_s": (10.0, 60.0, False),
    "veh_hours_rel_diff": (0.02, 0.05, False),
    "veh_km_rel_diff": (0.001, 0.01, False),
    "stop_arrival_abs_diff_mean_s": (2.0, 10.0, False),
    "stop_departure_abs_diff_mean_s": (2.0, 10.0, False),
    "boardings_abs_diff_rel": (0.02, 0.1, False),
    "score_abs_diff_max": (1e-6, 0.1, False),
    "score_mean_diff": (1e-6, 0.01, False),
    "share_within_1e-6": (0.999, 0.95, True),
}


def band(metric: str, value: float) -> str:
    if metric not in BANDS or value is None or np.isnan(value):
        return ""
    green, amber, higher = BANDS[metric]
    if higher:
        return "green" if value >= green else "amber" if value >= amber else "red"
    v = abs(value)
    return "green" if v <= green else "amber" if v <= amber else "red"
