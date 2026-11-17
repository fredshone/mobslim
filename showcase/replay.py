"""What the three replays of observed traffic (openDD, TGSIM I-395, TGSIM
I-90/I-94) share.

* :func:`simulate`: HermesSim on observed demand, as a table of link entry
  and exit times. Every replay runs one mobsim, HermesSim, on its own
  default clock, with MobSlim's default cell size and stuck time.
* The link rules (:data:`ROUNDABOUT`, :data:`FREEWAY`): a free speed and a
  capacity per lane for each type of road, rules of thumb, the same at
  every site. Nothing is fitted.
* The animation data for ``replay-anim.js`` (:func:`place`,
  :func:`segments`, :func:`write`): vehicles at :data:`FPS` frames per
  second, placed from link entry and exit times alone. A vehicle moves
  along its link at the free speed, then waits at the end, :data:`SPACING`
  behind the one in front (in its lane, where a link has several). Outside
  the camera's view, vehicles are drawn approaching and leaving at the free
  speed (:func:`lead_in`, :func:`coast`), and those held back on a virtual
  entry link are drawn queued back along the road (:func:`waiting`), so
  the whole of every trip is on screen.
* :func:`result` and :func:`headline`: the one table on the Correctness
  page, from every replay's saved results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mobslim import Scenario
from mobslim.events import (
    EventLog,
    EventsManager,
    LinkEnter,
    LinkLeave,
    PersonArrival,
)
from mobslim.mobsim import HermesSim, HermesSimParams
from mobslim.network import Network

ROOT = Path(__file__).resolve().parents[1]
DOCS_STATIC = ROOT / "docs" / "_static" / "showcase"
GEN = ROOT / "docs" / "_generated"
RESULTS = ROOT / "showcase" / "results"

MOBSIM = "HermesSim"
EFFECTIVE_CELL = 7.5  # [m] per vehicle: storage = lanes * length / 7.5 (MobSlim default)
STUCK_TIME = 10.0  # [s] (MobSlim and MATSim default)
VIRTUAL = dict(length=1.0, freespeed=1000.0, capacity=36000.0, lanes=1000.0)  # ~0 s, 133 vehicles



@dataclass(frozen=True)
class Rule:
    """A type of road's free speed [m/s] and capacity [veh/h per lane]."""

    freespeed: float
    capacity: float


ROUNDABOUT = Rule(25 / 3.6, 1800.0)  # 25 km/h on every lane, the ring and the arms alike
FREEWAY = Rule(55 * 0.44704, 1800.0)  # the posted 55 mph

FPS = 5
SPACING = 7.5  # [m] between queued vehicles in a lane
TAIL = 60.0  # [m] of road drawn beyond the view at each entry and exit


# ------------------------------------------------------------------ simulation


def simulate(net: Network, pop) -> pd.DataFrame:
    """One row per person and link entered: ``id, seq, lane, t_in, t_out``
    (``lane``: the link id)."""
    log = EventLog()
    HermesSim(HermesSimParams(stuck_time=STUCK_TIME, effective_cell_size=EFFECTIVE_CELL)).run(
        Scenario(net, pop), EventsManager([log]))
    open_, rows = {}, []
    for e in log.events:
        if isinstance(e, LinkEnter):
            if e.person in open_:  # entered the next link without a leave event
                rows.append(open_.pop(e.person) + [e.time])
            open_[e.person] = [e.person, e.link, e.time]
        elif isinstance(e, (LinkLeave, PersonArrival)) and e.person in open_:
            rows.append(open_.pop(e.person) + [e.time])
    df = pd.DataFrame(rows, columns=["person", "link", "t_in", "t_out"])
    df["id"] = [pop.ids.id(p) for p in df.person]
    df["lane"] = [net.link_ids.id(k) for k in df.link]
    df["seq"] = df.groupby("id").cumcount()
    return df[["id", "seq", "lane", "t_in", "t_out"]]


# ------------------------------------------------------------------ geometry


def along(line: np.ndarray, s) -> np.ndarray:
    """Points ``s`` metres along polyline ``line`` (clamped to its ends)."""
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    s = np.atleast_1d(s)
    return np.column_stack([np.interp(s, cum, line[:, 0]), np.interp(s, cum, line[:, 1])])


def length(line: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def offset(line: np.ndarray, s, lane: float) -> np.ndarray:
    """Points ``s`` along ``line``, shifted ``lane`` metres to the right of travel."""
    s = np.atleast_1d(s)
    p = along(line, s)
    d = along(line, s + 0.5) - along(line, s - 0.5)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
    return p + np.column_stack([d[:, 1], -d[:, 0]]) * lane


# ------------------------------------------------------------------ placing vehicles


def place(d: pd.DataFrame, L: float, fftime: float, times: np.ndarray, lanes: int = 1,
          slot: np.ndarray | None = None) -> list[tuple[int, object, int, float]]:
    """``(frame, id, lane, s)`` for every vehicle on one link at every frame
    of ``times`` it is on it. ``d``: ``id, t_in, t_out``. A vehicle moves at
    the free speed, then waits at the end behind the one in front in its
    lane (``slot`` per row, else the link's lanes in turn, in entry order)."""
    d = d.sort_values("t_in")
    t_in, t_out, ids = d.t_in.to_numpy(), d.t_out.to_numpy(), d.id.to_numpy()
    if slot is None:
        slot = np.arange(len(d)) % lanes
    out = []
    if not len(d):
        return out
    lo, hi = np.searchsorted(times, t_in.min()), np.searchsorted(times, t_out.max())
    for f in range(lo, min(hi + 1, len(times))):
        t = times[f]
        on = np.flatnonzero((t_in <= t) & (t_out > t))
        room = np.full(lanes, L)
        for j in on:  # in order of entry: the first is in front
            p = max(0.0, min(L * min(1.0, (t - t_in[j]) / fftime), room[slot[j]]))
            room[slot[j]] = p - SPACING
            out.append((f, ids[j], int(slot[j]), p))
    return out


def lead_in(tail: np.ndarray, speed: float, t_arrive: float, times: np.ndarray):
    """``(frame, s)`` along ``tail`` (which runs *back* from the view edge)
    for a vehicle arriving at the edge at ``t_arrive``, drawn approaching at
    ``speed``: ``s`` is its distance before the edge."""
    L = length(tail)
    if L <= 0 or speed <= 0:
        return []
    t0 = t_arrive - L / speed
    lo, hi = np.searchsorted(times, t0), np.searchsorted(times, t_arrive)
    return [(f, (t_arrive - times[f]) * speed) for f in range(lo, min(hi, len(times)))]


def coast(tail: np.ndarray, speed: float, t_leave: float, times: np.ndarray):
    """``(frame, s)`` along ``tail`` (onward from the view edge) for a
    vehicle leaving the edge at ``t_leave`` at ``speed``."""
    L = length(tail)
    if L <= 0 or speed <= 0:
        return []
    lo, hi = np.searchsorted(times, t_leave), np.searchsorted(times, t_leave + L / speed)
    return [(f, (times[f] - t_leave) * speed) for f in range(lo, min(hi, len(times)))]


def waiting(d: pd.DataFrame, times: np.ndarray, lanes: int = 1):
    """``(frame, id, lane, s)``: vehicles held on a virtual entry link
    (``d``: ``id, t_in, t_out``), queued back from the view edge in order
    of arrival, ``s`` metres before it, ``lanes`` abreast."""
    d = d.sort_values("t_in")
    t_in, t_out, ids = d.t_in.to_numpy(), d.t_out.to_numpy(), d.id.to_numpy()
    out = []
    if not len(d):
        return out
    lo, hi = np.searchsorted(times, t_in.min()), np.searchsorted(times, t_out.max())
    for f in range(lo, min(hi + 1, len(times))):
        t = times[f]
        on = np.flatnonzero((t_in <= t) & (t_out > t))
        for k, j in enumerate(on):
            out.append((f, ids[j], k % lanes, (k // lanes) * SPACING))
    return out


def segments(pos: dict) -> list[dict]:
    """``pos``: ``{id: {frame: (u, v)}}`` -> ``[{f, xy}]``, one per run of
    consecutive frames, positions in decimetres (a vehicle can be missing
    from a frame between two links; a gap of one frame is interpolated)."""
    out = []
    for fr in pos.values():
        f = np.array(sorted(fr))
        xy = np.array([fr[x] for x in f], dtype=float)
        full = np.arange(f[0], f[-1] + 1)
        xy = np.column_stack([np.interp(full, f, xy[:, 0]), np.interp(full, f, xy[:, 1])])
        keep = np.isin(full, f) | np.isin(full - 1, f) & np.isin(full + 1, f)
        for run in np.split(np.arange(len(full)), np.flatnonzero(np.diff(keep.astype(int)) != 0) + 1):
            if len(run) and keep[run[0]]:
                out.append({"f": int(full[run[0]]), "xy": np.round(xy[run] * 10).astype(int).ravel().tolist()})
    return out


def counts(windows, times: np.ndarray) -> list[int]:
    """Per frame, how many of ``windows`` (``(t0, t1)`` pairs) contain it."""
    w = np.array(list(windows), dtype=float).reshape(-1, 2)
    if not len(w):
        return [0] * len(times)
    a = np.searchsorted(times, w[:, 0])
    b = np.searchsorted(times, w[:, 1])
    c = np.zeros(len(times) + 1, dtype=int)
    np.add.at(c, a, 1)
    np.add.at(c, b, -1)
    return np.cumsum(c)[:len(times)].tolist()


def road(xy: np.ndarray, width: float, view: bool) -> dict:
    """A drawn road: its centreline (decimetres), width [m], in view or not."""
    return {"xy": np.round(np.asarray(xy, dtype=float) * 10).astype(int).ravel().tolist(),
            "w": round(float(width), 1), "view": bool(view)}


def write(key: str, data: dict, observed: list[dict]) -> tuple[Path, Path]:
    """``replay-<key>-sim.js`` (roads, the simulated run, the in-view
    counts) and ``replay-<key>-obs.js`` (the observed tracks, built locally
    and git-ignored). Each adds its replay to ``window.MOBSLIM_REPLAY`` (or
    ``_OBS``) under ``key``."""
    DOCS_STATIC.mkdir(parents=True, exist_ok=True)
    sim_path, obs_path = DOCS_STATIC / f"replay-{key}-sim.js", DOCS_STATIC / f"replay-{key}-obs.js"
    for path, var, d in ((sim_path, "MOBSLIM_REPLAY", {"fps": FPS, **data}),
                         (obs_path, "MOBSLIM_REPLAY_OBS", {"tracks": observed})):
        path.write_text(f"(window.{var} = window.{var} || {{}})[{json.dumps(key)}] = "
                        + json.dumps(d, separators=(",", ":")) + ";\n")
    return sim_path, obs_path


# ------------------------------------------------------------------ the Correctness page


def result(what: str, tt: pd.DataFrame) -> dict:
    """One row of the headline table: ``tt`` has, per vehicle seen coming
    into view and leaving it, ``t_dep`` and the observed and simulated
    ``t_exit``."""
    tt = tt.dropna(subset=["t_exit", "t_exit_sim"])
    obs, sim = tt.t_exit - tt.t_dep, tt.t_exit_sim - tt.t_dep
    return dict(what=what, trips=len(tt), obs=round(float(obs.mean()), 1), sim=round(float(sim.mean()), 1),
                error=round(float((sim - obs).abs().median()), 1))


REPLAYS = (*(f"opendd_rdb{i}" for i in range(1, 8)), "tgsim", "tgsim_i90")


def headline() -> None:
    """``docs/_generated/correctness_headline.md``: the ``headline`` rows of
    every replay's ``showcase/results/<replay>.json``; and
    ``correctness_groups.md``: their ``groups`` rows (I-90/I-94's lane groups)."""
    tables = {"headline": [], "groups": []}
    for name in REPLAYS:
        p = RESULTS / f"{name}.json"
        if p.exists():
            R = json.loads(p.read_text())
            for k, rows in tables.items():
                rows += [[h["what"], f"{h['trips']:,}", f"{h['obs']:.1f}", f"{h['sim']:.1f}", f"{h['error']:.1f}"]
                         for h in R.get(k, [])]
    head = ["Replay", "Vehicles", "Time in view, observed (s)", "Simulated (s)", "Typical error per vehicle (s)"]
    GEN.mkdir(parents=True, exist_ok=True)
    for k, rows in tables.items():
        out = ["| " + " | ".join(head) + " |", "|:--|--:|--:|--:|--:|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        (GEN / f"correctness_{k}.md").write_text("\n".join(out) + "\n")
