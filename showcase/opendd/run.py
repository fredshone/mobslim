"""The openDD pipeline, per roundabout: match every recording, simulate
(HermesSim) and compare the held-out recordings: openDD's own validation
split, ``r_1_val`` (rdb3 is in none of the ``r_1`` splits: its test split,
``r_C``). The roundabout rule (``replay.ROUNDABOUT``) was chosen on rdb1
and rdb2's training recordings, and never looked at rdb3-rdb7.

Matches are cached in ``showcase/output/opendd/match/<site>/`` (one parquet
file per recording and table), keyed by a hash of the matching code and the
view cuts, so changing either rematches everything.
"""

from __future__ import annotations

import hashlib
from functools import cache
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from showcase import replay
from showcase.opendd import load, match, network

OUT = Path(__file__).resolve().parents[1] / "output" / "opendd"
RESULTS = replay.RESULTS


def results_path(site: str) -> Path:
    return RESULTS / f"opendd_{site}.json"


def attribution(site: str) -> str:
    return ('Observed data: openDD (Breuer et al., 2020, "openDD: A Large-Scale Roundabout '
            f'Drone Dataset", ITSC), roundabout {site}, CC BY-ND 4.0.')


def available() -> list[str]:
    """The roundabouts in ``network.SITES`` whose full download is in ``data/opendd/``."""
    return [s for s in network.SITES if (load.DATA / s / f"trajectories_{s}_v3.sqlite").exists()]


@cache
def graph(site: str) -> load.LaneGraph:
    """The lane graph, split at the edges of the view."""
    return load.split_lanes(load.load_lanes(site), network.SITES[site].view_cuts)


def _key(site: str) -> str:
    h = hashlib.sha256()
    for m in (load, match):
        h.update(Path(m.__file__).read_bytes())
    h.update(repr(sorted(network.SITES[site].view_cuts.items())).encode())
    return h.hexdigest()[:12]


def _dir(rec: str) -> Path:
    site = load.site_of(rec)
    return OUT / "match" / site / _key(site)


def _match_one(rec: str) -> str:
    d = _dir(rec)
    if not (d / f"{rec}.veh.parquet").exists():
        g = graph(load.site_of(rec))
        links, veh = match.match(g, load.load_trajectories(rec, origin=g.origin))
        d.mkdir(parents=True, exist_ok=True)
        links.assign(rec=rec).to_parquet(d / f"{rec}.links.parquet")
        veh.assign(rec=rec).to_parquet(d / f"{rec}.veh.parquet")
    return rec


def matched(recs: list[str], processes: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(links, vehicles)`` of ``recs``, matching those not cached yet."""
    todo = [r for r in recs if not (_dir(r) / f"{r}.veh.parquet").exists()]
    if todo:
        print(f"matching {len(todo)} recordings")
        with Pool(min(processes, len(todo))) as p:
            for i, _ in enumerate(p.imap_unordered(_match_one, todo)):
                if i % 20 == 19:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    links = pd.concat([pd.read_parquet(_dir(r) / f"{r}.links.parquet") for r in recs], ignore_index=True)
    veh = pd.concat([pd.read_parquet(_dir(r) / f"{r}.veh.parquet") for r in recs], ignore_index=True)
    return links, veh


def split(site: str, name: str) -> list[str]:
    """A roundabout's recordings in an openDD split (``r_1_train``,
    ``r_1_val``; the files list every roundabout's)."""
    text = (load.DATA / site / "split_definition" / f"{name}.txt").read_text()
    return [x for x in text.split() if x.startswith(f"{site}_")]


def held_out(site: str) -> list[str]:
    return split(site, "r_1_val") or split(site, "r_C")


def evaluate(site: str, recs: list[str]) -> pd.DataFrame:
    """Time in view of every vehicle in ``recs`` (``compare.trip_times``), with a ``rec`` column."""
    from showcase.opendd import compare, demand

    g = graph(site)
    net = network.build(g)
    out = []
    for i, rec in enumerate(recs):
        links, _ = matched([rec])
        trips = demand.trips(g, links)
        sim = replay.simulate(net, demand.population(net, trips))
        out.append(compare.trip_times(sim, trips).assign(rec=rec))
        print(f"  {i + 1}/{len(recs)} {rec}", flush=True)
    return pd.concat(out, ignore_index=True)


def meta(site: str, val: list[str]) -> dict:
    s = network.SITES[site]
    _, veh = matched(load.recordings(site))
    return dict(site=site, recordings=len(load.recordings(site)), held_out=val, shown=s.shown,
                tracked=int(len(veh)), matched=int(veh.matched.sum()),
                cuts={str(k): v for k, v in s.view_cuts.items()}, outside=sorted(s.outside),
                attribution=attribution(site))


def animation(site: str) -> None:
    """Observed and simulated vehicles of the site's ``shown`` recording."""
    from showcase.opendd import demand, export

    g = graph(site)
    rec = network.SITES[site].shown
    links, _ = matched([rec])
    tr = load.load_trajectories(rec, origin=g.origin)
    trips = demand.trips(g, links)
    net = network.build(g)
    sim = replay.simulate(net, demand.population(net, trips))
    paths = export.write(g, rec, net, sim, trips, links, tr, attribution(site))
    print("wrote", *paths)


def main(sites: list[str] | None = None) -> None:
    """``python -m showcase opendd``: everything, from the data to the docs,
    for every roundabout downloaded."""
    import json

    for site in sites or available():
        print(f"== {site}")
        matched(load.recordings(site))
        val = held_out(site)
        print(f"evaluating {len(val)} held-out recordings")
        tt = evaluate(site, val)
        R = dict(meta=meta(site, val), headline=[replay.result(f"Roundabout {site}", tt)])
        RESULTS.mkdir(parents=True, exist_ok=True)
        results_path(site).write_text(json.dumps(R, indent=1) + "\n")
        print("wrote", results_path(site), R["headline"])
        animation(site)
    replay.headline()
