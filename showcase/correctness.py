"""Run the corridor on Java MATSim and every MobSlim mobsim; export animation data.

MATSim (QSim and Hermes) runs the scenario as written, routing the plans
itself. Every MobSlim mobsim then runs MATSim's routed plans (car routes and
pt routes), so all of them simulate exactly the same trips. Every MobSlim mobsim runs
on MATSim's clock (whole seconds, a one-second step at each junction, link
times rounded down), the setting that makes it closest to MATSim.

Every run writes MATSim ``events_v1`` XML. One parser
(:func:`read_run`) turns any of them into the same tables, so the animation
and the numbers below it are built the same way for every simulation.

``python -m showcase correctness`` runs both variants of the corridor
(congested and free-flow), writes ``docs/_static/showcase/corridor-data.js``
and prints a summary table per variant.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mobslim.config.matsim import MatsimConfig, mobsim_params
from mobslim.events import EventsManager, EventsXMLWriter
from showcase import corridor, dsim

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(__file__).parent / "output" / "corridor"
DOCS_STATIC = ROOT / "docs" / "_static" / "showcase"

# (key, label, family)
# (key, label, MATSim build)
MATSIM_SIMS = [("matsim_qsim", "MATSim QSim", "2025.0"), ("matsim_hermes", "MATSim Hermes", "2025.0"),
               ("matsim_dsim", "MATSim DSim", dsim.DSIM_MATSIM)]
MOBSLIM_SIMS = [("qsim", "QSim"), ("agentsim", "AgentSim"), ("hermes", "HermesSim"),
                ("psim", "PSim"), ("dsim", "DSim")]
# MATSim's clock, for every MobSlim mobsim (each has its own default; see docs/simulation/design-detail.md)
CLOCKS = {"matsim": dict(continuous_time=False, junction_delays=True, rounding="down")}


def _inputs_hash(variant: str) -> str:
    h = hashlib.sha256()
    for p in sorted(corridor.directory(variant).iterdir()):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:16]


def run_matsim_side(variant: str, force: bool = False) -> dict[str, Path]:
    from verification.matsim import run_matsim
    from verification.runner import _hermes_config

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "inputs.hash"
    force = force or not stamp.exists() or stamp.read_text() != _inputs_hash(variant)
    cfg = MatsimConfig.load(corridor.directory(variant) / "config.xml")
    out = {"matsim_qsim": run_matsim(cfg, out_dir / "matsim_qsim", force=force),
           "matsim_hermes": run_matsim(_hermes_config(cfg), out_dir / "matsim_hermes", force=force)}
    out["matsim_dsim"] = dsim.run(out_dir / "matsim_qsim.config.xml", out_dir / "matsim_dsim",
                                  threads=1, force=force)
    stamp.write_text(_inputs_hash(variant))
    return {k: v / "output_events.xml.gz" for k, v in out.items()}


def run_mobslim_side(variant: str) -> dict[str, Path]:
    from verification.cases import MOBSIMS
    from verification.runner import mobslim_scenario

    out_dir = OUT_ROOT / variant
    cfg = MatsimConfig.load(corridor.directory(variant) / "config.xml")
    kwargs, _ = mobsim_params(cfg)
    plans = out_dir / "matsim_qsim" / "output_plans.xml.gz"
    scenario = mobslim_scenario(cfg, plans, vehicles="network")
    paths = {}
    for clock, timing in CLOCKS.items():
        for key, _ in MOBSLIM_SIMS:
            path = out_dir / f"{key}.{clock}.events.xml.gz"
            sim = MOBSIMS[key](**kwargs, **timing)
            events = EventsManager([EventsXMLWriter(path, scenario.network, scenario.population,
                                                    scenario.transit, vehicle_per_trip=True)])
            events.reset(0)
            sim.run(scenario, events)
            events.finish()
            paths[f"{key}.{clock}"] = path
    return paths


# ------------------------------------------------------------------ parsing


def _plans(variant: str) -> dict[str, dict]:
    """person -> mode, home/work link and coordinates (from the scenario)."""
    return {p["id"]: p for p in corridor.persons(variant)}


def read_run(path: Path, variant: str) -> dict:
    """Vehicles (link segments and stop calls) and pt passengers of one run.

    A vehicle's segment opens when it enters a link, or when it is placed at
    the end of its first link (MATSim's ``vehicle enters traffic``; MobSlim's
    car ``departure`` and ``TransitDriverStarts``), and closes at its next
    link event or when it leaves traffic.
    """
    plans = _plans(variant)
    vehicles: dict[str, dict] = {}
    open_seg: dict[str, list] = {}
    passengers: dict[str, dict] = {}
    car_legs: dict[str, list] = {}
    bus_of_driver: dict[str, str] = {}

    def veh(v):
        return v.split("#")[0] if v else v

    def vrec(v, kind):
        return vehicles.setdefault(v, {"kind": kind, "links": [], "stops": []})

    def close(v, t):
        seg = open_seg.pop(v, None)
        if seg is not None:
            seg[2] = t

    def open_(v, link, t, kind, start=False):
        seg = open_seg.get(v)
        if seg is not None and seg[3] == 1 and seg[0] == link:
            return  # MobSlim also enters the first link it was placed on
        close(v, t)
        seg = [link, t, None, 1 if start else 0]
        vrec(v, kind)["links"].append(seg)
        open_seg[v] = seg

    with gzip.open(path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            a = el.attrib
            el.clear()
            t, typ = float(a["time"]), a["type"]
            person = a.get("person")
            v = veh(a.get("vehicle"))
            if typ == "TransitDriverStarts":
                bus = veh(a["vehicleId"])
                vrec(bus, "bus")["line"] = a.get("transitLineId")
                bus_of_driver[a["driverId"]] = bus
                if a["driverId"].startswith("pt_") and "#" in a["vehicleId"]:  # MobSlim
                    open_(bus, corridor.BUS_ROUTE[0], t, "bus", start=True)
            elif typ == "vehicle enters traffic":
                kind = "bus" if person in bus_of_driver else "car"
                open_(v, a["link"], t, kind, start=True)
            elif typ == "entered link":
                kind = "bus" if v in vehicles and vehicles[v]["kind"] == "bus" else "car"
                open_(v, a["link"], t, kind)
            elif typ in ("left link", "vehicle leaves traffic"):
                close(v, t)
            elif typ == "departure" and person in plans and a.get("legMode") == "car":
                car_legs.setdefault(person, [t, None])
                if "node" in a:  # MobSlim: starts at the end of its first link
                    open_(person, plans[person]["home"][0], t, "car", start=True)
            elif typ == "arrival" and person in plans and a.get("legMode") == "car":
                car_legs[person][1] = t
                close(person, t)
            elif typ in ("VehicleArrivesAtFacility", "VehicleDepartsAtFacility"):
                stops = vrec(v, "bus")["stops"]
                if typ == "VehicleArrivesAtFacility":
                    stops.append([a["facility"], t, None])
                else:
                    for s in reversed(stops):
                        if s[0] == a["facility"] and s[2] is None:
                            s[2] = t
                            break
            elif person in plans and plans[person]["mode"] == "pt":
                p = passengers.setdefault(person, {})
                if typ == "actend" and a.get("actType") == "home":
                    p["leave_home"] = t
                elif typ == "waitingForPt":
                    p["wait"], p["stop"] = t, a["atStop"]
                elif typ in ("PersonEntersVehicle", "PersonEntersPtVehicle"):  # DSim: ...Pt...
                    p["board"], p["bus"] = t, v
                elif typ in ("PersonLeavesVehicle", "PersonLeavesPtVehicle"):
                    p["alight"] = t
                elif typ == "actstart" and a.get("actType") == "work":
                    p["at_work"] = t
    for p in passengers.values():  # where they got off: the bus's stop at that time
        stops = vehicles.get(p.get("bus"), {}).get("stops", [])
        here = [s for s in stops if s[1] <= p.get("alight", -1)]
        p["alight_stop"] = here[-1][0] if here else None
    return {"vehicles": vehicles, "passengers": passengers, "car_legs": car_legs}


# ------------------------------------------------------------------ summary


def summarise(run: dict, ref: dict) -> dict:
    """Headline numbers of one run, and its differences from ``ref`` (MATSim QSim)."""
    legs = run["car_legs"]
    tt = np.array([b - a for a, b in legs.values()])
    common = sorted(set(legs) & set(ref["car_legs"]))
    arr = np.array([legs[p][1] for p in common])
    ref_arr = np.array([ref["car_legs"][p][1] for p in common])
    buses = {v: d for v, d in run["vehicles"].items() if d["kind"] == "bus"}
    ref_buses = {v: d for v, d in ref["vehicles"].items() if d["kind"] == "bus"}
    stop_delta = []
    for v, d in buses.items():
        for (s, a, _), (s2, a2, _) in zip(d["stops"], ref_buses.get(v, {}).get("stops", [])):
            if s == s2:
                stop_delta.append(abs(a - a2))
    pax = run["passengers"]
    same_bus = sum(1 for p, d in pax.items() if d.get("bus") == ref["passengers"].get(p, {}).get("bus"))
    return {
        "cars": len(tt),
        "mean_car_time": float(tt.mean()),
        "last_car_arrival": float(arr.max()),
        "car_arrival_mae": float(np.abs(arr - ref_arr).mean()),
        "car_arrival_p95": float(np.percentile(np.abs(arr - ref_arr), 95)),
        "bus_stop_mae": float(np.mean(stop_delta)) if stop_delta else float("nan"),
        "passengers": len(pax),
        "same_bus": same_bus,
    }


def _clock(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def summary_markdown(rows: list[tuple[str, dict]]) -> str:
    lines = ["| Simulation | Mean car trip | Last car arrives | Car arrival vs MATSim QSim (mean / p95) "
             "| Bus at stops vs MATSim QSim | Passengers on the same bus |",
             "|---|---|---|---|---|---|"]
    for label, s in rows:
        lines.append(
            f"| {label} | {s['mean_car_time']:.0f} s | {_clock(s['last_car_arrival'])} "
            f"| {s['car_arrival_mae']:.1f} s / {s['car_arrival_p95']:.0f} s "
            f"| {s['bus_stop_mae']:.1f} s | {s['same_bus']} of {s['passengers']} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ export


def _r(t):
    return None if t is None else round(float(t), 1)


def export(runs: dict[str, dict], variant: str) -> dict:
    """The animation's JSON: network, stops, persons and one compact record per run."""
    links = corridor.SHOWN
    link_ix = {k: i for i, k in enumerate(links)}
    stops = list(corridor.STOPS)
    stop_ix = {k: i for i, k in enumerate(stops)}
    plans = _plans(variant)
    car_ids = sorted(p for p, d in plans.items() if d["mode"] == "car")
    pax_ids = sorted(p for p, d in plans.items() if d["mode"] == "pt")
    bus_ids = [v for _, v, _ in corridor.BUS_DEPARTURES]

    def vehicle_rec(d):
        segs = [[link_ix[l], _r(a), _r(b), s] for l, a, b, s in d["links"] if l in link_ix]
        return {"l": segs, "s": [[stop_ix[s], _r(a), _r(b)] for s, a, b in d.get("stops", [])]}

    out_runs = []
    for key, run in runs.items():
        out_runs.append({
            "key": key,
            "cars": [vehicle_rec(run["vehicles"].get(c, {"links": []})) for c in car_ids],
            "buses": [vehicle_rec(run["vehicles"].get(b, {"links": []})) for b in bus_ids],
            "pax": [[_r(p.get(k)) for k in ("leave_home", "wait", "board", "alight", "at_work")]
                    + [stop_ix.get(p.get("stop"), -1), bus_ids.index(p["bus"]) if p.get("bus") in bus_ids else -1,
                       stop_ix.get(p.get("alight_stop"), -1)]
                    for p in (run["passengers"].get(i, {}) for i in pax_ids)],
        })
    return {
        "t0": corridor.T0 - 120,
        "nodes": {n: list(xy) for n, xy in corridor.NODES.items() if not n.startswith("R")},
        "links": [{"id": l, "from": corridor.LINKS[l][0], "to": corridor.LINKS[l][1],
                   "length": corridor._length(l), "freespeed": corridor.LINKS[l][3],
                   "capacity": corridor.LINKS[l][2]} for l in links],
        "stops": [{"id": s, "link": link_ix[corridor.STOPS[s][0]], "x": corridor.STOPS[s][1],
                   "y": corridor.STOPS[s][2], "name": corridor.STOPS[s][3]} for s in stops],
        "cars": [{"id": c, "group": "side" if c.startswith("s") else "main"} for c in car_ids],
        "buses": bus_ids,
        "pax": [{"id": p, "home": list(plans[p]["home"][1:]), "work": list(plans[p]["work"][1:])}
                for p in pax_ids],
        "runs": out_runs,
    }


def run_variant(variant: str, force_matsim: bool = False) -> dict:
    """Run one variant on every simulation; returns the animation data."""
    corridor.write(variant)
    paths = run_matsim_side(variant, force_matsim)
    paths.update(run_mobslim_side(variant))
    runs = {k: read_run(p, variant) for k, p in paths.items()}
    ref = runs["matsim_qsim"]

    labels = dict((k, l) for k, l, _ in MATSIM_SIMS) | dict(MOBSLIM_SIMS)
    rows = [(labels[k], summarise(runs[k], ref)) for k, _, _ in MATSIM_SIMS]
    sims = [{"key": k, "label": l, "family": "matsim", "source": f"Java MATSim {v}"}
            for k, l, v in MATSIM_SIMS]
    for clock in CLOCKS:
        for k, label in MOBSLIM_SIMS:
            rows.append((f"MobSlim {label}", summarise(runs[f"{k}.{clock}"], ref)))
            sims.append({"key": f"{k}.{clock}", "label": f"MobSlim {label}", "family": "mobslim",
                         "source": "MobSlim"})
    print(f"== {variant}\n{summary_markdown(rows)}")

    data = export(runs, variant)
    data["sims"] = sims
    data["summary"] = {r["key"]: summarise(runs[r["key"]], ref) for r in sims}
    return data


def main(force_matsim: bool = False) -> None:
    data = {v: run_variant(v, force_matsim) for v in corridor.VARIANTS}
    DOCS_STATIC.mkdir(parents=True, exist_ok=True)
    # a script, not JSON: the docs then also work opened from disk (file://)
    (DOCS_STATIC / "corridor-data.js").write_text(
        "window.MOBSLIM_CORRIDOR = " + json.dumps(data, separators=(",", ":")) + ";\n")
