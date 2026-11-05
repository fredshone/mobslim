"""MATSim population XML: reading ``plans_v4``/``population_v6``, writing ``population_v6``.

What the writer produces: what MATSim can read back, and :func:`read_population_xml`
reads back to the same plans (up to the details below):

* Person attributes go in each person's ``<attributes>`` (typed:
  ``java.lang.Boolean``, ``Integer``/``Long``, ``Double``, ``String``).
* Every plan, with its ``score`` (none if unscored) and ``selected``.
* Activities: ``type``, ``link`` (if the activity has one), ``x``/``y`` (its
  own coordinates, else its node's), ``end_time``, ``max_dur``. MATSim has no
  node attribute: reading the file back places an activity without a link on
  the node nearest its coordinates.
* Legs: ``mode``, ``trav_time`` (fixed teleport time). A network leg's route
  is written as a MATSim ``links`` route (the departure link first, then
  MobSlim's links) when both activities have links and the route ends on the
  arrival activity's link; otherwise the leg is written unrouted. pt routes
  are not written (MATSim's pt routes need its transit router; the legs are
  routed again on reading).
"""

from __future__ import annotations

import gzip
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from mobslim.formats.matsim_attributes import (
    read_attributes_element,
    read_person_attributes_xml,
)
from mobslim.network import Network
from mobslim.planbatch import PlanBatchBuilder
from mobslim.planstore import PlanStore
from mobslim.population import Population, validate_batch
from mobslim.timeutils import parse_time

_HEADER = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\n\n')


def format_matsim_time(seconds: float) -> str:
    """``HH:MM:SS`` (hours may exceed 24), with decimals (up to microseconds)
    only if needed."""
    seconds = round(seconds, 6)
    whole = math.floor(seconds)
    frac = seconds - whole
    h, rest = divmod(int(whole), 3600)
    m, s = divmod(rest, 60)
    text = f"{h:02d}:{m:02d}:{s:02d}"
    if frac:
        text += f"{frac:.6f}".rstrip("0")[1:]
    return text


def _attribute(name: str, value) -> str:
    if isinstance(value, (bool, np.bool_)):
        cls, text = "java.lang.Boolean", "true" if value else "false"
    elif isinstance(value, (int, np.integer)):
        cls = "java.lang.Integer" if -2**31 <= value < 2**31 else "java.lang.Long"
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        cls, text = "java.lang.Double", repr(float(value))
    else:
        cls, text = "java.lang.String", str(value)
    return f"\t\t\t<attribute name={quoteattr(name)} class={quoteattr(cls)}>{escape(text)}</attribute>\n"


def _num(v: float) -> str:
    return repr(float(v))


def write_population_xml(population: Population, path, network: Network) -> Path:
    """Write ``population`` as MATSim ``population_v6`` XML (``.gz`` to
    compress); see the module docs for what is kept."""
    path = Path(path)
    used = population.plan_ids >= 0
    person_of_plan, slot = np.nonzero(used)
    batch = population.plans.take(population.plan_ids[used].astype(np.int64))
    link_ids = [str(i) for i in network.link_ids]
    act_offsets = batch.act_offsets.tolist()
    types, modes = batch.act_types, batch.modes
    a_type, a_link = batch.act_type.tolist(), batch.act_link.tolist()
    xy = batch.resolved_act_xy(network).tolist()
    a_end, a_dur = batch.act_end_time.tolist(), batch.act_duration.tolist()
    l_mode, l_tt = batch.leg_mode.tolist(), batch.leg_travel_time.tolist()
    l_routed, l_pt = batch.leg_routed.tolist(), batch.leg_pt_route.tolist()
    l_rtt, l_rdist = batch.leg_route_time.tolist(), batch.leg_route_distance.tolist()
    r_off, links = batch.route_offsets.tolist(), batch.route_links.tolist()
    scores = population.scores[used].tolist()
    selected = (population.selected[person_of_plan] == slot).tolist()
    attrs = population.attributes
    starts = np.searchsorted(person_of_plan, np.arange(len(population) + 1)).tolist()

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        w = f.write
        w(_HEADER)
        w("<population>\n\n")
        for i, pid in enumerate(population.ids):
            w(f"\t<person id={quoteattr(str(pid))}>\n")
            row = attrs.row(i)
            if row:
                w("\t\t<attributes>\n")
                for name, value in row.items():
                    w(_attribute(name, value))
                w("\t\t</attributes>\n")
            for p in range(starts[i], starts[i + 1]):
                score = scores[p]
                score_attr = "" if score != score else f' score="{_num(score)}"'
                w(f'\t\t<plan{score_attr} selected="{"yes" if selected[p] else "no"}">\n')
                a0, a1 = act_offsets[p], act_offsets[p + 1]
                for a in range(a0, a1):
                    if a > a0:
                        _write_leg(w, a - p - 1, a - 1, a, modes, l_mode, l_tt, l_routed, l_pt,
                                   l_rtt, l_rdist, r_off, links, a_link, link_ids)
                    x, y = xy[a]
                    attr = f"type={quoteattr(types[a_type[a]])}"
                    if a_link[a] >= 0:
                        attr += f" link={quoteattr(link_ids[a_link[a]])}"
                    attr += f' x="{_num(x)}" y="{_num(y)}"'
                    if a_end[a] == a_end[a]:
                        attr += f' end_time="{format_matsim_time(a_end[a])}"'
                    if a_dur[a] == a_dur[a]:
                        attr += f' max_dur="{format_matsim_time(a_dur[a])}"'
                    w(f"\t\t\t<activity {attr} />\n")
                w("\t\t</plan>\n")
            w("\t</person>\n\n")
        w("</population>\n")
    return path


def _write_leg(w, leg, origin, dest, modes, l_mode, l_tt, l_routed, l_pt, l_rtt, l_rdist,
               r_off, links, a_link, link_ids) -> None:
    attr = f"mode={quoteattr(modes[l_mode[leg]])}"
    tt = l_tt[leg]
    if tt == tt:
        attr += f' trav_time="{format_matsim_time(tt)}"'
    route = None
    if l_routed[leg] and not l_pt[leg]:
        start, end = a_link[origin], a_link[dest]
        ids = links[r_off[leg]:r_off[leg + 1]]
        if start >= 0 and end >= 0 and (ids[-1] == end if ids else start == end):
            route = [start, *ids]
    if route is None:
        w(f"\t\t\t<leg {attr} />\n")
        return
    rattr = (f'type="links" start_link={quoteattr(link_ids[route[0]])} '
             f'end_link={quoteattr(link_ids[route[-1]])}')
    if l_rtt[leg] == l_rtt[leg]:
        rattr += f' trav_time="{format_matsim_time(l_rtt[leg])}"'
    if l_rdist[leg] == l_rdist[leg]:
        rattr += f' distance="{_num(l_rdist[leg])}"'
    text = " ".join(link_ids[k] for k in route)
    w(f"\t\t\t<leg {attr}>\n\t\t\t\t<route {rattr}>{escape(text)}</route>\n\t\t\t</leg>\n")


# ----------------------------------------------------------------- reading


def _open_xml(path):
    path = str(path)
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def is_stage_activity(act_type: str) -> bool:
    """MATSim's stage activities (``"pt interaction"``, ``"car interaction"``, ...)."""
    return act_type.endswith(" interaction")


# walk modes MATSim uses for the access, egress and transfer legs of a trip
_TRIP_WALKS = ("walk", "non_network_walk", "transit_walk", "access_walk", "egress_walk")


# plans_v4 person fields -> (attribute name, as MATSim's PersonUtils names
# them, and conversion)
_V4_PERSON_ATTRS = {
    "sex": ("sex", str),
    "age": ("age", int),
    "license": ("hasLicense", str),
    "car_avail": ("carAvail", str),
    "employed": ("employed", lambda v: v.strip().lower() in ("yes", "true")),
}


def _person_attributes(xml_person) -> dict:
    out = {name: conv(xml_person.get(k)) for k, (name, conv) in _V4_PERSON_ATTRS.items()
           if xml_person.get(k) is not None}
    attrs = xml_person.find("attributes")
    if attrs is not None:
        out.update(read_attributes_element(attrs))
    return out


def read_population_xml(path: str, network: Network, *, keep_routes: bool = False,
                        attributes_file: Optional[str] = None) -> Population:
    """Read a MATSim ``plans_v4``/``population_v6`` XML file (``.gz`` allowed).

    Person attributes are kept in ``population.attributes``: a
    ``population_v6`` person's ``<attributes>`` (typed by their ``class``),
    a ``plans_v4`` person's ``sex``/``age``/``license``/``car_avail``/
    ``employed`` (named ``sex``, ``age``, ``hasLicense``, ``carAvail``,
    ``employed`` as MATSim's ``PersonUtils`` does), and, if given, those of a MATSim ``objectAttributes``
    ``attributes_file`` (``plans.inputPersonAttributesFile``), which win
    where both name a value.

    Activity location is taken from (in order) a ``node`` attribute, the
    to-node of a ``link`` attribute (the link is kept as ``Activity.link``),
    or the node nearest to ``x``/``y``.
    Coordinates in the file are kept (``Activity.x, y``). All plans of a
    person are loaded; ``selected="yes"`` is honoured.

    Routes in the file are ignored (legs are routed before the first
    iteration) unless ``keep_routes``: then single-leg network trips keep
    their ``type="links"`` route (as in MATSim's output plans), or their
    ``plans_v4`` node route (the links between the nodes). MATSim's
    route starts on the departure link, which the vehicle does not
    traverse, so it is dropped: MobSlim's route runs from the departure
    link's to-node and includes the arrival link, the same links MATSim
    traverses. Its travel time and distance are the free-speed sums over
    those links. Transit trips are always re-routed.

    Plans that already contain MATSim stage activities (``pt interaction``,
    as in MATSim's output plans) are collapsed into one leg per trip, whose
    mode is the trip's ``routingMode`` attribute (or its main mode).
    """
    records = {}
    with _open_xml(path) as f:
        root = ET.parse(f).getroot()
    b = PlanBatchBuilder()
    ids, rows, scores, selected = [], [], [], []
    nan = math.nan
    for xml_person in root.findall("person"):
        pid = xml_person.get("id")
        ids.append(pid)
        if attrs := _person_attributes(xml_person):
            records[pid] = attrs
        person_rows, person_scores, sel = [], [], 0
        for p, xml_plan in enumerate(xml_person.findall("plan")):
            if xml_plan.get("selected") == "yes":
                sel = p
            acts, legs = [], []
            trip: list = []  # legs (and stage activities) of the current trip
            last_act = None
            for el in xml_plan:
                if el.tag in ("act", "activity"):
                    if is_stage_activity(el.get("type", "")):
                        trip.append(None)
                        continue
                    if trip:
                        leg = _trip_leg(trip)
                        if isinstance(leg, _FileLeg):  # the activity links, for v4 node routes
                            leg = _FileLeg(leg.mode, leg.travel_time, leg.xml,
                                           last_act.get("link"), el.get("link"))
                        legs.append(leg)
                        trip = []
                    elif acts:
                        raise ValueError(f"Person {pid!r}: two activities without a leg between.")
                    acts.append(_read_activity(el, network))
                    last_act = el
                elif el.tag == "leg":
                    if not acts:
                        raise ValueError(f"Person {pid!r}: a plan must start with an activity.")
                    trip.append(el)
            if not acts:
                raise ValueError(f"Person {pid!r}: a plan needs at least one activity.")
            b.act(*acts[0])
            for leg, act in zip(legs, acts[1:]):
                if isinstance(leg, _FileLeg):
                    b.leg(*(_file_route(leg, network) if keep_routes else (leg.mode, leg.travel_time)))
                else:
                    b.leg(leg, nan)
                b.act(*act)
            b.end_plan()
            person_rows.append(b.n_plans - 1)
            score = xml_plan.get("score")
            person_scores.append(float(score) if score else nan)
        if not person_rows:
            raise ValueError(f"Person {pid!r} has no plans.")
        rows.append(person_rows)
        scores.append(person_scores)
        selected.append(sel)
    if attributes_file is not None:
        for pid, attrs in read_person_attributes_xml(attributes_file).items():
            records[pid] = {**records.get(pid, {}), **attrs}
    batch = b.build()
    validate_batch(batch)
    width = max(map(len, rows), default=1)
    plan_ids = np.full((len(ids), width), -1, dtype=np.int32)
    score_table = np.full((len(ids), width), nan)
    for i, (r, sc) in enumerate(zip(rows, scores)):
        plan_ids[i, :len(r)] = r
        score_table[i, :len(sc)] = sc
    return Population.from_tables(ids, PlanStore.from_batch(batch), plan_ids, score_table,
                                  selected, records)


@dataclass(frozen=True)
class _FileLeg:
    """A single-leg trip as read, before deciding whether to keep its route."""

    mode: str
    travel_time: float  # NaN: none
    xml: ET.Element
    start_link: Optional[str] = None  # link ids of the activities around it
    end_link: Optional[str] = None


def _file_route(fl: _FileLeg, network: Network) -> tuple:
    """``PlanBatchBuilder.leg`` arguments for a leg with its file route kept."""
    el = fl.xml.find("route")
    plain = (fl.mode, fl.travel_time)
    if el is None:
        return plain
    if el.get("start_link") is None and el.get("type") is None:
        links = _node_route_links(el.text or "", fl, network)  # plans_v4: nodes
        if links is None:
            return plain
    elif el.get("type", "links") != "links" or el.get("start_link") is None:
        return plain  # a generic/pt route: route it
    else:
        ids = (el.text or "").split()
        if not ids or ids[0] != el.get("start_link") or ids[-1] != el.get("end_link"):
            raise ValueError(f"Route {ids} does not run from start_link {el.get('start_link')!r} "
                             f"to end_link {el.get('end_link')!r}.")
        links = np.array([network.link_index(i) for i in ids[1:]], dtype=np.int64)
    data = links.astype(np.int32).tobytes()
    if links.size == 0:
        return fl.mode, fl.travel_time, data, False, 0.0, 0.0
    travel_time = float(network.free_speed_travel_times()[links].sum())
    distance = float(network.length[links].sum())
    return fl.mode, fl.travel_time, data, False, travel_time, distance


def _node_route_links(text: str, fl: _FileLeg, network: Network) -> Optional[np.ndarray]:
    """A ``plans_v4`` route (the nodes between the departure link's to-node and
    the arrival link's from-node) as MobSlim links: the links between those
    nodes, then the arrival link. ``None`` if the activities have no links."""
    if fl.start_link is None or fl.end_link is None:
        return None
    start, end = network.link_index(fl.start_link), network.link_index(fl.end_link)
    if start == end and not text.split():
        return np.zeros(0, dtype=np.int64)
    nodes = [network.node_index(n) for n in text.split()]
    if not nodes:  # adjacent links
        nodes = [int(network.to_node[start])]
    if nodes[0] != int(network.to_node[start]) or nodes[-1] != int(network.from_node[end]):
        raise ValueError(f"Node route {text.split()} does not run from link {fl.start_link!r} "
                         f"to link {fl.end_link!r}.")
    offsets, out = network.out_links(fl.mode)
    links = []
    for u, v in zip(nodes, nodes[1:]):
        hit = [k for k in out[offsets[u]:offsets[u + 1]] if int(network.to_node[k]) == v]
        if not hit:
            raise ValueError(f"No {fl.mode!r} link from node {network.node_ids.id(u)!r} "
                             f"to {network.node_ids.id(v)!r}.")
        links.append(hit[0])
    links.append(end)
    return np.array(links, dtype=np.int64)


def _routing_mode(el) -> Optional[str]:
    """A leg's ``routingMode`` attribute (the mode of its trip), if any."""
    attrs = el.find("attributes")
    if attrs is not None:
        for a in attrs:
            if a.get("name") == "routingMode" and a.text:
                return a.text.strip()
    return None


def _trip_leg(trip: list):
    """One leg for a trip read as ``leg (stage leg)*``: its mode (a leg to
    route) or a :class:`_FileLeg` (a single leg, whose route may be kept).

    A single leg whose ``routingMode`` names another mode is a trip of that
    mode the router made of one leg, e.g. a ``pt`` trip MATSim walks
    entirely: it is read as an unrouted leg of the routing mode.
    """
    legs = [el for el in trip if el is not None]
    if len(legs) == 1 and legs[0].get("mode") != "transit_walk":
        routing = _routing_mode(legs[0])
        if routing is not None and routing != legs[0].get("mode"):
            return routing
        tt = legs[0].get("trav_time")
        return _FileLeg(legs[0].get("mode"), parse_time(tt) if tt else math.nan, legs[0])
    mode = None
    for el in legs:
        mode = _routing_mode(el)
        if mode is not None:
            break
    if mode is None:
        main = [el.get("mode") for el in legs if el.get("mode") not in _TRIP_WALKS]
        if main:
            mode = main[0]
        elif any(el.get("mode") == "transit_walk" for el in legs):
            mode = "pt"  # legacy files: a pt trip the router made walk-only
        else:
            mode = legs[0].get("mode")
    return mode


def _read_activity(el, network: Network) -> tuple:
    """``PlanBatchBuilder.act`` arguments for an ``<act>``/``<activity>``."""
    nan = math.nan
    x, y = el.get("x"), el.get("y")
    x, y = (float(x), float(y)) if x is not None and y is not None else (nan, nan)
    link = -1
    if el.get("node") is not None:
        node = network.node_index(el.get("node"))
    elif el.get("link") is not None:
        link = network.link_index(el.get("link"))
        node = int(network.to_node[link])
    else:
        if x != x:
            raise ValueError(f"Activity {el.get('type')!r} has no node, link or coordinates.")
        node = network.nearest_node(x, y)
    end_time = el.get("end_time")
    duration = el.get("dur") or el.get("max_dur")
    return (el.get("type"), node, parse_time(end_time) if end_time else nan,
            parse_time(duration) if duration else nan, x, y, link)
