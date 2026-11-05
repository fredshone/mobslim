"""Writing populations as MATSim ``population_v6`` XML.

What MATSim can read back, and :func:`~mobslim.population.read_population_xml`
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
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from mobslim.network import Network
from mobslim.population import Population

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
