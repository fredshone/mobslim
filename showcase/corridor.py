"""The showcase corridor: a small MATSim scenario for side-by-side animations.

A one-way main road with a side road merging into it, a short bottleneck
just after the merge, and a bus line with three stops that drives in the
same traffic::

    side home
         \\
          s
           \\
    H--a-->A---ab--->B---bc--->C==cd==>D---de--->E--ef-->F--fw-->W
                 [stop 1]          bottleneck  [stop 2]  [stop 3]

Two variants share the network, the buses and the passengers:

* ``congested``: 77 cars leave home within four minutes (most on the main
  road, some on the side road), more than the bottleneck can take, so a
  queue forms on it and spills back past the merge, where the two
  approaches compete.
* ``free-flow``: 20 cars spread over eight minutes, well below every
  link's capacity.

Two buses, five minutes apart, pick up passengers at stop 1 and stop 2 and
drop them at stops 2 and 3, driving in the same traffic as the cars.

Every input is written as MATSim XML, so Java MATSim reads exactly the same
files. Departure times are whole seconds from a fixed seed.

``python -m showcase scenario`` rewrites ``showcase/scenarios/corridor/<variant>``.
"""

from __future__ import annotations

import random
from pathlib import Path

HERE = Path(__file__).parent
SCENARIOS = HERE / "scenarios" / "corridor"

T0 = 7 * 3600  # 07:00

# id: (x, y)
NODES = {
    "H": (-100, 0), "A": (0, 0), "B": (500, 0), "C": (1000, 0), "D": (1200, 0),
    "E": (1700, 0), "F": (2100, 0), "W": (2200, 0),
    "S0": (560, 190), "S": (650, 150),
    "R": (1100, -300),  # the return road (never used; MATSim needs a connected network)
}
# id: (from, to, capacity veh/h, freespeed m/s); length = Euclidean distance
LINKS = {
    "a": ("H", "A", 3600, 13.89),
    "ab": ("A", "B", 1800, 13.89),
    "bc": ("B", "C", 1800, 13.89),
    "s0": ("S0", "S", 3600, 13.89),
    "s": ("S", "C", 1200, 11.11),
    "cd": ("C", "D", 720, 8.33),  # the bottleneck: one car every 5 s, 26 cars of room
    "de": ("D", "E", 1800, 13.89),
    "ef": ("E", "F", 1800, 13.89),
    "fw": ("F", "W", 3600, 13.89),
    "r1": ("W", "R", 3600, 13.89),
    "r2": ("R", "H", 3600, 13.89),
    "r3": ("R", "S0", 3600, 13.89),
}
SHOWN = [k for k in LINKS if not k.startswith("r")]
# id: (link, x, y, name); a stop sits at the end of its link
STOPS = {
    "stop1": ("ab", 500, 0, "Mill Lane"),
    "stop2": ("de", 1700, 0, "Market"),
    "stop3": ("ef", 2100, 0, "Station"),
}
BUS_ROUTE = ["a", "ab", "bc", "cd", "de", "ef", "fw"]
# (departure id, vehicle id, time)
BUS_DEPARTURES = [("dep1", "bus1", T0 + 60), ("dep2", "bus2", T0 + 360)]
# schedule offsets from the route start: (arrival, departure)
BUS_OFFSETS = {"stop1": (40, 50), "stop2": (180, 190), "stop3": (220, 220)}

# cars on the main road, on the side road, and the window they leave home in (s)
VARIANTS = {
    "congested": dict(main=55, side=22, window=240),
    "free-flow": dict(main=14, side=6, window=480),
}
# (count, from stop, to stop, time window after T0)
PT_GROUPS = [(6, "stop1", "stop3", (-110, -30)), (4, "stop1", "stop2", (-110, -30)),
             (3, "stop2", "stop3", (60, 160)), (4, "stop1", "stop3", (190, 300)),
             (2, "stop2", "stop3", (330, 430))]
SEED = 7


def _length(link: str) -> float:
    (x0, y0), (x1, y1) = NODES[LINKS[link][0]], NODES[LINKS[link][1]]
    return round(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1)


def _hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def network_xml() -> str:
    nodes = "\n".join(f'    <node id="{n}" x="{x}" y="{y}"/>' for n, (x, y) in NODES.items())
    links = "\n".join(
        f'    <link id="{i}" from="{f}" to="{t}" length="{_length(i)}" freespeed="{v}" '
        f'capacity="{c}" permlanes="1" modes="{"car" if i.startswith("r") else "car,pt"}"/>'
        for i, (f, t, c, v) in LINKS.items())
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">\n'
            f'<network>\n  <nodes>\n{nodes}\n  </nodes>\n'
            f'  <links capperiod="01:00:00">\n{links}\n  </links>\n</network>\n')


def schedule_xml() -> str:
    stops = "\n".join(f'    <stopFacility id="{s}" x="{x}" y="{y}" linkRefId="{link}" name="{name}"/>'
                      for s, (link, x, y, name) in STOPS.items())
    profile = "\n".join(
        f'        <stop refId="{s}" arrivalOffset="{_hms(a)}" departureOffset="{_hms(d)}" '
        f'awaitDeparture="true"/>' for s, (a, d) in BUS_OFFSETS.items())
    route = "\n".join(f'        <link refId="{link}"/>' for link in BUS_ROUTE)
    deps = "\n".join(f'        <departure id="{d}" departureTime="{_hms(t)}" vehicleRefId="{v}"/>'
                     for d, v, t in BUS_DEPARTURES)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE transitSchedule SYSTEM "http://www.matsim.org/files/dtd/transitSchedule_v1.dtd">\n'
            f'<transitSchedule>\n  <transitStops>\n{stops}\n  </transitStops>\n'
            '  <transitLine id="line7">\n    <transitRoute id="line7_east">\n'
            '      <transportMode>pt</transportMode>\n'
            f'      <routeProfile>\n{profile}\n      </routeProfile>\n'
            f'      <route>\n{route}\n      </route>\n'
            f'      <departures>\n{deps}\n      </departures>\n'
            '    </transitRoute>\n  </transitLine>\n</transitSchedule>\n')


def vehicles_xml() -> str:
    vehicles = "\n".join(f'  <vehicle id="{v}" type="bus"/>' for _, v, _ in BUS_DEPARTURES)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://www.matsim.org/files/dtd '
            'http://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd">\n'
            '  <vehicleType id="bus">\n    <description>bus</description>\n'
            '    <capacity>\n      <seats persons="40"/>\n      <standingRoom persons="30"/>\n'
            '    </capacity>\n    <length meter="12.0"/>\n'
            '    <accessTime secondsPerPerson="1.0"/>\n    <egressTime secondsPerPerson="1.0"/>\n'
            '  </vehicleType>\n'
            f'{vehicles}\n</vehicleDefinitions>\n')


def directory(variant: str) -> Path:
    return SCENARIOS / variant


def persons(variant: str) -> list[dict]:
    """Every person's plan: ``id, mode, home (link, x, y), work (link, x, y), end``."""
    rng = random.Random(SEED)
    out = []
    xy = lambda link: NODES[LINKS[link][1]]  # noqa: E731  (end of the link)
    v = VARIANTS[variant]
    for i in range(v["main"]):
        out.append(dict(id=f"m{i:02d}", mode="car", home=("a", *NODES["A"]), work=("fw", *xy("fw")),
                        end=T0 + rng.randint(0, v["window"])))
    for i in range(v["side"]):
        out.append(dict(id=f"s{i:02d}", mode="car", home=("s0", *NODES["S"]), work=("fw", *xy("fw")),
                        end=T0 + rng.randint(0, v["window"])))
    n = 0
    for count, origin, dest, (lo, hi) in PT_GROUPS:
        for _ in range(count):
            olink, ox, oy, _ = STOPS[origin]
            dlink, dx, dy, _ = STOPS[dest]
            # homes and workplaces a short walk from the stops, off the road
            home = (olink, ox - rng.randint(10, 40), oy - rng.randint(20, 45))
            work = (dlink, dx - rng.randint(0, 40), dy - rng.randint(20, 45))
            out.append(dict(id=f"p{n:02d}", mode="pt", home=home, work=work,
                            end=T0 + rng.randint(lo, hi)))
            n += 1
    out.sort(key=lambda p: (p["end"], p["id"]))
    return out


def population_xml(variant: str) -> str:
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">', "<plans>"]
    for p in persons(variant):
        (hl, hx, hy), (wl, wx, wy) = p["home"], p["work"]
        lines += [f'  <person id="{p["id"]}">', '    <plan selected="yes">',
                  f'      <act type="home" link="{hl}" x="{hx}" y="{hy}" end_time="{_hms(p["end"])}"/>',
                  f'      <leg mode="{p["mode"]}"/>',
                  f'      <act type="work" link="{wl}" x="{wx}" y="{wy}"/>',
                  "    </plan>", "  </person>"]
    lines.append("</plans>")
    return "\n".join(lines) + "\n"


CONFIG = """<?xml version="1.0" ?>
<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">
<config>
  <module name="global">
    <param name="randomSeed" value="4711" />
    <param name="coordinateSystem" value="Atlantis" />
    <param name="numberOfThreads" value="1" />
  </module>
  <module name="network">
    <param name="inputNetworkFile" value="network.xml" />
  </module>
  <module name="plans">
    <param name="inputPlansFile" value="population.xml" />
  </module>
  <module name="transit">
    <param name="useTransit" value="true" />
    <param name="transitScheduleFile" value="transitSchedule.xml" />
    <param name="vehiclesFile" value="transitVehicles.xml" />
    <param name="transitModes" value="pt" />
  </module>
  <module name="controller">
    <param name="firstIteration" value="0" />
    <param name="lastIteration" value="0" />
    <param name="eventsFileFormat" value="xml" />
    <param name="writeEventsInterval" value="1" />
    <param name="writePlansInterval" value="1" />
  </module>
  <module name="qsim">
    <param name="endTime" value="09:00:00" />
    <param name="stuckTime" value="3600" />
    <param name="numberOfThreads" value="1" />
  </module>
  <module name="scoring">
    <parameterset type="scoringParameters">
      <param name="performing" value="+6" />
      <parameterset type="modeParams">
        <param name="mode" value="car"/>
        <param name="marginalUtilityOfTraveling_util_hr" value="-6.0" />
      </parameterset>
      <parameterset type="modeParams">
        <param name="mode" value="pt"/>
      </parameterset>
      <parameterset type="modeParams">
        <param name="mode" value="walk"/>
      </parameterset>
      <parameterset type="activityParams">
        <param name="activityType" value="home" />
        <param name="typicalDuration" value="12:00:00" />
      </parameterset>
      <parameterset type="activityParams">
        <param name="activityType" value="work" />
        <param name="typicalDuration" value="08:00:00" />
      </parameterset>
    </parameterset>
  </module>
  <module name="replanning">
    <parameterset type="strategysettings">
      <param name="strategyName" value="BestScore"/>
      <param name="weight" value="1.0"/>
    </parameterset>
  </module>
</config>
"""


def write(variant: str) -> Path:
    d = directory(variant)
    d.mkdir(parents=True, exist_ok=True)
    for name, text in [("network.xml", network_xml()), ("transitSchedule.xml", schedule_xml()),
                       ("transitVehicles.xml", vehicles_xml()), ("population.xml", population_xml(variant)),
                       ("config.xml", CONFIG)]:
        (d / name).write_text(text)
    return d
