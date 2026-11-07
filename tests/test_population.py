import pytest

from mobslim import Activity, Leg, Person, Plan, Population, read_population_xml


def test_plan_validation():
    with pytest.raises(ValueError):
        Plan(())
    with pytest.raises(ValueError):
        Plan((Activity("h", 0, end_time=1), Leg("car")))  # ends with a leg
    with pytest.raises(ValueError):
        Plan((Activity("h", 0), Leg("car"), Activity("w", 1)))  # first act has no end
    with pytest.raises(ValueError):
        Plan((Activity("h", 0, end_time=1), Activity("w", 1)))  # not alternating


def test_builder_and_trips():
    plan = Plan.builder().act("h", 0, end_time=10).leg("car").act("w", 3).build()
    assert [t[2:] for t in plan.trips()] == [(0, 3)]
    assert plan.activities[1].type == "w"


def test_planned_end_semantics():
    by_end = Activity("h", 0, end_time=100)
    assert by_end.planned_end(50) == 100
    assert by_end.planned_end(150) == 150  # late arrival: leave immediately
    by_dur = Activity("w", 0, duration=60)
    assert by_dur.planned_end(150) == 210
    assert Activity("h", 0).planned_end(0) is None


def test_plans_are_immutable_and_have_no_score():
    plan = Plan.builder().act("h", 0, end_time=10).leg("car").act("w", 3).build()
    with pytest.raises(AttributeError):
        plan.elements = ()
    assert not hasattr(plan, "score")


def test_person_plan_memory():
    p1 = Plan.builder().act("h", 0).build()
    p2 = Plan.builder().act("h", 1).build()
    person = Person("a", (p1,), scores=(1.0,)).add_plan(p2, score=2.0)
    assert person.selected == 1 and person.selected_score == 2.0
    person = person.remove_plan(0)
    assert person.plans == (p2,) and person.scores == (2.0,) and person.selected == 0
    with pytest.raises(ValueError):
        person.remove_plan(0)


def test_person_scores():
    plan = Plan.builder().act("h", 0).build()
    person = Person("a", (plan, plan))
    assert person.scores == (None, None)  # default: unscored
    assert person.select(1).with_selected_score(3.0).scores == (None, 3.0)
    with pytest.raises(ValueError):
        Person("a", (plan,), scores=(1.0, 2.0))  # one score per plan


def test_population_tables():
    shared = Plan.builder().act("h", 0).build()
    other = Plan.builder().act("h", 1).build()
    pop = Population([
        Person("a", (shared,), scores=(1.0,)),
        Person("b", (shared, other), selected=1, scores=(None, 2.0)),
    ])
    assert len(pop.plans) == 2  # the shared plan is stored once
    assert pop.plan_ids.tolist() == [[0, -1], [0, 1]]
    assert pop.selected.tolist() == [0, 1] and pop.n_plans.tolist() == [1, 2]
    assert pop.selected_scores().tolist() == [1.0, 2.0]
    assert pop.selected_plans() == [shared, other]
    assert list(pop)[1] == pop.by_id("b") == pop[-1]
    assert pop[1].scores == (None, 2.0)  # NaN is None again on the Person
    with pytest.raises(ValueError):
        pop.scores[0, 0] = 5.0  # read-only


def test_population_updates_share_unchanged_tables():
    pop = Population.from_plans({"a": Plan.builder().act("h", 0).build()})
    scored = pop.with_selected_scores([4.0])
    assert scored[0].selected_score == 4.0 and pop[0].selected_score is None
    assert scored.plans is pop.plans and scored.ids is pop.ids
    moved = scored.map_plans(lambda p: Plan.builder().act("h", 9).build())
    assert moved[0].selected_plan.activities[0].node == 9
    assert moved.scores is scored.scores


def test_population_replace_keeps_order():
    pop = Population.from_plans({"a": Plan.builder().act("h", 0).build(),
                                 "b": Plan.builder().act("h", 1).build()})
    assert pop.ids.index("b") == 1
    with pytest.raises(ValueError):
        pop.replace(reversed(list(pop)))
    assert pop.replace(list(pop))[1].id == "b"


def test_read_equil_population(equil_scenario):
    pop = equil_scenario.population
    net = equil_scenario.network
    assert len(pop) == 100
    plan = pop.by_id("1").selected_plan
    assert [a.type for a in plan.activities] == ["h", "w", "w", "h"]
    assert plan.activities[0].end_time == 6 * 3600  # clock time, not a duration
    assert plan.activities[1].duration == 600
    assert plan.activities[0].node == net.node_index("2")
    assert all(leg.mode == "car" and leg.route is None for leg in plan.legs)


def test_read_population_by_coordinates(equil_scenario, tmp_path):
    xml = tmp_path / "plans.xml"
    xml.write_text(
        '<plans><person id="x"><plan>'
        '<act type="h" x="-25000" y="0" end_time="06:00"/><leg mode="walk"/>'
        '<act type="w" link="20"/></plan></person></plans>'
    )
    net = equil_scenario.network
    plan = read_population_xml(xml, net).by_id("x").selected_plan
    assert plan.activities[0].node == net.node_index("1")
    assert plan.activities[1].node == net.node_index("13")  # to-node of link 20
    assert plan.activities[0].link is None
    assert plan.activities[1].link == net.link_index("20")  # the link is kept


def test_read_population_legacy_transit_walk_trips_are_pt(equil_scenario, tmp_path):
    """Old MATSim files: a pt trip the router made walk-only is a lone
    ``transit_walk`` leg; a ridden one is ``transit_walk``, ``pt`` stages.
    Both read as one ``pt`` leg, as MATSim's main-mode identifier does."""
    xml = tmp_path / "plans.xml"
    xml.write_text(
        '<plans><person id="x"><plan>'
        '<act type="h" node="1" end_time="06:00"/>'
        '<leg mode="transit_walk" trav_time="00:06:06"/>'
        '<act type="w" node="2" end_time="07:00"/>'
        '<leg mode="transit_walk"/><act type="pt interaction" node="3" max_dur="00:00:00"/>'
        '<leg mode="pt"/><act type="pt interaction" node="4" max_dur="00:00:00"/>'
        '<leg mode="transit_walk"/>'
        '<act type="h" node="1"/></plan></person></plans>'
    )
    plan = read_population_xml(xml, equil_scenario.network).by_id("x").selected_plan
    assert [leg.mode for leg in plan.legs] == ["pt", "pt"]
    assert plan.legs[0].travel_time is None  # routed afresh, not teleported for 366 s
    assert [a.type for a in plan.activities] == ["h", "w", "h"]


def test_read_population_scores_and_selection(equil_scenario, tmp_path):
    xml = tmp_path / "plans.xml"
    plan = '<act type="h" node="1" end_time="06:00"/><leg mode="car"/><act type="w" node="2"/>'
    xml.write_text(
        f'<plans><person id="x"><plan score="1.5">{plan}</plan>'
        f'<plan selected="yes">{plan}</plan></person></plans>'
    )
    person = read_population_xml(xml, equil_scenario.network).by_id("x")
    assert person.scores == (1.5, None) and person.selected == 1


def test_read_population_keep_routes(equil_scenario, tmp_path):
    """MATSim output-plan routes: the departure link is dropped (not traversed),
    the arrival link kept; a same-link leg gets an empty route."""
    net = equil_scenario.network
    xml = tmp_path / "plans.xml"
    xml.write_text(
        '<population><person id="x"><plan selected="yes">'
        '<activity type="h" link="1" end_time="06:00:00"/>'
        '<leg mode="car"><route type="links" start_link="1" end_link="20">1 6 15 20</route></leg>'
        '<activity type="w" link="20" max_dur="00:10:00"/>'
        '<leg mode="car"><route type="links" start_link="20" end_link="20">20</route></leg>'
        '<activity type="w" link="20" max_dur="03:30:00"/>'
        '<leg mode="car"/>'
        '<activity type="h" link="1"/></plan></person></population>'
    )
    ignored = read_population_xml(xml, net).by_id("x").selected_plan
    assert all(leg.route is None for leg in ignored.legs)

    legs = read_population_xml(xml, net, keep_routes=True).by_id("x").selected_plan.legs
    links = [net.link_index(i) for i in ("6", "15", "20")]
    assert legs[0].route.links == tuple(links)
    assert legs[0].route.distance == pytest.approx(float(net.length[links].sum()))
    assert legs[0].route.travel_time == pytest.approx(
        float(net.free_speed_travel_times()[links].sum()))
    assert legs[1].route.links == ()
    assert legs[2].route is None  # no route in the file: routed as usual


def test_read_population_keep_routes_v4_node_routes(equil_scenario, tmp_path):
    """plans_v4 routes list the nodes between the departure link's to-node and
    the arrival link's from-node; MobSlim adds the arrival link."""
    net = equil_scenario.network
    xml = tmp_path / "plans.xml"
    xml.write_text(
        '<plans><person id="x"><plan>'
        '<act type="h" link="1" end_time="06:00"/>'
        '<leg mode="car"><route>2 7 12</route></leg>'
        '<act type="w" link="20" dur="00:10"/>'
        '<leg mode="car"><route> </route></leg>'
        '<act type="w" link="20" dur="03:30"/>'
        '<leg mode="car"><route>13 14 15 1</route></leg>'
        '<act type="h" link="1"/></plan></person></plans>'
    )
    legs = read_population_xml(xml, net, keep_routes=True).by_id("x").selected_plan.legs
    ids = [[net.link_ids.id(k) for k in leg.route.links] for leg in legs]
    assert ids == [["6", "15", "20"], [], ["21", "22", "23", "1"]]


def test_read_population_keep_routes_checks_endpoints(equil_scenario, tmp_path):
    xml = tmp_path / "plans.xml"
    xml.write_text(
        '<population><person id="x"><plan>'
        '<activity type="h" link="1" end_time="06:00:00"/>'
        '<leg mode="car"><route type="links" start_link="1" end_link="20">6 15 20</route></leg>'
        '<activity type="w" link="20"/></plan></person></population>'
    )
    with pytest.raises(ValueError, match="start_link"):
        read_population_xml(xml, equil_scenario.network, keep_routes=True)


# A small population of each format with everything the reader handles:
# person attributes (typed, plans_v4 fields, an attributes file), several
# plans with scores and selection, kept routes (links and v4 nodes), pt trips
# with stage activities, and root children that are not persons.
_RM = '<attributes><attribute name="routingMode" class="java.lang.String">{}</attribute></attributes>'
_V6 = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">
<population>
  <attributes><attribute name="coordinateReferenceSystem" class="java.lang.String">EPSG:0</attribute></attributes>
  <!-- persons follow -->
  <person id="a">
    <attributes>
      <attribute name="age" class="java.lang.Integer">31</attribute>
      <attribute name="car" class="java.lang.Boolean">true</attribute>
      <attribute name="income" class="java.lang.Double">1.5</attribute>
      <attribute name="subpopulation" class="java.lang.String">x</attribute>
    </attributes>
    <plan score="12.5" selected="no">
      <activity type="h" link="1" x="-25000.0" y="0.0" end_time="06:00:00"/>
      <leg mode="car" trav_time="00:20:00">{_RM.format("car")}
        <route type="links" start_link="1" end_link="20" trav_time="00:20:00" distance="25000.0">1 6 15 20</route>
      </leg>
      <activity type="w" link="20" max_dur="08:00:00"/>
      <leg mode="car"><route type="links" start_link="20" end_link="1">20 21 22 23 1</route></leg>
      <activity type="h" link="1"/>
    </plan>
    <plan score="-3.25" selected="yes">
      <activity type="h" link="1" end_time="07:00:00"/>
      <leg mode="walk" trav_time="00:05:00">{_RM.format("pt")}<route type="generic" start_link="1" end_link="6"/></leg>
      <activity type="pt interaction" link="6" x="1.0" y="2.0" max_dur="00:00:00"/>
      <leg mode="pt" trav_time="00:15:00">{_RM.format("pt")}<route type="default_pt" start_link="6" end_link="20">{{"transitLineId":"l"}}</route></leg>
      <activity type="pt interaction" link="20" max_dur="00:00:00"/>
      <leg mode="walk" trav_time="00:02:00">{_RM.format("pt")}<route type="generic" start_link="20" end_link="20"/></leg>
      <activity type="w" link="20" end_time="17:00:00"/>
      <leg mode="walk" trav_time="00:30:00">{_RM.format("pt")}<route type="generic" start_link="20" end_link="1"/></leg>
      <activity type="h" link="1"/>
    </plan>
  </person>
  <person id="b">
    <plan>
      <activity type="h" x="-25000.0" y="0.0" end_time="08:00:00"/>
      <leg mode="bike" trav_time="00:40:00"/>
      <activity type="w" link="20" max_dur="01:00:00"/>
      <leg mode="car"><route type="links" start_link="20" end_link="20">20</route></leg>
      <activity type="w2" link="20"/>
    </plan>
  </person>
</population>
'''
_OBJ = '''<?xml version="1.0" encoding="utf-8"?>
<objectAttributes>
  <object id="a"><attribute name="age" class="java.lang.Integer">32</attribute></object>
  <object id="b"><attribute name="carAvail" class="java.lang.String">never</attribute></object>
  <object id="c"><attribute name="age" class="java.lang.Long">5</attribute></object>
</objectAttributes>
'''
_V4 = '''<?xml version="1.0" ?>
<!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">
<plans>
  <person id="p1" sex="f" age="40" license="yes" car_avail="always" employed="yes">
    <plan score="5" selected="yes">
      <act type="h" link="1" x="-25000" y="0" end_time="06:00"/>
      <leg mode="car" trav_time="00:10"><route>2 7 12</route></leg>
      <act type="w" link="20" dur="00:10"/>
      <leg mode="car"><route>13 14 15 1</route></leg>
      <act type="h" link="1"/>
    </plan>
    <plan selected="no">
      <act type="h" node="2" end_time="06:00"/>
      <leg mode="walk"/>
      <act type="w" node="13"/>
    </plan>
  </person>
  <person id="p2" employed="no"><plan><act type="h" link="1"/></plan></person>
</plans>
'''


def _summary(pop, net):
    """Everything a read population holds, with node and link ids."""
    out = []
    for i, person in enumerate(pop):
        plans = []
        for plan in person.plans:
            els = []
            for e in plan.elements:
                if isinstance(e, Leg):
                    r = e.route
                    els.append(("leg", e.mode, e.travel_time, None if r is None else (
                        [net.link_ids.id(k) for k in r.links], round(r.travel_time, 3), round(r.distance, 3))))
                else:
                    els.append(("act", e.type, net.node_ids.id(e.node),
                                None if e.link is None else net.link_ids.id(e.link),
                                e.end_time, e.duration, e.x, e.y))
            plans.append(els)
        out.append((person.id, person.selected, person.scores, pop.attributes.row(i), plans))
    return out


def _act(type, node, link=None, end=None, dur=None, x=None, y=None):
    return ("act", type, node, link, end, dur, x, y)


# What the reader produced before it streamed the file (whole-document
# ElementTree), with keep_routes=True.
_EXPECTED_V6 = [
    ("a", 1, (12.5, -3.25), {"age": 32.0, "car": True, "income": 1.5, "subpopulation": "x"},
     [[_act("h", "2", "1", end=21600.0, x=-25000.0, y=0.0),
       ("leg", "car", 1200.0, (["6", "15", "20"], 899.928, 25000.0)),
       _act("w", "13", "20", dur=28800.0),
       ("leg", "car", None, (["21", "22", "23", "1"], 2339.813, 65000.0)),
       _act("h", "2", "1")],
      [_act("h", "2", "1", end=25200.0),
       ("leg", "pt", None, None),  # walk, pt interaction, pt, pt interaction, walk
       _act("w", "13", "20", end=61200.0),
       ("leg", "pt", None, None),  # a pt trip walked entirely
       _act("h", "2", "1")]]),
    ("b", 0, (None,), {"carAvail": "never"},
     [[_act("h", "1", end=28800.0, x=-25000.0, y=0.0),
       ("leg", "bike", 2400.0, None),
       _act("w", "13", "20", dur=3600.0),
       ("leg", "car", None, ([], 0.0, 0.0)),
       _act("w2", "13", "20")]]),
]
_EXPECTED_V4 = [
    ("p1", 0, (5.0, None),
     {"age": 40.0, "carAvail": "always", "employed": True, "hasLicense": "yes", "sex": "f"},
     [[_act("h", "2", "1", end=21600.0, x=-25000.0, y=0.0),
       ("leg", "car", 600.0, (["6", "15", "20"], 899.928, 25000.0)),
       _act("w", "13", "20", dur=600.0),
       ("leg", "car", None, (["21", "22", "23", "1"], 2339.813, 65000.0)),
       _act("h", "2", "1")],
      [_act("h", "2", end=21600.0), ("leg", "walk", None, None), _act("w", "13")]]),
    ("p2", 0, (None,), {"employed": False}, [[_act("h", "2", "1")]]),
]


def _without_routes(expected):
    return [(*person[:4], [[e[:3] + (None,) if e[0] == "leg" else e for e in plan]
                           for plan in person[4]])
            for person in expected]


@pytest.fixture(params=["python", "compiled"])
def plans_reader(request, monkeypatch):
    """Runs a test with the Python reader, then with the compiled one
    (``mobslim_io``, skipped if it is not installed)."""
    from mobslim.formats import matsim_plans

    if request.param == "python":
        monkeypatch.setattr(matsim_plans, "_read_compiled", lambda *args, **kwargs: None)
    else:
        pytest.importorskip("mobslim_io")
    return request.param


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("keep_routes", [False, True])
def test_read_population_streamed_matches_whole_document_reader(equil_scenario, tmp_path,
                                                                 keep_routes, gz, plans_reader):
    """Both readers give what the whole-document reader gave."""
    import gzip

    from mobslim.formats.matsim_plans import _read_compiled

    net = equil_scenario.network
    paths = {}
    for name, text in (("v6.xml", _V6), ("v4.xml", _V4), ("attrs.xml", _OBJ)):
        path = tmp_path / (name + ".gz" if gz else name)
        (gzip.open if gz else open)(path, "wt", encoding="utf-8").write(text)
        paths[name] = path
    v6 = read_population_xml(paths["v6.xml"], net, keep_routes=keep_routes,
                             attributes_file=paths["attrs.xml"])
    v4 = read_population_xml(paths["v4.xml"], net, keep_routes=keep_routes)
    fix = (lambda e: e) if keep_routes else _without_routes
    assert _summary(v6, net) == fix(_EXPECTED_V6)
    assert _summary(v4, net) == fix(_EXPECTED_V4)
    if plans_reader == "compiled":  # read by it, not handed to the Python reader
        assert _read_compiled(paths["v6.xml"], net, keep_routes) is not None
        # plans_v4 node routes are only kept by the Python reader
        assert (_read_compiled(paths["v4.xml"], net, keep_routes) is None) == keep_routes


def _same_read(a: tuple, b: tuple) -> None:
    """Two readers' ``(batch, ids, n_plans, scores, selected, records)``."""
    import dataclasses

    import numpy as np

    for f in dataclasses.fields(a[0]):
        x, y = getattr(a[0], f.name), getattr(b[0], f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), f.name
        else:
            assert x == y, f.name
    assert list(a[1]) == list(b[1])
    assert np.asarray(a[2]).tolist() == np.asarray(b[2]).tolist()
    assert np.array_equal(np.asarray(a[3], dtype=float), np.asarray(b[3], dtype=float), equal_nan=True)
    assert list(a[4]) == list(b[4])
    assert a[5] == b[5] and [list(r.items()) for r in a[5].values()] == \
        [list(r.items()) for r in b[5].values()] and list(a[5]) == list(b[5])
    for r, s in zip(a[5].values(), b[5].values()):
        assert [type(v) for v in r.values()] == [type(v) for v in s.values()]


_FIXTURE_PLANS = ["equil/plans100.xml", "pt_simple/population.xml",
                  "pt_lineswitch/population.xml", "pt_tutorial/population.xml.gz",
                  "matsim_reference/equil-100/output_plans.xml.gz",
                  "matsim_reference/pt-tutorial/output_plans.xml.gz",
                  "matsim_reference/equil-2000-cap0.5/output_plans.xml.gz"]


@pytest.mark.parametrize("chunk_bytes", [None, 1])
@pytest.mark.parametrize("keep_routes", [False, True])
@pytest.mark.parametrize("plans", _FIXTURE_PLANS)
def test_compiled_plans_reader_matches_python(plans, keep_routes, chunk_bytes):
    """The compiled reader reads MATSim plans files as the Python reader does,
    also cut into many chunks (``chunk_bytes=1``: one per person)."""
    pytest.importorskip("mobslim_io")
    from mobslim import read_network_xml
    from mobslim.formats.matsim_plans import _read_compiled, _read_python
    from tests.conftest import FIXTURES

    path = FIXTURES / plans
    net = read_network_xml(path.parent / "network.xml")
    options = {} if chunk_bytes is None else {"chunk_bytes": chunk_bytes, "threads": 3}
    compiled = _read_compiled(path, net, keep_routes, **options)
    if keep_routes and plans == "equil/plans100.xml":
        assert compiled is None  # plans_v4 node routes: left to the Python reader
        return
    assert compiled is not None
    _same_read(compiled, _read_python(path, net, keep_routes))


@pytest.mark.parametrize("text, falls_back", [
    # a cut before "<person" in a comment leaves a chunk that does not parse
    ('<!-- <person id="no"> --><person id="b">{plan}</person>', True),
    ('<![CDATA[ <person id="no"> ]]><person id="b">{plan}</person>', True),
    ('<person id="b"><attributes><attribute name="n" class="java.lang.String">'
     '&lt;person id="no"&gt;</attribute></attributes>{plan}</person>', False),
    ('<person id="b">{plan}</person></population><!-- after -->', True),
    ('<person id="b"><attributes><attribute name="n" class="java.lang.Integer">1_000</attribute>'
     '</attributes>{plan}</person>', True),  # an int only Python reads
])
def test_compiled_plans_reader_falls_back(equil_scenario, tmp_path, text, falls_back):
    """Whatever the compiled reader hands over, the Python reader reads."""
    pytest.importorskip("mobslim_io")
    from mobslim.formats.matsim_plans import _read_compiled, _read_python

    plan = '<plan><act type="h" link="1" end_time="06:00"/><leg mode="car"/><act type="w" link="20"/></plan>'
    body = f'<person id="a">{plan}</person>' + text.format(plan=plan)
    if "</population>" not in body:
        body += "</population>"
    path = tmp_path / "plans.xml"
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n<population>' + body)
    net = equil_scenario.network
    compiled = _read_compiled(path, net, False, chunk_bytes=1)
    assert (compiled is None) == falls_back
    pop = read_population_xml(path, net)
    assert list(pop.ids) == ["a", "b"]
    if compiled is not None:
        _same_read(compiled, _read_python(path, net, False))


def test_compiled_plans_reader_latin1(equil_scenario, tmp_path):
    """Other encodings are left to the Python reader."""
    pytest.importorskip("mobslim_io")
    from mobslim.formats.matsim_plans import _read_compiled

    path = tmp_path / "plans.xml"
    path.write_bytes('<?xml version="1.0" encoding="ISO-8859-1"?>\n<plans><person id="\u00e9">'
                     '<plan><act type="h" link="1"/></plan></person></plans>'.encode("latin-1"))
    assert _read_compiled(path, equil_scenario.network, False) is None
    assert list(read_population_xml(path, equil_scenario.network).ids) == ["\u00e9"]


def test_read_person_attributes_xml_streamed(tmp_path):
    from mobslim.formats.matsim_attributes import read_person_attributes_xml

    (tmp_path / "attrs.xml").write_text(_OBJ)
    assert read_person_attributes_xml(tmp_path / "attrs.xml") == {
        "a": {"age": 32}, "b": {"carAvail": "never"}, "c": {"age": 5}}


def test_iter_root_children_streams(tmp_path):
    """Only the root's own children of the tag; each is cleared and dropped
    from the root once the next is asked for."""
    import io
    import xml.etree.ElementTree as ET

    from mobslim.formats._xmlstream import iter_root_children

    xml = (b'<population><attributes><person id="nested"/></attributes>'
           b'<person id="1"><plan/></person><!-- c --><other/><person id="2"/></population>')
    seen = []
    for el in iter_root_children(io.BytesIO(xml), "person"):
        if seen:
            assert len(seen[-1][0]) == 0 and seen[-1][0].get("id") is None  # cleared
        seen.append((el, el.get("id"), len(el)))
    assert [s[1:] for s in seen] == [("1", 1), ("2", 0)]
    with pytest.raises(ET.ParseError):
        list(iter_root_children(io.BytesIO(b""), "person"))


def test_with_elements_is_validated():
    plan = Plan.builder().act("h", 0, end_time=1).leg("walk").act("w", 1).build()
    assert plan.with_elements(plan.elements) == plan
    with pytest.raises(ValueError):
        plan.with_elements(plan.elements[:2])  # still validated


def test_person_select_validates_index():
    plan = Plan.builder().act("h", 0).build()
    person = Person("p", (plan, plan))
    assert person.select(1).selected == 1
    assert person.select(0) is person
    with pytest.raises(ValueError):
        person.select(2)


# ------------------------------------------------------- packed routes


def test_route_accepts_ints_arrays_and_packed_bytes():
    import numpy as np

    from mobslim.population import Route

    a = Route((3, 1, 2), 10.0, 30.0)
    b = Route(np.array([3, 1, 2], dtype=np.int64), 10.0, 30.0)
    c = Route(np.array([3, 1, 2], dtype=np.int32).tobytes(), 10.0, 30.0)
    d = Route.packed(c.data, 10.0, 30.0)
    assert a == b == c == d and len({a, b, c, d}) == 1
    assert a.links == (3, 1, 2) and a.n_links == 3
    assert a != Route((3, 1, 2), 11.0, 30.0) and a != Route((3, 1), 10.0, 30.0)
    assert repr(a) == "Route(links=(3, 1, 2), travel_time=10.0, distance=30.0)"
    view = a.link_array
    assert view.dtype == np.int32 and view.itemsize == 4 and view.tolist() == [3, 1, 2]
    assert a.data == np.array([3, 1, 2], dtype="=i4").tobytes()
    assert not view.flags.writeable
    assert Route((), 0.0, 0.0).links == () and Route((), 0.0, 0.0).n_links == 0


def test_route_is_immutable_and_pickles():
    import dataclasses
    import pickle

    from mobslim.population import Route

    r = Route((5, 6), 1.0, 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.travel_time = 3.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        del r.data
    assert pickle.loads(pickle.dumps(r)) == r


def test_route_rejects_bad_links():
    import numpy as np

    from mobslim.population import Route

    with pytest.raises(ValueError, match="multiple of 4"):
        Route(b"abc", 0.0, 0.0)
    with pytest.raises(TypeError):
        Route(np.array([1.5]), 0.0, 0.0)
    with pytest.raises(ValueError, match="int32"):
        Route(np.array([-1]), 0.0, 0.0)
    with pytest.raises(ValueError, match="int32"):
        Route((2**40,), 0.0, 0.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        Route(np.zeros((2, 2), dtype=np.int32), 0.0, 0.0)


def test_plan_batch_joins_packed_routes():
    import numpy as np

    from mobslim.planbatch import PlanBatch
    from mobslim.population import Route

    plans = [
        Plan.builder().act("h", 0, end_time=1).leg("car", Route((4, 5, 6), 1, 1))
        .act("w", 1, end_time=2).leg("walk").act("h", 0).build(),
        Plan.builder().act("h", 2, end_time=1).leg("car", Route((), 0, 0))
        .act("w", 2, end_time=2).leg("car", Route(np.array([9]), 1, 1)).act("h", 3).build(),
    ]
    batch = PlanBatch.from_plans(plans)
    assert batch.route_links.dtype == np.int32
    assert batch.route_links.tolist() == [4, 5, 6, 9]
    assert batch.route_offsets.tolist() == [0, 3, 3, 3, 4]
    assert batch.leg_routed.tolist() == [True, False, True, True]
    assert PlanBatch.from_plans([]).route_links.shape == (0,)


def test_single_leg_trip_takes_its_routing_mode(tmp_path, equil_scenario):
    # MATSim writes a pt trip it walks entirely as one walk leg with routingMode=pt
    (tmp_path / "plans.xml").write_text("""<?xml version="1.0" encoding="utf-8"?>
<population>
  <person id="1">
    <plan selected="yes">
      <activity type="h" node="1" end_time="07:00:00"/>
      <leg mode="walk" trav_time="00:08:27">
        <attributes><attribute name="routingMode" class="java.lang.String">pt</attribute></attributes>
        <route type="generic" start_link="1" end_link="20" trav_time="00:08:27"></route>
      </leg>
      <activity type="w" node="12" end_time="17:00:00"/>
      <leg mode="walk" trav_time="00:08:27">
        <attributes><attribute name="routingMode" class="java.lang.String">walk</attribute></attributes>
      </leg>
      <activity type="h" node="1"/>
    </plan>
  </person>
</population>""")
    for keep_routes in (False, True):
        pop = read_population_xml(tmp_path / "plans.xml", equil_scenario.network, keep_routes=keep_routes)
        to_work, to_home = pop[0].selected_plan.legs
        assert to_work.mode == "pt" and to_work.route is None
        assert to_home.mode == "walk" and to_home.travel_time == 507.0
