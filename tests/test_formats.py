import numpy as np
import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from mobslim import route_scenario  # noqa: E402
from mobslim.formats import (  # noqa: E402
    population_format,
    read_population,
    write_population,
)
from mobslim.formats.__main__ import main as cli  # noqa: E402
from mobslim.formats.matsim import format_matsim_time  # noqa: E402
from mobslim.population import (  # noqa: E402
    Activity,
    Leg,
    Person,
    Plan,
    Population,
    read_population_xml,
)
from tests.conftest import FIXTURES  # noqa: E402
from tests.test_transit_scenarios import load  # noqa: E402


def rich_population(scenario) -> Population:
    """Routed equil plans plus: a second plan for some persons (some scored,
    the second selected), activities without coordinates or with links,
    a teleported leg, and attributes of every kind with missing values."""
    pop = route_scenario(scenario).population
    persons = []
    for i, person in enumerate(pop):
        if i % 3 == 0:
            plan = person.selected_plan
            alt = plan.with_elements(
                [Leg("walk", travel_time=123.5) if k == 1 else e for k, e in enumerate(plan.elements)])
            person = person.with_selected_score(float(i)).add_plan(alt, select=i % 2 == 0)
        persons.append(person)
    n = len(pop)
    attrs = {
        "age": list(range(n)),
        "income": [None if i % 4 == 0 else i * 1.5 for i in range(n)],
        "car": [i % 2 == 0 for i in range(n)],
        "licence": [None if i % 5 == 0 else i % 3 == 0 for i in range(n)],
        "subpopulation": [None if i % 7 == 0 else ("a" if i % 2 else "b") for i in range(n)],
    }
    return pop.replace(persons).with_attributes(
        {pid: {k: v[i] for k, v in attrs.items() if v[i] is not None}
         for i, pid in enumerate(pop.ids)})


def same_population(a: Population, b: Population) -> None:
    assert list(a.ids) == list(b.ids)
    assert a.n_plans.tolist() == b.n_plans.tolist()
    assert a.selected.tolist() == b.selected.tolist()
    assert np.array_equal(a.scores, b.scores, equal_nan=True)
    assert list(a) == list(b)
    assert a.attributes == b.attributes


@pytest.fixture
def equil(equil_scenario):
    return equil_scenario, rich_population(equil_scenario)


def test_parquet_round_trip(tmp_path, equil):
    scenario, pop = equil
    out = write_population(pop, tmp_path / "pop", scenario.network)
    assert sorted(p.name for p in out.iterdir()) == [
        "activities.parquet", "legs.parquet", "persons.parquet", "plans.parquet"]
    back = read_population(out, scenario.network)
    same_population(pop, back)
    assert back.attributes["age"].dtype == np.int64
    assert back.attributes["income"].dtype == np.float64
    assert back.attributes["car"].dtype == bool


def test_parquet_tables_are_readable_and_ordered(tmp_path, equil):
    scenario, pop = equil
    out = write_population(pop, tmp_path / "pop", scenario.network)
    legs = pq.read_table(out / "legs.parquet").to_pylist()
    first = legs[0]
    assert first["person"] == "1" and first["plan"] == 0 and first["seq"] == 0
    route = pop[0].plans[0].legs[0].route
    assert first["mode"] == "car" and first["route"] == [
        scenario.network.link_ids.id(k) for k in route.links]
    assert first["route_time"] == route.travel_time
    acts = pq.read_table(out / "activities.parquet").to_pylist()
    assert acts[0]["node"] == "2" and acts[0]["end_time"] == 6 * 3600
    assert acts[1]["duration"] == 600 and acts[1]["end_time"] is None


def test_parquet_reader_sorts_shuffled_rows(tmp_path, equil):
    scenario, pop = equil
    out = write_population(pop, tmp_path / "pop", scenario.network)
    rng = np.random.default_rng(0)
    for name in ("plans", "activities", "legs"):
        table = pq.read_table(out / f"{name}.parquet")
        pq.write_table(table.take(pa.array(rng.permutation(len(table)))), out / f"{name}.parquet")
    same_population(pop, read_population(out, scenario.network))


def test_parquet_drop_routes(tmp_path, equil):
    scenario, pop = equil
    out = write_population(pop, tmp_path / "pop", scenario.network)
    back = read_population(out, scenario.network, keep_routes=False)
    assert all(leg.route is None for p in back.selected_plans() for leg in p.legs)


def test_parquet_location_fallbacks(tmp_path):
    """No node: the link's to-node, else the node nearest x/y."""
    from mobslim import read_network_xml
    from mobslim.formats.parquet import read_population_parquet

    net = read_network_xml(FIXTURES / "equil" / "network.xml")
    d = tmp_path / "pop"
    d.mkdir()
    pq.write_table(pa.table({"person": ["a"]}), d / "persons.parquet")
    pq.write_table(pa.table({"person": ["a"], "plan": [0], "score": pa.array([None], pa.float64()),
                             "selected": [True]}), d / "plans.parquet")
    pq.write_table(pa.table({
        "person": ["a", "a"], "plan": [0, 0], "seq": [0, 1], "type": ["h", "w"],
        "node": pa.array([None, None], pa.string()), "link": pa.array(["1", None]),
        "x": pa.array([None, 4950.0]), "y": pa.array([None, -150.0]),
        "end_time": pa.array([3600.0, None]), "duration": pa.array([None, None], pa.float64()),
    }), d / "activities.parquet")
    pq.write_table(pa.table({
        "person": ["a"], "plan": [0], "seq": [0], "mode": ["car"],
        "travel_time": pa.array([None], pa.float64()), "route_time": pa.array([None], pa.float64()),
        "route_distance": pa.array([None], pa.float64()),
        "route": pa.array([None], pa.list_(pa.string())),
        "rides": pa.array([None], pa.list_(pa.struct([("line", pa.string()), ("route", pa.string()),
                                                      ("board", pa.int32()), ("alight", pa.int32())]))),
    }), d / "legs.parquet")
    plan = read_population_parquet(d, net).selected_plans()[0]
    assert plan.activities[0].node == int(net.to_node[net.link_index("1")])
    assert plan.activities[0].link == net.link_index("1")
    assert plan.activities[1].node == net.node_index("13")
    assert plan.legs[0].route is None


def test_parquet_rejects_inconsistent_tables(tmp_path, equil):
    scenario, pop = equil
    out = write_population(pop, tmp_path / "pop", scenario.network)
    legs = pq.read_table(out / "legs.parquet")
    pq.write_table(legs.slice(1), out / "legs.parquet")
    with pytest.raises(ValueError, match="one leg fewer"):
        read_population(out, scenario.network)


def test_parquet_keeps_pt_rides_with_a_schedule(tmp_path, caplog):
    scenario = route_scenario(load("pt_tutorial"))
    pop = scenario.population
    rides = [leg.route for p in pop.selected_plans() for leg in p.legs if leg.mode == "pt"]
    assert any(r is not None and r.n_rides for r in rides)
    out = write_population(pop, tmp_path / "pop", scenario.network,
                           transit_schedule=scenario.transit)
    same_population(pop, read_population(out, scenario.network, transit_schedule=scenario.transit))
    without = read_population(out, scenario.network)  # no schedule: rides dropped, routed again
    assert "pt rides" in caplog.text
    assert all(leg.route is None for p in without.selected_plans() for leg in p.legs
               if leg.mode == "pt")


def test_xml_round_trip(tmp_path, equil):
    scenario, pop = equil
    for name in ("plans.xml", "plans.xml.gz"):
        out = write_population(pop, tmp_path / name, scenario.network)
        back = read_population_xml(out, scenario.network, keep_routes=True)
        assert list(back.ids) == list(pop.ids)
        assert back.selected.tolist() == pop.selected.tolist()
        assert np.array_equal(back.scores, pop.scores, equal_nan=True)
        assert back.attributes == pop.attributes
        for a, b in zip(pop, back):
            for pa_, pb in zip(a.plans, b.plans):
                assert [(x.type, x.node, x.end_time, x.duration) for x in pa_.activities] == \
                       [(x.type, x.node, x.end_time, x.duration) for x in pb.activities]
                # equil's activities have nodes, no links: MATSim routes need them
                assert all(x.link is None for x in pa_.activities)
                assert [(leg.mode, leg.travel_time) for leg in pa_.legs] == \
                       [(leg.mode, leg.travel_time) for leg in pb.legs]
                assert all(leg.route is None for leg in pb.legs)


def test_xml_writes_routes_only_between_activity_links(tmp_path):
    from mobslim import read_network_xml
    from mobslim.population import Route

    net = read_network_xml(FIXTURES / "equil" / "network.xml")
    l1, l6, l11 = (net.link_index(i) for i in ("1", "6", "11"))
    with_links = Plan((Activity("h", int(net.to_node[l1]), end_time=10.0, link=l1),
                       Leg("car", Route([l6, l11], 5.0, 7.0)),
                       Activity("w", int(net.to_node[l11]), link=l11)))
    no_links = Plan((Activity("h", int(net.to_node[l1]), end_time=10.0),
                     Leg("car", Route([l6, l11], 5.0, 7.0)),
                     Activity("w", int(net.to_node[l11]))))
    pop = Population([Person("a", (with_links,)), Person("b", (no_links,))])
    text = write_population(pop, tmp_path / "p.xml", net).read_text()
    assert '<route type="links" start_link="1" end_link="11" trav_time="00:00:05" ' \
           'distance="7.0">1 6 11</route>' in text
    assert text.count("<route") == 1
    back = read_population_xml(tmp_path / "p.xml", net, keep_routes=True)
    assert back[0].selected_plan.legs[0].route.links == (l6, l11)


def test_format_matsim_time():
    assert format_matsim_time(3600) == "01:00:00"
    assert format_matsim_time(90000.25) == "25:00:00.25"
    assert format_matsim_time(59.9999999) == "00:01:00"


def test_population_format():
    assert population_format("a.xml") == "xml" and population_format("a.XML.gz") == "xml"
    assert population_format("somewhere/new") == "parquet"
    assert population_format("a.xml", "parquet") == "parquet"
    with pytest.raises(ValueError):
        population_format("x", "csv")


def test_cli_convert_round_trip(tmp_path, capsys):
    net = str(FIXTURES / "equil" / "network.xml")
    plans = str(FIXTURES / "equil" / "plans100.xml")
    assert cli(["convert", plans, str(tmp_path / "pq"), "--network", net]) == 0
    assert cli(["convert", str(tmp_path / "pq"), str(tmp_path / "out.xml.gz"), "--network", net]) == 0
    assert cli(["info", str(tmp_path / "pq"), "--network", net]) == 0
    assert "activity types: h, w" in capsys.readouterr().out
    from mobslim import read_network_xml

    network = read_network_xml(net)
    a = read_population_xml(plans, network, keep_routes=True)
    b = read_population(tmp_path / "pq", network)
    same_population(a, b)
    c = read_population(tmp_path / "out.xml.gz", network)
    assert [[x.node for x in p.activities] for p in a.selected_plans()] == \
           [[x.node for x in p.activities] for p in c.selected_plans()]


def test_config_reads_parquet_plans(tmp_path):
    from mobslim import read_network_xml
    from mobslim.config.mobslim import MobslimConfig

    net = FIXTURES / "equil" / "network.xml"
    network = read_network_xml(net)
    pop = read_population_xml(FIXTURES / "equil" / "plans100.xml", network, keep_routes=True)
    write_population(pop, tmp_path / "plans", network)
    cfg = MobslimConfig.from_dict({"inputs": {"network": str(net), "plans": str(tmp_path / "plans")}},
                                  base_dir=tmp_path)
    same_population(pop, cfg.read_inputs().population)
