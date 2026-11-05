import math
import pickle
from random import Random

import numpy as np
import pandas as pd
import pytest

from mobslim import (
    ActivityParams,
    Person,
    PersonAttributes,
    Plan,
    Population,
    Scenario,
    ScoringParams,
    read_population_xml,
)
from mobslim.config.matsim import (
    MatsimConfig,
    controller_kwargs,
    scoring_params,
)
from mobslim.events import EventsManager
from mobslim.ids import IdMap
from mobslim.mobsim import QSim, QSimParams
from mobslim.replanning import (
    ChangeTripMode,
    ProcessPoolRunner,
    ReplanningContext,
    SerialRunner,
    StrategyManager,
)
from mobslim.routing import TravelTimeTable
from mobslim.scoring import CharyparNagelScorer, score_all
from tests.conftest import FIXTURES, chain_network, commuters, routed

H = 3600.0


# ------------------------------------------------------------ the table


def test_column_types_and_missing_values():
    ids = IdMap(["a", "b", "c"])
    attrs = PersonAttributes.from_records(ids, {
        "a": {"age": 30, "income": 1.5, "sub": "x", "car": True, "n": 1},
        "b": {"age": 40, "car": False, "n": 2},
        "c": {"age": 50, "n": 3},
        "zz": {"age": 99},  # not a person: ignored
    })
    assert attrs.names == ("age", "income", "sub", "car", "n")
    assert attrs["age"].dtype == np.int64 and attrs["age"].tolist() == [30, 40, 50]
    assert attrs["income"].dtype == float and math.isnan(attrs["income"][1])
    assert attrs["sub"].dtype == object and attrs["sub"].tolist() == ["x", None, None]
    assert attrs["car"].dtype == object  # bool with a missing value
    assert attrs.missing("sub").tolist() == [False, True, True]
    assert attrs.missing("income").tolist() == [False, True, True]
    assert attrs.missing("nope").all()
    assert attrs.row(1) == {"age": 40, "car": False, "n": 2}
    assert attrs.value(0, "income") == 1.5 and attrs.value(2, "income", -1) == -1
    with pytest.raises(ValueError):
        attrs["age"][0] = 1
    with pytest.raises(KeyError):
        attrs["nope"]


def test_frame_round_trip_and_updates():
    ids = IdMap(["a", "b"])
    df = pd.DataFrame({"age": [30, 40], "sub": ["x", None]}, index=["b", "a"])
    attrs = PersonAttributes.from_frame(df, ids)
    assert attrs["age"].tolist() == [40, 30]  # aligned to the person index
    assert attrs["sub"].tolist() == [None, "x"]
    frame = attrs.to_frame(ids)
    assert list(frame.index) == ["a", "b"] and frame.loc["b", "sub"] == "x"
    assert PersonAttributes.from_frame(frame, ids) == attrs
    more = attrs.with_column("w", [1.0, 2.0])
    assert "w" in more and "w" not in attrs
    assert more.without("w") == attrs
    merged = attrs.merged(PersonAttributes(2, {"sub": ["y", None], "k": [1, 2]}))
    assert merged["sub"].tolist() == ["y", "x"] and merged["k"].tolist() == [1, 2]
    assert pickle.loads(pickle.dumps(attrs)) == attrs
    with pytest.raises(ValueError):
        PersonAttributes(3, {"x": [1, 2]})


def test_population_keeps_attributes_through_updates():
    pop = commuters(3, 0, 1)
    assert len(pop.attributes) == 0 and pop.attributes.n_persons == 3
    pop = pop.with_attributes({0: {"sub": "a"}, 2: {"sub": "b"}})
    assert pop.attributes["sub"].tolist() == ["a", None, "b"]
    assert pop.with_selected_scores([1.0, 2.0, 3.0]).attributes is pop.attributes
    assert pop.replace(list(pop)).attributes is pop.attributes
    assert pop.map_plans(lambda p: p).attributes is pop.attributes
    assert pickle.loads(pickle.dumps(pop)).attributes == pop.attributes
    with pytest.raises(ValueError):
        pop.with_attributes(PersonAttributes(2))


# ------------------------------------------------------------ MATSim files

_POP_V6 = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">
<population>
  <attributes>
    <attribute name="coordinateReferenceSystem" class="java.lang.String">EPSG:27700</attribute>
  </attributes>
  <person id="p1">
    <attributes>
      <attribute name="subpopulation" class="java.lang.String">freight</attribute>
      <attribute name="age" class="java.lang.Integer">42</attribute>
      <attribute name="income" class="java.lang.Double">1234.5</attribute>
      <attribute name="hasLicense" class="java.lang.Boolean">true</attribute>
    </attributes>
    <plan selected="yes">
      <activity type="h" node="1" end_time="06:00:00"/>
      <leg mode="walk"/>
      <activity type="h" node="1"/>
    </plan>
  </person>
  <person id="p2">
    <plan selected="yes">
      <activity type="h" node="1" end_time="06:00:00"/>
      <leg mode="walk"/>
      <activity type="h" node="1"/>
    </plan>
  </person>
</population>
"""

_OBJECT_ATTRS = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE objectAttributes SYSTEM "http://matsim.org/files/dtd/objectattributes_v1.dtd">
<objectAttributes>
  <object id="p2">
    <attribute name="subpopulation" class="java.lang.String">commuter</attribute>
  </object>
  <object id="p1">
    <attribute name="age" class="java.lang.Integer">43</attribute>
  </object>
  <object id="ghost">
    <attribute name="age" class="java.lang.Integer">1</attribute>
  </object>
</objectAttributes>
"""


def test_read_population_v6_and_object_attributes(tmp_path, equil_scenario):
    (tmp_path / "pop.xml").write_text(_POP_V6)
    (tmp_path / "attrs.xml").write_text(_OBJECT_ATTRS)
    net = equil_scenario.network
    pop = read_population_xml(tmp_path / "pop.xml", net)
    a = pop.attributes
    assert a.row(0) == {"subpopulation": "freight", "age": 42, "income": 1234.5, "hasLicense": True}
    assert a.row(1) == {}
    pop = read_population_xml(tmp_path / "pop.xml", net, attributes_file=tmp_path / "attrs.xml")
    assert pop.attributes["subpopulation"].tolist() == ["freight", "commuter"]
    assert pop.attributes.value(0, "age") == 43  # the attributes file wins
    assert pop.attributes.missing("age").tolist() == [False, True]


def test_read_plans_v4_person_attributes(tmp_path, equil_scenario):
    (tmp_path / "plans.xml").write_text("""<?xml version="1.0" ?>
<!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">
<plans>
<person id="1" sex="f" age="31" car_avail="always" license="yes" employed="no">
  <plan><act type="h" node="1" end_time="06:00"/><leg mode="walk"/><act type="h" node="1"/></plan>
</person>
</plans>""")
    pop = read_population_xml(tmp_path / "plans.xml", equil_scenario.network)
    assert pop.attributes.row(0) == {"sex": "f", "age": 31, "hasLicense": "yes",
                                    "carAvail": "always", "employed": False}


def test_equil_plans_have_no_attributes(equil_scenario):
    assert len(equil_scenario.population.attributes) == 0


# ------------------------------------------------------------ scoring


def _params(performing):
    return ScoringParams(performing=performing,
                         activities={"home": ActivityParams(12 * H), "work": ActivityParams(8 * H)},
                         default_activity=None)


def _run(pop):
    net = chain_network([100, 100], freespeed=10)
    scenario = routed(Scenario(net, pop))
    return QSim(QSimParams()).run(scenario, EventsManager()), scenario.population


def test_subpopulation_scoring_params():
    pop = commuters(3, 0, 2, dep=8 * H).with_attributes(
        {1: {"subpopulation": "lazy"}, 2: {"subpopulation": "keen"}})
    tr, pop = _run(pop)
    base = CharyparNagelScorer(_params(6.0))
    scorer = CharyparNagelScorer(_params(6.0), subpopulations={"lazy": _params(1.0),
                                                               "keen": _params(12.0)})
    plain = score_all(base, tr)
    scores = score_all(scorer, tr, pop.attributes)
    lazy = score_all(CharyparNagelScorer(_params(1.0)), tr)
    keen = score_all(CharyparNagelScorer(_params(12.0)), tr)
    np.testing.assert_allclose(scores, [plain[0], lazy[1], keen[2]])
    # the per-plan path agrees
    plans = tr.experienced_plans()
    per_plan = [scorer.score(p, attributes=pop.attributes.row(i)) for i, p in enumerate(plans)]
    np.testing.assert_allclose(per_plan, scores)
    # without attributes everyone gets the default params
    np.testing.assert_allclose(score_all(scorer, tr), plain)


def test_subpopulation_activity_types_need_only_their_own_params():
    freight = ScoringParams(activities={"home": ActivityParams(12 * H)}, default_activity=None)
    person = ScoringParams(activities={"work": ActivityParams(8 * H)}, default_activity=None)
    pop = commuters(2, 0, 2, dep=8 * H).with_attributes({0: {"subpopulation": "freight"}})
    tr, pop = _run(pop)
    scorer = CharyparNagelScorer(person, subpopulations={"freight": freight})
    # person 1 (default params) does "home", which only freight's params know
    with pytest.raises(KeyError, match="home"):
        score_all(scorer, tr, pop.attributes)


def test_unknown_subpopulation_raises():
    pop = commuters(1, 0, 2).with_attributes({0: {"subpopulation": "ghost"}})
    tr, pop = _run(pop)
    scorer = CharyparNagelScorer(_params(6.0), subpopulations={"x": _params(1.0)})
    with pytest.raises(KeyError, match="ghost"):
        score_all(scorer, tr, pop.attributes)


def test_custom_scorer_receives_attributes():
    class AgeScorer:
        def score(self, plan, attributes=None):
            return float(attributes.get("age", 0))

    pop = commuters(2, 0, 2).with_attributes({1: {"age": 7}})
    tr, pop = _run(pop)
    assert score_all(AgeScorer(), tr, pop.attributes).tolist() == [0.0, 7.0]


# ------------------------------------------------------------ MATSim config

_CONFIG = """<?xml version="1.0" ?>
<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">
<config>
  <module name="network"><param name="inputNetworkFile" value="{net}"/></module>
  <module name="plans">
    <param name="inputPlansFile" value="pop.xml"/>
    <param name="inputPersonAttributesFile" value="attrs.xml"/>
  </module>
  <module name="scoring">
    <parameterset type="scoringParameters">
      <param name="performing" value="6.0"/>
      <parameterset type="activityParams">
        <param name="activityType" value="h"/><param name="typicalDuration" value="12:00:00"/>
      </parameterset>
    </parameterset>
    <parameterset type="scoringParameters">
      <param name="subpopulation" value="freight"/>
      <param name="performing" value="2.0"/>
      <parameterset type="activityParams">
        <param name="activityType" value="h"/><param name="typicalDuration" value="12:00:00"/>
      </parameterset>
    </parameterset>
    <parameterset type="scoringParameters">
      <param name="subpopulation" value="commuter"/>
      <parameterset type="activityParams">
        <param name="activityType" value="h"/><param name="typicalDuration" value="12:00:00"/>
      </parameterset>
    </parameterset>
  </module>
  <module name="replanning">
    <parameterset type="strategysettings">
      <param name="strategyName" value="BestScore"/><param name="weight" value="1.0"/>
    </parameterset>
  </module>
</config>
"""


def test_matsim_config_reads_attributes_and_subpopulation_scoring(tmp_path):
    (tmp_path / "pop.xml").write_text(_POP_V6)
    (tmp_path / "attrs.xml").write_text(_OBJECT_ATTRS)
    (tmp_path / "config.xml").write_text(
        _CONFIG.format(net=FIXTURES / "equil" / "network.xml"))
    cfg = MatsimConfig.load(tmp_path / "config.xml")
    kwargs, unsupported = controller_kwargs(cfg)
    assert not [u for u in unsupported if "ttributes" in u or "subpopulation" in u]
    attrs = kwargs["scenario"].population.attributes
    assert attrs["subpopulation"].tolist() == ["freight", "commuter"]
    scorer = kwargs["scorer"]
    assert scorer.params.performing == 6.0
    assert scorer.subpopulations["freight"].performing == 2.0
    assert set(scorer.subpopulations) == {"freight", "commuter"}
    # the default-only translator still flags the subpopulation sets
    _, u = scoring_params(cfg)
    assert any("subpopulation freight" in x for x in u)


# ------------------------------------------------------------ replanning


class PickByAttribute:
    """Selects the plan named by the person's ``pick`` attribute (default 0)."""

    is_innovative = False

    def apply(self, person, ctx, rng):
        return person.select(int(ctx.attribute("pick", 0)))


def _two_plan_population(n):
    plans = [Plan.builder().act("h", i % 2, end_time=0).leg("car").act("w", 0).build()
             for i in range(2)]
    return Population([Person(i, tuple(plans)) for i in range(n)])


def test_context_binds_person_attributes(equil_scenario):
    pop = commuters(3, 1, 2).with_attributes({1: {"carAvail": "never", "age": 9}})
    ctx = ReplanningContext.from_scenario(Scenario(equil_scenario.network, pop),
                                          TravelTimeTable.free_speed(equil_scenario.network))
    assert ctx.attributes is pop.attributes
    with pytest.raises(RuntimeError):
        ctx.attribute("age")
    bound = ctx.for_person(1)
    assert bound.attribute("carAvail") == "never" and bound.person_attributes() == {
        "carAvail": "never", "age": 9}
    assert ctx.for_person(0).attribute("age", -1) == -1
    assert bound._routers is ctx._routers and bound.for_person(1) is bound
    assert pickle.loads(pickle.dumps(bound)).attribute("age") == 9


@pytest.mark.parametrize("runner", ["serial", "pool"])
def test_runners_give_strategies_their_persons_attributes(equil_scenario, runner):
    n = 7
    pop = _two_plan_population(n).with_attributes({i: {"pick": 1} for i in (1, 4, 6)})
    scenario = Scenario(equil_scenario.network, pop)
    ctx = ReplanningContext.from_scenario(scenario, TravelTimeTable.free_speed(scenario.network), 1)
    manager = StrategyManager([(1.0, PickByAttribute())])
    if runner == "serial":
        out = SerialRunner().run(pop, manager, ctx, seed=1)
    else:
        with ProcessPoolRunner(workers=2, chunks_per_worker=2) as r:
            out = r.run(pop, manager, ctx, seed=1)
    assert out.selected.tolist() == [0, 1, 0, 0, 1, 0, 1]
    assert out.attributes is pop.attributes


def test_change_trip_mode_respects_car_availability(equil_scenario):
    pop = commuters(3, 1, 2, mode="walk").with_attributes(
        {0: {"carAvail": "never"}, 1: {"hasLicense": "no"}, 2: {"carAvail": "always"}})
    ctx = ReplanningContext.from_scenario(Scenario(equil_scenario.network, pop),
                                          TravelTimeTable.free_speed(equil_scenario.network))
    plans = pop.selected_plans()
    strict = ChangeTripMode(["car", "walk", "bike"], ignore_car_availability=False)
    for i in (0, 1):
        modes = {strict.mutate(plans[i], ctx.for_person(i), Random(s)).legs[0].mode
                 for s in range(30)}
        assert modes == {"bike"}
    modes = {strict.mutate(plans[2], ctx.for_person(2), Random(s)).legs[0].mode for s in range(30)}
    assert modes == {"car", "bike"}
    # MATSim's default ignores availability
    loose = ChangeTripMode(["car", "walk", "bike"])
    modes = {loose.mutate(plans[0], ctx.for_person(0), Random(s)).legs[0].mode for s in range(30)}
    assert modes == {"car", "bike"}
    # nothing left to choose: unchanged
    only_car = ChangeTripMode(["car", "walk"], ignore_car_availability=False)
    assert only_car.mutate(plans[0], ctx.for_person(0), Random(0)) is plans[0]
