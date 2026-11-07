"""The mobslim YAML config: conditions, loading, defaults, precheck, output
config, MATSim equivalence, and the per-person behaviour it configures."""

import logging
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from mobslim import Controller, ControllerParams
from mobslim.attributes import PersonAttributes
from mobslim.config.__main__ import main as cli
from mobslim.config.conditions import (
    ConditionError,
    check_condition,
    evaluate,
    parse_condition,
)
from mobslim.config.convert import matsim_to_mobslim
from mobslim.config.matsim import MatsimConfig
from mobslim.config.mobslim import (
    CAR_AVAILABILITY,
    ConfigError,
    Filtered,
    MobslimConfig,
)
from mobslim.events import EventsManager
from mobslim.mobsim import shard
from mobslim.replanning import PersonStrategyManager, route_scenario
from mobslim.replanning.context import ReplanningContext
from mobslim.replanning.modules import routing_groups
from mobslim.routing import TravelTimeTable
from tests.conftest import FIXTURES

EQUIL_NET = FIXTURES / "equil" / "network.xml"


# ------------------------------------------------------------------ helpers

def _write_inputs(d: Path, n: int = 30) -> None:
    """Plans on the equil network (alternating car and walk commuters) and
    attributes: age 20 + 2i, has_license except every third, every fifth freight."""
    persons = []
    for i in range(n):
        mode = "walk" if i % 2 else "car"
        persons.append(f"""<person id="p{i}"><plan>
  <act type="h" x="-15050" y="-150" node="2" end_time="06:{i % 60:02d}" />
  <leg mode="{mode}"/>
  <act type="w" x="-14050" y="-150" node="3" dur="08:00" />
  <leg mode="{mode}"/>
  <act type="h" x="-15050" y="-150" node="2" />
</plan></person>""")
    (d / "plans.xml").write_text(
        '<?xml version="1.0" ?>\n<!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">\n'
        "<plans>\n" + "\n".join(persons) + "\n</plans>\n")
    objects = []
    for i in range(n):
        sub = '<attribute name="subpopulation" class="java.lang.String">freight</attribute>' \
            if i % 5 == 0 else ""
        objects.append(f"""<object id="p{i}">
  <attribute name="age" class="java.lang.Integer">{20 + 2 * i}</attribute>
  <attribute name="has_license" class="java.lang.Boolean">{"false" if i % 3 == 0 else "true"}</attribute>
  {sub}
</object>""")
    (d / "attrs.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<objectAttributes>\n' + "\n".join(objects)
        + "\n</objectAttributes>\n")


def _config(d: Path, extra: str = "", iterations: int = 2) -> Path:
    _write_inputs(d)
    text = textwrap.dedent(f"""\
        run:
          iterations: {iterations}
          output_dir: out
        inputs:
          network: {EQUIL_NET}
          plans: plans.xml
          person_attributes: attrs.xml
        scoring:
          activities:
            h: {{typical_duration: "12:00:00"}}
            w: {{typical_duration: "08:00:00"}}
        """)
    path = d / "config.yaml"
    path.write_text(_merge(text, extra))
    return path


def _merge(base: str, extra: str) -> str:
    """Deep-merge two YAML documents (extra wins)."""
    def merge(a, b):
        for k, v in b.items():
            a[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return a

    return yaml.safe_dump(merge(yaml.safe_load(base), yaml.safe_load(textwrap.dedent(extra)) or {}),
                          sort_keys=False)


def _problems(fn) -> str:
    with pytest.raises(ConfigError) as e:
        fn()
    return "\n".join(e.value.problems)


# --------------------------------------------------------------- conditions

def test_condition_forms_agree():
    a = PersonAttributes(5, {"age": [20, 55, 80, None, 51], "lic": [True, False, True, True, False],
                             "sub": ["a", None, "b", "a", "b"]})
    pairs = [
        ("age > 50", {"age": {"gt": 50}}),
        ("18 <= age < 70", {"age": {"ge": 18, "lt": 70}}),
        ("lic == False and age >= 18", {"all": [{"lic": False}, {"age": {"ge": 18}}]}),
        ("sub in ['a', 'b']", {"sub": {"in": ["a", "b"]}}),
        ("sub != 'a' or age < 30", {"any": [{"sub": {"ne": "a"}}, {"age": {"lt": 30}}]}),
        ("not (age > 50)", {"not": {"age": {"gt": 50}}}),
        ("50 < age", {"age": {"gt": 50}}),
    ]
    for text, structured in pairs:
        np.testing.assert_array_equal(evaluate(parse_condition(text), a),
                                      evaluate(parse_condition(structured), a), err_msg=text)
    assert evaluate(parse_condition("age > 50"), a).tolist() == [False, True, True, False, True]
    # missing values match no comparison, ne included; `not` negates the whole condition
    assert evaluate(parse_condition("sub != 'a'"), a).tolist() == [False, False, True, False, True]
    assert evaluate(parse_condition("not (age > 50)"), a).tolist() == [True, False, False, True, False]


@pytest.mark.parametrize("bad, message", [
    ("age >> 3", "not a comparison"),
    ("age > x", "one attribute name and one literal"),
    ("age", "compare 'age' with a value"),
    ("age > 3 +", "cannot parse"),
    ("__import__('os')", "not a comparison"),
    ("age < 'old'", "expected a number"),
    ("age == None", "null"),
    ({"age": {"gtt": 1}}, "unknown operator"),
    ({"age": {"in": []}}, "non-empty list"),
    ({"all": []}, "non-empty list"),
])
def test_condition_errors(bad, message):
    with pytest.raises(ConditionError, match=message):
        parse_condition(bad)


def test_condition_checks_attributes_and_types():
    a = PersonAttributes(2, {"age": [1, 2], "name": ["x", "y"], "lic": [True, False]})
    assert check_condition(parse_condition("age > 1 and lic == True"), a) == []
    assert "no person attribute 'agee'" in check_condition(parse_condition("agee > 1"), a)[0]
    assert "holds number values, not str" in check_condition(parse_condition("age == 'old'"), a)[0]
    assert "cannot be compared with >" in check_condition(parse_condition("name > 1"), a)[0]
    assert "not number values" in check_condition(parse_condition("lic == 1"), a)[0]


# ----------------------------------------------------------- loading, defaults

def test_defaults_are_matsims_and_logged(tmp_path, caplog):
    cfg = MobslimConfig.load(_config(tmp_path))
    assert cfg.get("scoring.activities.h.zero_utility") == "relative"
    assert cfg.get("mobsim.stuck_time") == 11.0
    assert set(cfg.get("scoring.modes")) >= {"car", "pt", "walk", "bike", "ride", "other"}
    assert "scoring.performing" in cfg.defaults and "inputs.network" not in cfg.defaults
    with caplog.at_level(logging.INFO, logger="mobslim.config"):
        cfg.log_defaults()
    assert "default: scoring.performing = 6.0" in caplog.text
    assert f"{len(cfg.defaults)} defaults filled in" in caplog.text
    params = cfg.scoring_params()
    assert params.stuck_penalty == pytest.approx(-432.0)  # MATSim's abortedPlanScore
    assert params.default_activity is None


def test_load_collects_every_problem(tmp_path):
    path = _config(tmp_path, """
        scoring:
          performin: 6
          modes: {car: {constant: high}}
          activities: {x: {priority: 0}}
        mobsim:
          type: matsim
          flow_capacity_factor: -1
          stuck_time: {default: 10, rules: [{when: age > 1, set: 5}]}
        replanning:
          strategy_sets:
            default: [{name: change_exp_bta, weight: 1}]
            other: [{name: change_trip_mode, weight: 1, modes: [car]}]
        persons:
          car_availability: {default: true, rules: [{when: age >> 1, set: false}, {when: age > 1}]}
        """)
    text = _problems(lambda: MobslimConfig.load(path))
    for expected in ["scoring.performin: unknown parameter; did you mean 'performing'?",
                     "scoring.modes.car.constant: expected a number, got 'high'",
                     "scoring.activities.x.typical_duration: required",
                     "scoring.activities.x.priority: expected a value > 0",
                     "mobsim.type: expected one of qsim, hermes, agentsim, parallel, shard",
                     "mobsim.flow_capacity_factor: expected a value > 0",
                     "mobsim.stuck_time: cannot vary by person",
                     "did you mean 'change_exp_beta'?",
                     "strategy_sets.other[0].modes: needs at least two modes",
                     "car_availability.rules[0].when", "car_availability.rules[1]: a rule is a mapping"]:
        assert expected in text, expected


def test_yaml_yes_no_are_strings(tmp_path):
    path = _config(tmp_path)  # appended as text: the test helper's own YAML would read 'no' as false
    path.write_text(path.read_text() + textwrap.dedent("""
        persons:
          car_availability:
            default: true
            rules: [{when: {hasLicense: no}, set: false}]
        """))
    rule = MobslimConfig.load(path).get("persons.car_availability").rules[0]
    assert rule.when.value == "no"


def test_output_config_round_trips(tmp_path):
    path = _config(tmp_path, """
        scoring:
          modes:
            walk:
              marginal_utility_of_traveling:
                default: -10
                rules: [{when: age > 50, set: -15}, {when: {age: {gt: 70}}, set: -30}]
        replanning:
          strategy_set: {default: default, rules: [{when: subpopulation == 'freight', set: frozen}]}
          strategy_sets:
            frozen: [{name: keep_selected, weight: 1}]
        """)
    cfg = MobslimConfig.load(path)
    text = cfg.to_yaml()
    assert "performing: 6.0  # default" in text
    assert "stuck_penalty: -432.0  # derived" in text
    assert "- when: age > 50" in text
    out = tmp_path / "output_config.yaml"
    out.write_text(text)
    again = MobslimConfig.load(out)

    def body(t):  # without the header and the default / derived notes
        return [line.split("  # ")[0] for line in t.splitlines()[3:]]

    assert body(again.to_yaml()) == body(text)
    assert not again.defaults  # everything is explicit now


def test_output_config_values_reload_identically(tmp_path):
    cfg = MobslimConfig.load(_config(tmp_path, """
        scoring: {performing: {default: 6, rules: [{when: age > 40, set: 5}]}}
        """))
    out = tmp_path / "output_config.yaml"
    out.write_text(cfg.to_yaml())
    again = MobslimConfig.load(out)
    assert again.scoring_params() == cfg.scoring_params()
    assert [(f, v.default, [(str(r.when), r.value) for r in v.rules]) for f, v in again.scoring_rules()] \
        == [(f, v.default, [(str(r.when), r.value) for r in v.rules]) for f, v in cfg.scoring_rules()]
    assert again.get("run") == cfg.get("run")
    assert again.get("replanning") == cfg.get("replanning")


# ------------------------------------------------------------------ precheck

def test_precheck_reports_attributes_types_activities_modes(tmp_path):
    path = _config(tmp_path, """
        scoring:
          activities: {w: null}
          performing: {default: 6, rules: [{when: agee > 50, set: 1}, {when: age == 'old', set: 1}]}
        replanning:
          strategy_sets:
            default: [{name: change_trip_mode, weight: 1, modes: [car, scooter]}]
        modes:
          car: {max_speed: 10}
        """)
    path.write_text(path.read_text().replace("w: null", "s: {typical_duration: 3600}"))
    cfg = MobslimConfig.load(path)
    text = _problems(cfg.build_scenario)
    for expected in ["scoring.performing.rules[0].when: no person attribute 'agee' (attributes are: age, "
                     "has_license, subpopulation)",
                     "scoring.performing.rules[1].when: age == 'old': 'age' holds number values",
                     "no parameters for activity type(s) 'w'",
                     "strategy_sets.default[0].modes: unknown mode(s) 'scooter'",
                     "modes.car: 'car' is a network mode"]:
        assert expected in text, expected


def test_unknown_strategy_set_is_a_load_error(tmp_path):
    path = _config(tmp_path, "replanning: {strategy_set: {default: default, rules: "
                             "[{when: age > 1, set: missing}]}}")
    assert "no strategy set 'missing'" in _problems(lambda: MobslimConfig.load(path))


# --------------------------------------------------------- MATSim equivalence

@pytest.mark.parametrize("name", ["equil-100", "pt-tutorial"])
def test_matsim_config_and_translation_run_identically(name, tmp_path):
    cfg = MatsimConfig.load(FIXTURES / "matsim_reference" / name / "config.xml")
    raw, unsupported = matsim_to_mobslim(cfg)
    assert unsupported == []
    cc = ControllerParams(iterations=3, seed=11, output_dir=None)
    a = Controller.from_matsim_config(cfg, params=cc).run()
    b = Controller.from_mobslim_config(MobslimConfig.from_dict(raw, tmp_path), params=cc).run()
    assert a.metrics == b.metrics
    np.testing.assert_array_equal(a.scenario.population.scores, b.scenario.population.scores)


def test_deterministic_pt_modes_from_matsim(tmp_path):
    """SBB's deterministic pt (``SBBPt.deterministicServiceModes``) becomes
    ``transit.schedule_modes``: pt-tutorial's trains keep the timetable,
    its buses drive in traffic, from the MATSim config and its translation alike."""
    cfg = MatsimConfig.load(FIXTURES / "matsim_reference" / "pt-tutorial" / "config.xml")
    cfg.set("SBBPt", "deterministicServiceModes", "train")
    raw, unsupported = matsim_to_mobslim(cfg)
    assert unsupported == [] and raw["transit"]["schedule_modes"] == ["train"]
    cc = ControllerParams(iterations=1, seed=11, output_dir=None)
    a = Controller.from_matsim_config(cfg, params=cc)
    b = Controller.from_mobslim_config(MobslimConfig.from_dict(raw, tmp_path), params=cc)
    for c in (a, b):
        t = c.scenario.transit
        assert t.params.schedule_modes == ("train",)
        modes = np.array(t.schedule.modes)[t.schedule.route_mode[t.schedule.dep_route]]
        np.testing.assert_array_equal(t.trip_on_network, modes != "train")
    assert a.run().metrics == b.run().metrics


_SUBPOP_CONFIG = """<?xml version="1.0" ?>
<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">
<config>
  <module name="network"><param name="inputNetworkFile" value="{net}"/></module>
  <module name="plans">
    <param name="inputPlansFile" value="plans.xml"/>
    <param name="inputPersonAttributesFile" value="attrs.xml"/>
  </module>
  <module name="changeMode">
    <param name="modes" value="car,walk"/><param name="ignoreCarAvailability" value="false"/>
  </module>
  <module name="scoring">
    <parameterset type="scoringParameters">
      <parameterset type="activityParams">
        <param name="activityType" value="h"/><param name="typicalDuration" value="12:00:00"/>
      </parameterset>
      <parameterset type="activityParams">
        <param name="activityType" value="w"/><param name="typicalDuration" value="08:00:00"/>
      </parameterset>
    </parameterset>
    <parameterset type="scoringParameters">
      <param name="subpopulation" value="freight"/>
      <param name="performing" value="2.0"/>
      <parameterset type="modeParams">
        <param name="mode" value="walk"/><param name="marginalUtilityOfTraveling_util_hr" value="-20"/>
      </parameterset>
      <parameterset type="activityParams">
        <param name="activityType" value="h"/><param name="typicalDuration" value="10:00:00"/>
      </parameterset>
      <parameterset type="activityParams">
        <param name="activityType" value="w"/><param name="typicalDuration" value="08:00:00"/>
      </parameterset>
    </parameterset>
  </module>
  <module name="replanning">
    <parameterset type="strategysettings">
      <param name="strategyName" value="ChangeExpBeta"/><param name="weight" value="0.7"/>
    </parameterset>
    <parameterset type="strategysettings">
      <param name="strategyName" value="ChangeTripMode"/><param name="weight" value="0.3"/>
    </parameterset>
    <parameterset type="strategysettings">
      <param name="strategyName" value="KeepLastSelected"/><param name="weight" value="1"/>
      <param name="subpopulation" value="freight"/>
    </parameterset>
  </module>
</config>
"""


def test_translation_turns_subpopulations_into_rules(tmp_path):
    _write_inputs(tmp_path)
    (tmp_path / "attrs.xml").write_text((tmp_path / "attrs.xml").read_text().replace(
        'name="has_license" class="java.lang.Boolean">false', 'name="hasLicense" class="java.lang.String">no')
        .replace('name="has_license" class="java.lang.Boolean">true', 'name="hasLicense" class="java.lang.String">yes'))
    (tmp_path / "config.xml").write_text(_SUBPOP_CONFIG.format(net=EQUIL_NET))
    matsim = MatsimConfig.load(tmp_path / "config.xml")
    raw, unsupported = matsim_to_mobslim(matsim)
    assert unsupported == []
    s = raw["scoring"]
    assert s["performing"] == {"default": 6.0, "rules": [{"when": "subpopulation == 'freight'", "set": 2.0}]}
    assert s["modes"]["walk"]["marginal_utility_of_traveling"]["rules"][0]["set"] == -20.0
    assert s["activities"]["h"]["typical_duration"]["rules"][0]["set"] == 36000.0
    assert set(raw["replanning"]["strategy_sets"]) == {"default", "freight"}
    assert raw["persons"]["car_availability"]["rules"] == [{"when": "hasLicense == 'no'", "set": False}]

    cfg = MobslimConfig.from_dict(raw, tmp_path)
    scenario = cfg.build_scenario()
    attrs = scenario.population.attributes
    # scoring: the same per-person params as the MATSim scorer's subpopulation sets
    from mobslim.config.matsim import controller_kwargs

    matsim_scorer = controller_kwargs(matsim)[0]["scorer"]
    ours = cfg.scorer().person_params(attrs)
    theirs = matsim_scorer.person_params(attrs)
    for i in range(attrs.n_persons):
        assert ours.params_of(i) == theirs.params_of(i), i
    # strategies: freight keeps its plans
    manager = cfg.strategies(attrs)
    assert isinstance(manager, PersonStrategyManager)
    freight = attrs["subpopulation"] == "freight"
    assert all(manager.names[c] == ("freight" if f else "default")
               for c, f in zip(manager.codes.tolist(), freight.tolist()))
    np.testing.assert_array_equal(attrs[CAR_AVAILABILITY], attrs["hasLicense"] != "no")


# --------------------------------------------------- per-person behaviour

def test_scoring_rules_last_match_wins(tmp_path):
    cfg = MobslimConfig.load(_config(tmp_path, """
        scoring:
          modes:
            walk:
              marginal_utility_of_traveling:
                default: -10
                rules:
                  - when: age > 50
                    set: -15
                  - when: age > 70
                    set: -30
        """))
    scenario = cfg.build_scenario()
    attrs = scenario.population.attributes
    pp = cfg.scorer().person_params(attrs)
    age = attrs["age"]
    expected = np.where(age > 70, -30.0, np.where(age > 50, -15.0, -10.0))
    np.testing.assert_array_equal(pp.value("modes.walk.marginal_utility_of_traveling"), expected)


def test_strategy_sets_per_person(tmp_path):
    path = _config(tmp_path, """
        replanning:
          strategy_set:
            default: busy
            rules: [{when: subpopulation == 'freight', set: frozen}]
          strategy_sets:
            busy: [{name: time_allocation_mutator, weight: 1, mutation_range: 3600}]
            frozen: [{name: keep_selected, weight: 1}]
        """, iterations=3)
    result = Controller.from_mobslim_config(path).run()
    pop = result.scenario.population
    freight = pop.attributes["subpopulation"] == "freight"
    assert (pop.n_plans[freight] == 1).all()
    assert (pop.n_plans[~freight] == 3).all()


def test_car_availability_limits_mode_choice(tmp_path):
    path = _config(tmp_path, """
        persons:
          car_availability:
            default: true
            rules: [{when: has_license == False, set: false}]
        replanning:
          strategy_sets:
            default:
              - {name: change_trip_mode, weight: 1, modes: [car, walk], ignore_car_availability: false}
        """, iterations=3)
    pop = Controller.from_mobslim_config(path).run().scenario.population
    no_car = ~pop.attributes["has_license"]
    for i in np.flatnonzero(no_car):
        person = pop[int(i)]
        new_plans = person.plans[1:]  # the first is the initial plan
        assert new_plans and all(leg.mode != "car" for p in new_plans for leg in p.legs)


def test_mobsim_timing_options(tmp_path):
    cfg = MobslimConfig.load(_config(tmp_path, """
        mobsim: {type: qsim, continuous_time: false, junction_delays: true, rounding: down}
        """))
    p = cfg.mobsim().params
    assert (p.continuous_time, p.junction_delays, p.rounding) == (False, True, "down")
    # null keeps each mobsim's own
    p = MobslimConfig.load(_config(tmp_path, "mobsim: {type: hermes}")).mobsim().params
    assert (p.continuous_time, p.junction_delays, p.rounding) == (False, False, "nearest")
    with pytest.raises(ValueError, match="whole seconds only"):
        MobslimConfig.load(_config(tmp_path, """
            mobsim: {type: hermes, continuous_time: true}
            """)).mobsim()


@pytest.mark.parametrize("mobsim", ["qsim", "hermes", "agentsim", "parallel", "shard"])
def test_per_person_teleport_speed(tmp_path, mobsim):
    if mobsim == "shard" and not shard.rust_available():
        pytest.skip("mobslim_shard extension not built")
    cfg = MobslimConfig.load(_config(tmp_path, f"""
        mobsim: {{type: {mobsim}}}
        modes:
          walk:
            max_speed:
              default: 1.0
              rules: [{{when: age > 50, set: 0.5}}]
        """))
    scenario = route_scenario(cfg.build_scenario())
    tr = cfg.mobsim().run(scenario, EventsManager())
    b = tr.plans
    walk = b.leg_mode == b.modes.index("walk")
    person = b.plan_of_leg()[walk]
    old = scenario.population.attributes["age"][person] > 50
    travel = (tr.leg_arrival - tr.leg_departure)[walk]
    expected = tr.leg_distance[walk] / np.where(old, 0.5, 1.0)
    tol = 1.0 if mobsim in ("hermes", "parallel", "shard") else 1e-6  # whole-second clocks
    np.testing.assert_allclose(travel, expected, atol=tol)
    assert old.any() and (~old).any()


def test_speed_groups_split_routing(tmp_path):
    cfg = MobslimConfig.load(_config(tmp_path, """
        modes:
          walk: {max_speed: {default: 1.0, rules: [{when: age > 50, set: 0.5}]}}
        """))
    scenario = cfg.build_scenario()
    ctx = ReplanningContext.from_scenario(scenario, TravelTimeTable.free_speed(scenario.network))
    codes, weights, persons = routing_groups(ctx)
    age = scenario.population.attributes["age"]
    assert len(set(codes.tolist())) == 2
    assert len(set(codes[age > 50].tolist())) == 1 and len(set(codes[age <= 50].tolist())) == 1
    assert weights == [None, None]
    assert {bool(age[p] > 50) for p in persons} == {True, False}


# ------------------------------------------------------------------ the run

def test_run_writes_output_config(tmp_path, caplog):
    path = _config(tmp_path, iterations=1)
    with caplog.at_level(logging.INFO, logger="mobslim.config"):
        Controller.from_mobslim_config(path).run()
    out = tmp_path / "out" / "output_config.yaml"
    assert out.exists()
    assert "# default" in out.read_text()
    assert "default: mobsim.type = qsim" in caplog.text
    again = MobslimConfig.load(out)
    assert again.scoring_params() == MobslimConfig.load(path).scoring_params()
    assert isinstance(again.get("persons.car_availability"), bool)


def test_filtered_value_without_default_uses_the_schema_default(tmp_path):
    cfg = MobslimConfig.load(_config(tmp_path, "scoring: {performing: {rules: [{when: age > 1, set: 1}]}}"))
    value = cfg.get("scoring.performing")
    assert isinstance(value, Filtered) and value.default == 6.0
    assert "scoring.performing.default" in cfg.defaults


def test_cli(tmp_path, capsys):
    path = _config(tmp_path)
    assert cli(["check", str(path)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text(path.read_text().replace("h:", "home:"))
    assert cli(["check", str(bad)]) == 1
    assert "no parameters for activity type(s) 'h'" in capsys.readouterr().err
    assert cli(["reference"]) == 0
    ref = capsys.readouterr().out
    assert "| `scoring.modes.<mode>.marginal_utility_of_traveling` | float | `-6.0` | yes |" in ref
    out = tmp_path / "translated.yaml"
    assert cli(["from-matsim", str(FIXTURES / "matsim_reference" / "equil-100" / "config.xml"),
                "-o", str(out)]) == 0
    MobslimConfig.load(out)
