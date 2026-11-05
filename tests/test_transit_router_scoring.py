"""The transit router's utilities come from the scoring (as MATSim's SwissRailRaptor)."""
import pickle

import pytest

from mobslim import ActivityParams, Controller, ModeParams, ScoringParams
from mobslim.config.matsim import MatsimConfig, scoring_params
from mobslim.replanning import TransitRouterFactory
from mobslim.replanning.context import default_transit_router
from mobslim.scoring import CharyparNagelScorer
from mobslim.transit import RaptorParams
from tests.test_transit_scenarios import load

H = 3600.0


def test_from_scoring_subtracts_performing():
    # MATSim's RaptorUtils / TransitRouterConfig: utility - performing
    cfg = RaptorParams.from_scoring(ScoringParams())
    assert (cfg.travel, cfg.waiting, cfg.walk, cfg.line_switch) == (-12.0, -12.0, -12.0, -1.0)
    params = ScoringParams(performing=4.0, marginal_utility_of_waiting_pt=-2.0,
                           utility_of_line_switch=-3.0, walk_mode="foot",
                           modes={"pt": ModeParams(marginal_utility_of_traveling=-1.0),
                                  "foot": ModeParams(marginal_utility_of_traveling=-8.0)})
    cfg = RaptorParams.from_scoring(params, max_transfers=3)
    assert (cfg.travel, cfg.waiting, cfg.walk, cfg.line_switch) == (-5.0, -6.0, -12.0, -3.0)
    assert cfg.max_transfers == 3


def test_factory_is_picklable_and_builds_routers_with_its_config():
    scenario = load("pt_simple")
    f = TransitRouterFactory.from_scoring(ScoringParams(utility_of_line_switch=-7.0))
    assert pickle.loads(pickle.dumps(f)) == f
    assert f(scenario.network, scenario.transit).params.line_switch == -7.0
    assert default_transit_router(scenario.network, scenario.transit).params == RaptorParams()


def test_controller_routes_pt_with_the_scorers_utilities():
    scenario = load("pt_simple")
    params = ScoringParams(performing=3.0, activities={"h": ActivityParams(12 * H)},
                           utility_of_line_switch=-2.0)
    c = Controller(scenario, scorer=CharyparNagelScorer(params))
    assert c.transit_router_factory.params == RaptorParams.from_scoring(params, "pt")
    # an explicit factory wins
    mine = TransitRouterFactory(RaptorParams(line_switch=-50.0))
    assert Controller(scenario, transit_router_factory=mine).transit_router_factory is mine

    # a scorer without ScoringParams: RaptorParams's defaults
    class Flat:
        def score(self, plan):
            return 0.0

    assert Controller(scenario, scorer=Flat()).transit_router_factory is default_transit_router


@pytest.mark.parametrize("case", ["pt-tutorial"])
def test_matsim_config_routes_pt_with_its_scoring(case):
    from verification.freeze import FIXTURES

    cfg = MatsimConfig.load(FIXTURES / case / "config.xml")
    params, _ = scoring_params(cfg)
    c = Controller.from_matsim_config(cfg)
    assert c.transit_router_factory.params == RaptorParams.from_scoring(params, "pt")
    assert c.transit_router_factory.params.travel == params.mode("pt").marginal_utility_of_traveling - 6.0


# ------------------------------------------------- person-specific weights

from dataclasses import replace  # noqa: E402

from mobslim import Population  # noqa: E402
from mobslim.replanning import (  # noqa: E402
    ProcessPoolRunner,
    ReplanningContext,
    Reroute,
    SerialRunner,
    StrategyManager,
    route_scenario,
)
from mobslim.replanning.selectors import KeepSelectedSelector  # noqa: E402
from mobslim.replanning.strategy import PlanStrategy  # noqa: E402
from mobslim.routing import TravelTimeTable  # noqa: E402
from mobslim.transit import (  # noqa: E402
    PersonRaptorWeights,
    RaptorRouter,
    RaptorWeights,
    RustRaptorRouter,
    rust_raptor_available,
)
from tests.test_transit_scenarios import random_queries  # noqa: E402

WALKER = RaptorWeights(walk=-0.5)  # hardly minds walking
TRANSFER_AVERSE = RaptorWeights(line_switch=-60.0)


def _routers(scenario):
    yield RaptorRouter
    if rust_raptor_available():
        yield RustRaptorRouter


@pytest.mark.parametrize("name", ["pt_lineswitch", "pt_tutorial"])
def test_per_query_weights_equal_a_router_with_those_weights(name):
    scenario = load(name)
    o, d, t = random_queries(scenario, 200, seed=3)
    o, d, t = [tuple(p) for p in o.tolist()], [tuple(p) for p in d.tolist()], t.tolist()
    sets = [None, WALKER, TRANSFER_AVERSE]
    weights = [sets[q % 3] for q in range(len(o))]
    results = []
    for cls in _routers(scenario):
        router = cls(scenario.network, scenario.transit)
        mixed = router.route_many_xy(o, d, t, weights=weights)
        for w in sets:
            cfg = RaptorParams() if w is None else replace(RaptorParams(), travel=w.travel,
                                                           waiting=w.waiting, walk=w.walk,
                                                           line_switch=w.line_switch)
            own = cls(scenario.network, scenario.transit, cfg).route_many_xy(o, d, t)
            assert [r for r, x in zip(mixed, weights) if x == w] == \
                [r for r, x in zip(own, weights) if x == w]
        results.append(mixed)
    assert all(r == results[0] for r in results)  # Rust == Python
    # the weights matter
    plain = RaptorRouter(scenario.network, scenario.transit).route_many_xy(o, d, t)
    assert any(a != b for a, b in zip(plain, results[0]))


def _pt_tutorial_split():
    """pt-tutorial with every second person a walker (weights) sharing one plan store."""
    scenario = load("pt_tutorial")
    n = len(scenario.population)
    weights = PersonRaptorWeights.from_sequence([WALKER if i % 2 else None for i in range(n)])
    return scenario, weights


def _rides(population):
    return [tuple(leg.route.rides for leg in p.selected_plan.legs if leg.mode == "pt")
            for p in population]


def test_route_scenario_routes_each_person_with_their_weights():
    scenario, weights = _pt_tutorial_split()
    # share each plan between a walker and a non-walker: plans must be split
    pop = scenario.population
    shared = Population([replace(p, plans=pop[i - i % 2].plans) for i, p in enumerate(pop)])
    scenario = scenario.with_population(shared)
    mixed = route_scenario(scenario, transit_weights=weights)
    plain = route_scenario(scenario)
    walkers = route_scenario(scenario, transit_router_factory=TransitRouterFactory(
        RaptorParams(walk=WALKER.walk)))
    got, a, b = _rides(mixed.population), _rides(plain.population), _rides(walkers.population)
    assert got == [b[i] if i % 2 else a[i] for i in range(len(got))]
    assert got != a
    assert len(mixed.population.plans) > len(plain.population.plans)  # split where they differ


@pytest.mark.parametrize("runner", ["serial", "serial_unbatched", "pool"])
def test_replanning_reroutes_each_person_with_their_weights(runner):
    scenario, weights = _pt_tutorial_split()
    plain = route_scenario(scenario)
    walkers = route_scenario(scenario, transit_router_factory=TransitRouterFactory(
        RaptorParams(walk=WALKER.walk)))
    expected = [b if i % 2 else a for i, (a, b) in
                enumerate(zip(_rides(plain.population), _rides(walkers.population)))]
    manager = StrategyManager([(1.0, PlanStrategy(KeepSelectedSelector(), [Reroute()]))])
    ctx = ReplanningContext.from_scenario(plain, TravelTimeTable.free_speed(plain.network), 1,
                                          transit_weights=weights)
    if runner == "pool":
        with ProcessPoolRunner(workers=2) as r:
            out = r.run(plain.population, manager, ctx, seed=1)
    else:
        out = SerialRunner(batch_routing=runner == "serial").run(plain.population, manager, ctx, 1)
    assert _rides(out) == expected


def test_controller_derives_weights_from_subpopulation_scoring():
    scenario = load("pt_lineswitch")
    n = len(scenario.population)
    scenario = scenario.with_population(scenario.population.with_attributes(
        {scenario.population.ids.id(0): {"subpopulation": "averse"}}))
    base = ScoringParams(activities={"h": ActivityParams(12 * H), "w": ActivityParams(8 * H)})
    averse = replace(base, utility_of_line_switch=-60.0)
    scorer = CharyparNagelScorer(base, subpopulations={"averse": averse, "same": base})
    c = Controller(scenario, scorer=scorer)
    w = c.person_transit_weights(scenario)
    assert w.of(0) == RaptorWeights.from_scoring(averse) and all(w.of(i) is None for i in range(1, n))
    # a function of the attributes overrides the scorer's
    c = Controller(scenario, scorer=scorer, transit_weights=lambda a: [WALKER] * a.n_persons)
    assert c.person_transit_weights(scenario).sets == (WALKER,)
    # nobody differs: no weights
    assert Controller(scenario, scorer=CharyparNagelScorer(base)).person_transit_weights(scenario) is None
    with pytest.raises(ValueError):
        Controller(scenario, transit_weights=lambda a: [WALKER] * (n + 1)).person_transit_weights(scenario)
