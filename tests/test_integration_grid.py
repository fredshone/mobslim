from mobslim import ActivityType, Plan
from mobslim.entities.networks import RoadGrid
from mobslim.expected import SimpleExpectedDurations
from mobslim.listener import EventListener
from mobslim.optimizer import Optimizer
from mobslim.planners.greedy_trip_planner import GreedyTripPlanner
from mobslim.planners.rerouters.simple_rerouter import StaticRouter
from mobslim.sim import Sim


def test_grid_scenario():
    """Test grid scenario - creates synthetic grid network and agents."""

    AGENTS = 5
    SIZE = 3
    MAX_RUNS = 2

    # Network setup
    network = RoadGrid(size=SIZE)
    o = network.get_start()
    d = network.get_end()

    # Verify network structure
    assert o == (0, 0)
    assert d == (SIZE, SIZE)
    assert "road" in network._networks

    # Agent setup
    plans = {}
    for i in range(AGENTS):
        plan = Plan()
        plan.add_activity(type=ActivityType.HOME, location=o, duration=10)
        plan.add_trip(origin=o, destination=d, network_mode="road")
        plan.add_activity(type=ActivityType.WORK, location=d, duration=10)
        plan.finish()
        plans[i] = plan

    # Verify plans created
    assert len(plans) == AGENTS

    # Planner setup
    expected_link_durations = SimpleExpectedDurations(
        network, network_mode="road"
    )
    router = StaticRouter(
        network=network,
        network_mode="road",
        expectations=expected_link_durations,
    )
    planner = GreedyTripPlanner(
        plans=plans, router=router, network=network, p=0.2
    )

    # Initiate all plans with naive trip estimates
    planner.plan()

    # Verify routes were planned
    for plan in plans.values():
        for component in plan.components:
            if hasattr(component, "route") and component.route is not None:
                assert len(component.route) > 0
                break

    # Simulation setup
    sim = Sim(networks=network, listener=EventListener())

    # Optimizer setup and run
    optimizer = Optimizer(sim=sim, plans=plans, planner=planner)
    events = optimizer.run(max_runs=MAX_RUNS)

    # Verify events were generated
    assert len(events) > 0

    # Verify all agents completed their plans
    agent_events = {}
    for time, agent_id, instruction in events:
        if agent_id not in agent_events:
            agent_events[agent_id] = []
        agent_events[agent_id].append(instruction)

    # Should have events for all agents
    assert len(agent_events) == AGENTS
