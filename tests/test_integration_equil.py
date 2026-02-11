from pathlib import Path

from mobslim.entities.agents import load_plans_from_xml
from mobslim.entities.networks import Networks
from mobslim.expected import SimpleExpectedDurations
from mobslim.listener import EventListener
from mobslim.optimizer import Optimizer
from mobslim.planners.greedy_trip_planner import GreedyTripPlanner
from mobslim.planners.rerouters.simple_rerouter import StaticRouter
from mobslim.sim import Sim


def test_equil_scenario():
    """Test equil scenario - loads network and plans from XML, runs optimization."""

    # Find scenarios directory
    scenarios_dir = Path(__file__).parent / "fixtures" / "equil"

    network_file = scenarios_dir / "network.xml"
    plans_file = scenarios_dir / "plans100.xml"

    # Network setup
    network = Networks()
    network.load_xml(str(network_file))

    # Verify network loaded
    assert len(network._networks) > 0
    assert len(network.node_locations) > 0

    # Agent setup
    plans = load_plans_from_xml(str(plans_file), networks=network)

    # Verify plans loaded
    assert len(plans) > 0

    # Planner setup
    network_mode = list(network._networks.keys())[0]
    expected_link_durations = SimpleExpectedDurations(
        network, network_mode=network_mode
    )
    router = StaticRouter(
        network=network,
        network_mode=network_mode,
        expectations=expected_link_durations,
    )
    planner = GreedyTripPlanner(
        plans=plans, router=router, network=network, p=0.2
    )

    # Initiate all plans with naive trip estimates
    planner.plan()

    # Verify plans have routes
    for plan in plans.values():
        for component in plan.components:
            if hasattr(component, "route") and component.route is not None:
                assert len(component.route) > 0
                break

    # Simulation setup
    sim = Sim(networks=network, listener=EventListener())

    # Optimizer setup and run (just 2 iterations for testing)
    optimizer = Optimizer(
        sim=sim, plans=plans, planner=planner, network_modes=[network_mode]
    )
    events = optimizer.run(max_runs=2)

    # Verify events were generated
    assert len(events) > 0

    # Verify events have expected structure
    for event in events[:10]:  # Check first 10 events
        assert len(event) == 3  # (time, agent_id, instruction)
        time, agent_id, instruction = event
        assert isinstance(time, int)
        assert len(instruction) == 4  # (type, info, asset, duration)
