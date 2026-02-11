from mobslim import ActivityType, Plan
from mobslim.entities.networks import RoadGrid
from mobslim.expected import SimpleExpectedDurations
from mobslim.listener import EventListener
from mobslim.optimizer import Optimizer
from mobslim.planners.greedy_trip_planner import GreedyTripPlanner
from mobslim.planners.rerouters.simple_rerouter import StaticRouter
from mobslim.sim import Sim


class TestOptimizer:
    """Test Optimizer class."""

    def test_optimizer_init(self):
        """Test Optimizer initialization."""
        network = RoadGrid(size=2)
        sim = Sim(networks=network, listener=EventListener())

        plans = {}
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.2
        )

        optimizer = Optimizer(
            sim=sim, plans=plans, planner=planner, network_modes=["road"]
        )

        assert optimizer.sim == sim
        assert optimizer.plans == plans
        assert optimizer.planner == planner
        assert optimizer.network_modes == ["road"]

    def test_optimizer_run_single_iteration(self):
        """Test optimizer running single iteration."""
        network = RoadGrid(size=2, length=50, freespeed=10)
        sim = Sim(networks=network, listener=EventListener())

        # Create simple plan
        plan = Plan()
        plan.add_activity(type=ActivityType.HOME, location=(0, 0), duration=10)
        plan.add_trip(origin=(0, 0), destination=(2, 2), network_mode="road")
        plan.add_activity(type=ActivityType.WORK, location=(2, 2), duration=10)
        plan.finish()

        plans = {0: plan}

        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.2
        )

        optimizer = Optimizer(
            sim=sim, plans=plans, planner=planner, network_modes=["road"]
        )

        # Run single iteration
        events = optimizer.run(max_runs=1)

        # Should have events
        assert len(events) > 0

    def test_optimizer_run_multiple_iterations(self):
        """Test optimizer running multiple iterations."""
        network = RoadGrid(size=2, length=50, freespeed=10)
        sim = Sim(networks=network, listener=EventListener())

        # Create multiple agents
        plans = {}
        for i in range(3):
            plan = Plan()
            plan.add_activity(
                type=ActivityType.HOME, location=(0, 0), duration=10
            )
            plan.add_trip(
                origin=(0, 0), destination=(2, 2), network_mode="road"
            )
            plan.add_activity(
                type=ActivityType.WORK, location=(2, 2), duration=10
            )
            plan.finish()
            plans[i] = plan

        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.2
        )

        optimizer = Optimizer(
            sim=sim, plans=plans, planner=planner, network_modes=["road"]
        )

        # Run multiple iterations
        events = optimizer.run(max_runs=3)

        # Should have events from final iteration
        assert len(events) > 0

    def test_optimizer_with_multiple_modes(self):
        """Test optimizer with multiple network modes."""
        network = RoadGrid(size=2)
        sim = Sim(networks=network, listener=EventListener())

        plans = {}
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.2
        )

        optimizer = Optimizer(
            sim=sim,
            plans=plans,
            planner=planner,
            network_modes=["road", "bike"],  # Multiple modes
        )

        assert "road" in optimizer.network_modes
        assert "bike" in optimizer.network_modes


class TestOptimizerReporting:
    """Test Optimizer reporting functionality."""

    def test_report_generates_output(self, capsys):
        """Test that report method generates output."""
        network = RoadGrid(size=2, length=50, freespeed=10)
        sim = Sim(networks=network, listener=EventListener())

        # Create simple plan
        plan = Plan()
        plan.add_activity(type=ActivityType.HOME, location=(0, 0), duration=10)
        plan.add_trip(origin=(0, 0), destination=(1, 1), network_mode="road")
        plan.add_activity(type=ActivityType.WORK, location=(1, 1), duration=10)
        plan.finish()

        plans = {0: plan}

        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=1.0
        )
        planner.plan()

        optimizer = Optimizer(
            sim=sim, plans=plans, planner=planner, network_modes=["road"]
        )

        # Set and run sim to get events
        sim.set(plans=plans)
        events = sim.run()

        # Call report
        optimizer.report(0, events)

        # Check output was generated
        captured = capsys.readouterr()
        assert "Av. road trip duration" in captured.out
        assert "Av. road trip length" in captured.out
        assert "Av. road link duration" in captured.out
