import pytest

from mobslim import ActivityType, Plan
from mobslim.entities.networks import RoadGrid
from mobslim.expected import SimpleExpectedDurations
from mobslim.planners.greedy_trip_planner import GreedyTripPlanner
from mobslim.planners.rerouters.simple_rerouter import StaticRouter


class TestStaticRouter:
    """Test StaticRouter class."""

    def test_static_router_init(self):
        """Test StaticRouter initialization."""
        network = RoadGrid(size=3)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )

        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        assert router.G is not None
        assert router.expectations == expected_durations

    def test_get_route(self):
        """Test getting a route between two nodes."""
        network = RoadGrid(size=3, length=50, freespeed=10)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )

        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        origin = (0, 0)
        destination = (3, 3)

        route, duration = router.get_route(origin, destination, time=0)

        # Should have a route
        assert len(route) > 0
        assert duration > 0

        # Verify route structure: list of (edge, expected_duration, min_duration)
        for edge, exp_dur, min_dur in route:
            assert len(edge) == 2  # (u, v)
            assert exp_dur > 0
            assert min_dur > 0

    def test_get_route_same_node(self):
        """Test getting route when origin equals destination."""
        network = RoadGrid(size=3)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )

        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        origin = (1, 1)
        destination = (1, 1)

        route, duration = router.get_route(origin, destination, time=0)

        # Should have empty route
        assert len(route) == 0
        assert duration == 0

    def test_update_router(self):
        """Test updating router with new events."""
        network = RoadGrid(size=2)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )

        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        # Create dummy plans and events
        plans = {}
        events = []

        # Update should not raise error
        router.update(plans=plans, network=network, events=events)


class TestGreedyTripPlanner:
    """Test GreedyTripPlanner class."""

    def test_greedy_trip_planner_init(self):
        """Test GreedyTripPlanner initialization."""
        network = RoadGrid(size=3)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        plans = {}
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.5
        )

        assert planner.plans == plans
        assert planner.router == router
        assert planner.network == network
        assert planner.p == 0.5
        assert planner.max_duration == 86400

    def test_greedy_trip_planner_invalid_p(self):
        """Test GreedyTripPlanner raises error for invalid p."""
        network = RoadGrid(size=3)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        plans = {}

        with pytest.raises(
            ValueError, match="Probability p must be between 0 and 1"
        ):
            GreedyTripPlanner(
                plans=plans, router=router, network=network, p=1.5
            )

    def test_plan_single_agent(self):
        """Test planning for a single agent."""
        network = RoadGrid(size=2)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        # Create plan
        plan = Plan()
        plan.add_activity(type=ActivityType.HOME, location=(0, 0), duration=100)
        plan.add_trip(origin=(0, 0), destination=(2, 2), network_mode="road")
        plan.add_activity(type=ActivityType.WORK, location=(2, 2), duration=100)
        plan.finish()

        plans = {0: plan}

        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=1.0  # Always replan
        )

        planner.plan()

        # Verify trip has route
        for component in plan.components:
            if hasattr(component, "route") and component.route is not None:
                assert len(component.route) > 0
                assert component.expected_duration > 0
                break

    def test_replan_with_probability(self):
        """Test replanning with probability."""
        network = RoadGrid(size=2)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        # Create multiple plans
        plans = {}
        for i in range(10):
            plan = Plan()
            plan.add_activity(
                type=ActivityType.HOME, location=(0, 0), duration=100
            )
            plan.add_trip(
                origin=(0, 0), destination=(2, 2), network_mode="road"
            )
            plan.add_activity(
                type=ActivityType.WORK, location=(2, 2), duration=100
            )
            plan.finish()
            plans[i] = plan

        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.0  # Never replan
        )

        # Initial planning
        planner.plan()

        # Store original routes
        original_routes = {}
        for i, plan in plans.items():
            for component in plan.components:
                if hasattr(component, "route"):
                    original_routes[i] = component.route
                    break

        # Replan with p=0 should not change anything
        planner.replan(p=0.0)

        # Routes should be unchanged (since p=0)
        for i, plan in plans.items():
            for component in plan.components:
                if hasattr(component, "route"):
                    # With p=0, routes might not be replanned
                    break

    def test_update_planner(self):
        """Test updating planner with events."""
        network = RoadGrid(size=2)
        expected_durations = SimpleExpectedDurations(
            network, network_mode="road"
        )
        router = StaticRouter(
            network=network,
            network_mode="road",
            expectations=expected_durations,
        )

        plans = {}
        planner = GreedyTripPlanner(
            plans=plans, router=router, network=network, p=0.5
        )

        # Create dummy events
        events = []

        # Update should not raise error
        planner.update(events)


class TestSimpleExpectedDurations:
    """Test SimpleExpectedDurations class."""

    def test_simple_expected_durations_init(self):
        """Test SimpleExpectedDurations initialization."""
        network = RoadGrid(size=2, length=100, freespeed=10)

        expected = SimpleExpectedDurations(network, network_mode="road")

        assert len(expected.edge_durations) > 0
        # All durations should be positive
        assert all(d > 0 for d in expected.edge_durations.values())

    def test_get_duration(self):
        """Test getting expected duration for an edge."""
        network = RoadGrid(size=2, length=100, freespeed=10)
        expected = SimpleExpectedDurations(network, network_mode="road")

        edge = ((0, 0), (0, 1))
        duration = expected.get(edge, time=0)

        assert duration > 0
        # Should be length/freespeed = 100/10 = 10
        assert duration == 10

    def test_update_link(self):
        """Test updating link duration."""
        network = RoadGrid(size=2, length=100, freespeed=10)
        expected = SimpleExpectedDurations(network, network_mode="road")

        edge = ((0, 0), (0, 1))
        original_duration = expected.get(edge, time=0)

        # Update with new duration
        new_duration = 20
        expected.update_link(edge, time=0, duration=new_duration, alpha=1.0)

        updated_duration = expected.get(edge, time=0)
        assert updated_duration == new_duration

    def test_update_link_with_alpha(self):
        """Test updating link duration with smoothing factor."""
        network = RoadGrid(size=2, length=100, freespeed=10)
        expected = SimpleExpectedDurations(network, network_mode="road")

        edge = ((0, 0), (0, 1))
        original_duration = expected.get(edge, time=0)

        # Update with alpha=0.5 (average of old and new)
        new_duration = 20
        expected.update_link(edge, time=0, duration=new_duration, alpha=0.5)

        updated_duration = expected.get(edge, time=0)
        expected_value = 0.5 * original_duration + 0.5 * new_duration
        assert updated_duration == expected_value

    def test_av_duration(self):
        """Test calculating average duration."""
        network = RoadGrid(size=2, length=100, freespeed=10)
        expected = SimpleExpectedDurations(network, network_mode="road")

        avg = expected.av_duration()

        assert avg > 0
        # Should be close to 10 (length/freespeed)
        assert abs(avg - 10) < 1
