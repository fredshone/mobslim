from mobslim import ActivityType, Plan
from mobslim.entities.networks import RoadGrid
from mobslim.listener import EventListener
from mobslim.sim import Sim, SimLink


class TestSimLink:
    """Test SimLink class."""

    def test_simlink_init(self):
        """Test SimLink initialization."""
        attributes = {
            "length": 1000,
            "lanes": 2,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        assert sim_link.storage_capacity == 2000  # length * lanes
        assert sim_link.flow_capacity == 1  # int(1 / (0.5 * 2))
        assert sim_link.min_duration == 100  # length / freespeed
        assert sim_link.queue == []
        assert sim_link.earliest_next_exit == 0

    def test_can_enter(self):
        """Test checking if vehicle can enter link."""
        attributes = {
            "length": 100,
            "lanes": 1,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        # Empty link should allow entry
        assert sim_link.can_enter(size=4, time=0)

        # Add vehicle to fill capacity
        sim_link.enter("agent1", 4, 0)
        # Should still allow if under capacity
        assert sim_link.can_enter(size=4, time=0)

    def test_enter(self):
        """Test entering a link."""
        attributes = {
            "length": 100,
            "lanes": 1,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        sim_link.enter("agent1", 4, 0)

        assert len(sim_link.queue) == 1
        assert sim_link.queue[0][0] == "agent1"
        assert sim_link.queue[0][1] == 4
        assert sim_link.queue[0][2] == 10  # time + min_duration

    def test_can_exit(self):
        """Test checking if vehicle can exit link."""
        attributes = {
            "length": 100,
            "lanes": 1,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        sim_link.enter("agent1", 4, 0)

        # Cannot exit before min_duration
        assert not sim_link.can_exit(time=5)
        # Can exit after min_duration
        assert sim_link.can_exit(time=10)

    def test_exit(self):
        """Test exiting a link."""
        attributes = {
            "length": 100,
            "lanes": 1,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        sim_link.enter("agent1", 4, 0)
        result = sim_link.exit("agent1", 10)

        assert len(sim_link.queue) == 0
        assert result[0] == "agent1"
        assert sim_link.earliest_next_exit == 12  # 10 + flow_capacity

    def test_has_storage_capacity(self):
        """Test checking storage capacity."""
        attributes = {
            "length": 100,
            "lanes": 1,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        # Should have capacity when empty
        assert sim_link.has_storage_capacity(4)

        # Fill to capacity
        sim_link.enter("agent1", 50, 0)
        assert sim_link.has_storage_capacity(50)
        assert not sim_link.has_storage_capacity(51)

    def test_has_flow_capacity(self):
        """Test checking flow capacity."""
        attributes = {
            "length": 100,
            "lanes": 1,
            "freespeed": 10,
            "flow_capacity": 0.5,
        }
        sim_link = SimLink(attributes)

        # Initially should have flow capacity
        assert sim_link.has_flow_capacity(0)

        # After exit, should respect flow capacity
        sim_link.enter("agent1", 4, 0)
        sim_link.exit("agent1", 10)
        assert not sim_link.has_flow_capacity(11)
        assert sim_link.has_flow_capacity(12)


class TestSim:
    """Test Sim class."""

    def test_sim_init(self):
        """Test Sim initialization."""
        network = RoadGrid(size=2)
        listener = EventListener()

        sim = Sim(networks=network, listener=listener)

        assert sim.networks == network
        assert sim.event_listener == listener

    def test_set_plans(self):
        """Test setting plans for simulation."""
        network = RoadGrid(size=2, length=50, freespeed=10)
        listener = EventListener()
        sim = Sim(networks=network, listener=listener)

        # Create simple plan
        plans = {}
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location=(0, 0), duration=100)
        plan.add_trip(origin=(0, 0), destination=(1, 1), network_mode="road")
        plan.add_activity(ActivityType.WORK, location=(1, 1), duration=100)
        plan.finish()

        # Set route for trip
        for component in plan.components:
            if hasattr(component, "route"):
                component.route = [
                    (((0, 0), (0, 1)), 5, 5),
                    (((0, 1), (1, 1)), 5, 5),
                ]

        plans["agent1"] = plan

        sim.set(plans=plans)

        assert sim.time == 0
        assert len(sim.queue) > 0
        assert len(sim.sim_links) > 0

    def test_run_simulation(self):
        """Test running a simulation."""
        network = RoadGrid(size=2, length=50, freespeed=10)
        listener = EventListener()
        sim = Sim(networks=network, listener=listener)

        # Create simple plan
        plans = {}
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location=(0, 0), duration=10)
        plan.add_trip(origin=(0, 0), destination=(0, 1), network_mode="road")
        plan.add_activity(ActivityType.WORK, location=(0, 1), duration=10)
        plan.finish()

        # Set route for trip
        for component in plan.components:
            if hasattr(component, "route"):
                component.route = [(((0, 0), (0, 1)), 5, 5)]

        plans["agent1"] = plan

        sim.set(plans=plans)
        events = sim.run(steps=100)

        assert len(events) > 0
        # Should have recorded events
        assert listener.log == events

    def test_can_exit(self):
        """Test can_exit method."""
        from mobslim import InstructionType

        network = RoadGrid(size=2, length=50, freespeed=10)
        listener = EventListener()
        sim = Sim(networks=network, listener=listener)

        # Create plans to initialize sim_links
        plans = {}
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location=(0, 0), duration=10)
        plan.finish()
        plans["agent1"] = plan
        sim.set(plans=plans)

        # Test ExitLink instruction
        instruction = (InstructionType.ExitLink, "road", ((0, 0), (0, 1)), 0)

        # Enter agent into link first
        sim.sim_links[("road", (0, 0), (0, 1))].enter("agent1", 4, 0)

        # Should not be able to exit immediately
        assert not sim.can_exit("agent1", instruction)

    def test_can_enter(self):
        """Test can_enter method."""
        from mobslim import InstructionType

        network = RoadGrid(size=2, length=50, freespeed=10)
        listener = EventListener()
        sim = Sim(networks=network, listener=listener)

        # Create plans to initialize sim_links
        plans = {}
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location=(0, 0), duration=10)
        plan.finish()
        plans["agent1"] = plan
        sim.set(plans=plans)

        # Test EnterLink instruction
        instruction = (InstructionType.EnterLink, "road", ((0, 0), (0, 1)), 0)

        # Should be able to enter empty link
        assert sim.can_enter("agent1", instruction)


class TestEventListener:
    """Test EventListener class."""

    def test_event_listener_init(self):
        """Test EventListener initialization."""
        listener = EventListener()

        assert listener.log == []

    def test_add_event(self):
        """Test adding an event."""
        from mobslim import InstructionType

        listener = EventListener()
        instruction = (InstructionType.SOS, None, None, 0)

        listener.add(time=0, a="agent1", b=instruction)

        assert len(listener.log) == 1
        assert listener.log[0][0] == 0
        assert listener.log[0][1] == "agent1"
        assert listener.log[0][2] == instruction

    def test_reset(self):
        """Test resetting event listener."""
        from mobslim import InstructionType

        listener = EventListener()
        instruction = (InstructionType.SOS, None, None, 0)
        listener.add(time=0, a="agent1", b=instruction)

        listener.reset()

        assert listener.log == []
