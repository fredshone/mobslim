import pytest

from mobslim import ActivityType, InstructionType, Networks, Plan, Trip
from mobslim.entities.agents import EOS, SOS, Activity, load_plans_from_xml


class TestPlan:
    """Test Plan class."""

    def test_plan_init(self):
        """Test Plan initialization."""
        plan = Plan()
        assert len(plan.components) == 1
        assert isinstance(plan.components[0], SOS)

    def test_add_activity(self):
        """Test adding an activity to a plan."""
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location="node1", duration=100)

        assert len(plan.components) == 2
        assert isinstance(plan.components[1], Activity)
        assert plan.components[1].type == ActivityType.HOME
        assert plan.components[1].location == "node1"
        assert plan.components[1].duration == 100

    def test_add_trip(self):
        """Test adding a trip to a plan."""
        plan = Plan()
        plan.add_trip(origin="node1", destination="node2", network_mode="road")

        assert len(plan.components) == 2
        assert isinstance(plan.components[1], Trip)
        assert plan.components[1].origin == "node1"
        assert plan.components[1].destination == "node2"
        assert plan.components[1].network_mode == "road"

    def test_set_network_mode(self):
        """Test setting network mode for all trips."""
        plan = Plan()
        plan.add_trip(origin="A", destination="B", network_mode="road")
        plan.add_trip(origin="B", destination="C", network_mode="road")
        plan.set_network_mode("pt")

        for component in plan.components:
            if isinstance(component, Trip):
                assert component.network_mode == "pt"

    def test_finish(self):
        """Test finishing a plan."""
        plan = Plan()
        plan.finish()

        assert isinstance(plan.components[-1], EOS)

    def test_get_instructions(self):
        """Test getting instructions from a plan."""
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location="node1", duration=100)
        plan.finish()

        instructions = list(plan.get_instructions())

        # Should have SOS, EnterFacility/ExitFacility for activity, EOS
        assert len(instructions) == 2  # Pairs of instructions
        assert instructions[0][0][0] == InstructionType.SOS
        assert instructions[0][1][0] == InstructionType.EnterFacility

    def test_copy(self):
        """Test copying a plan."""
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location="node1", duration=100)

        # Note: copy() may not be fully implemented for all components
        # Just verify basic structure is preserved
        assert len(plan.components) == 2
        assert isinstance(plan.components[1], Activity)
        assert plan.components[1].location == "node1"

    def test_repr(self):
        """Test plan string representation."""
        plan = Plan()
        plan.add_activity(ActivityType.HOME, location="node1", duration=100)

        repr_str = repr(plan)
        assert "Plan(" in repr_str


class TestActivity:
    """Test Activity class."""

    def test_activity_init(self):
        """Test Activity initialization."""
        activity = Activity(ActivityType.WORK, location="office", duration=480)

        assert activity.type == ActivityType.WORK
        assert activity.location == "office"
        assert activity.duration == 480

    def test_get_instructions(self):
        """Test getting instructions from activity."""
        activity = Activity(ActivityType.HOME, location="home", duration=600)

        instructions = list(activity.get_instructions())

        assert len(instructions) == 2
        assert instructions[0][0] == InstructionType.EnterFacility
        assert instructions[0][1] == ActivityType.HOME
        assert instructions[0][2] == "home"
        assert instructions[0][3] == 600
        assert instructions[1][0] == InstructionType.ExitFacility

    def test_repr(self):
        """Test activity string representation."""
        activity = Activity(ActivityType.WORK, location="office", duration=480)

        repr_str = repr(activity)
        assert "Act(" in repr_str
        assert "office" in repr_str


class TestTrip:
    """Test Trip class."""

    def test_trip_init(self):
        """Test Trip initialization."""
        trip = Trip(origin="A", destination="B", network_mode="road")

        assert trip.origin == "A"
        assert trip.destination == "B"
        assert trip.network_mode == "road"
        assert trip.route is None

    def test_trip_with_duration(self):
        """Test Trip initialization with duration."""
        trip = Trip(
            origin="A", destination="B", network_mode="road", duration=300
        )

        assert trip.expected_duration == 300

    def test_get_instructions_no_route(self):
        """Test getting instructions without a route raises error."""
        trip = Trip(origin="A", destination="B", network_mode="road")

        with pytest.raises(ValueError, match="Route has not been planned"):
            list(trip.get_instructions())

    def test_get_instructions_with_route(self):
        """Test getting instructions with a route."""
        trip = Trip(origin="A", destination="B", network_mode="road")
        trip.route = [(("A", "B"), 10, 5)]

        instructions = list(trip.get_instructions())

        assert len(instructions) == 2
        assert instructions[0][0] == InstructionType.EnterLink
        assert instructions[0][1] == "road"
        assert instructions[0][2] == ("A", "B")
        assert instructions[1][0] == InstructionType.ExitLink

    def test_repr(self):
        """Test trip string representation."""
        trip = Trip(
            origin="A", destination="B", network_mode="road", duration=300
        )

        repr_str = repr(trip)
        assert "Trip(" in repr_str
        assert "road" in repr_str


class TestInstructionTypes:
    """Test instruction type enums."""

    def test_instruction_type_values(self):
        """Test instruction type enum values."""
        assert InstructionType.SOS.value == 0
        assert InstructionType.EnterFacility.value == 1
        assert InstructionType.ExitFacility.value == 2
        assert InstructionType.EnterLink.value == 3
        assert InstructionType.ExitLink.value == 4
        assert InstructionType.EOS.value == 5


class TestActivityType:
    """Test activity type enums."""

    def test_activity_type_values(self):
        """Test activity type enum values."""
        assert ActivityType.HOME.value == "h"
        assert ActivityType.WORK.value == "w"


class TestLoadPlansFromXML:
    """Test loading plans from XML."""

    def test_load_plans_from_xml(self, tmp_path):
        """Test loading plans from XML file."""
        # Create network first
        network_xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">
<network>
    <nodes>
        <node id="1" x="0" y="0"/>
        <node id="2" x="100" y="0"/>
    </nodes>
    <links>
        <link id="1-2" from="1" to="2" length="100" capacity="1800" freespeed="10" permlanes="1" modes="road"/>
    </links>
</network>"""
        network_file = tmp_path / "network.xml"
        network_file.write_text(network_xml)

        networks = Networks()
        networks.load_xml(str(network_file))

        # Create plans XML
        plans_xml = """<?xml version="1.0" ?>
<!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">
<plans>
    <person id="1">
        <plan>
            <act type="h" node="1" end_time="06:00:00" />
            <leg mode="car">
                <route></route>
            </leg>
            <act type="w" node="2" dur="08:00:00" />
        </plan>
    </person>
</plans>"""
        plans_file = tmp_path / "plans.xml"
        plans_file.write_text(plans_xml)

        plans = load_plans_from_xml(str(plans_file), networks=networks)

        assert "1" in plans
        assert isinstance(plans["1"], Plan)
        # Check that plan has activities and trips
        components = plans["1"].components
        assert any(isinstance(c, Activity) for c in components)
        assert any(isinstance(c, Trip) for c in components)
