"""``Controller.from_matsim_config`` on the frozen MATSim reference configs."""

import xml.etree.ElementTree as ET

import pytest

from mobslim import Controller, ControllerParams, HermesSim, QSim
from mobslim.config import MatsimConfig, UnsupportedConfigError
from mobslim.config.matsim import controller_kwargs
from tests.conftest import FIXTURES

REF = FIXTURES / "matsim_reference"
EQUIL = REF / "equil-100" / "config.xml"
PT = REF / "pt-tutorial" / "config.xml"


def test_equil_translates_fully():
    kwargs, unsupported = controller_kwargs(MatsimConfig.load(EQUIL))
    assert unsupported == []
    assert isinstance(kwargs["mobsim"], QSim)
    assert len(kwargs["scenario"].population) == 100
    assert kwargs["scenario"].transit is None
    c = kwargs["params"]
    assert (c.iterations, c.seed, c.output_dir) == (1, 4711, "./output")


def test_pt_tutorial_reads_transit_and_runs():
    controller = Controller.from_matsim_config(PT, strict=True, params=ControllerParams(iterations=2))
    assert controller.scenario.transit is not None
    assert controller.scenario.transit.params.passenger_modes == ("pt",)
    assert controller.strategies.max_plans == 5
    result = controller.run()
    assert len(result.metrics) == 2


def test_controller_params(tmp_path):
    cfg = (MatsimConfig.load(EQUIL)
           .set("controller", "lastIteration", 4)
           .set("global", "randomSeed", 42)
           .set("scoring", "learningRate", 0.5)
           .set("travelTimeCalculator", "traveltimeBinSize", "00:05:00"))
    c = Controller.from_matsim_config(cfg.write(tmp_path / "config.xml"), strict=True).params
    assert (c.iterations, c.seed, c.learning_rate, c.travel_time_bin_size) == (5, 42, 0.5, 300.0)


def test_time_allocation_mutator_settings():
    """mutateAroundInitialEndTimeOnly and latestActivityEndTime (MATSim's
    default 24:00) reach the mutator; mutationRangeStep is not honoured."""
    from mobslim.config.matsim import strategy_manager
    from mobslim.replanning import TimeAllocationMutator

    def mutator(cfg):
        manager, unsupported = strategy_manager(cfg)
        tams = [m for _, s in manager.strategies for m in s.modules
                if isinstance(m, TimeAllocationMutator)]
        return tams, unsupported

    cfg = MatsimConfig.load(PT)
    (tam,), unsupported = mutator(cfg)
    assert (tam.around_initial_end_time, tam.latest_end_time) == (False, 86400.0)
    assert unsupported == []
    cfg.set("timeAllocationMutator", "mutateAroundInitialEndTimeOnly", "true")
    cfg.set("timeAllocationMutator", "latestActivityEndTime", "30:00:00")
    cfg.set("timeAllocationMutator", "mutationRangeStep", "60")
    (tam,), unsupported = mutator(cfg)
    assert (tam.around_initial_end_time, tam.latest_end_time) == (True, 30 * 3600.0)
    assert unsupported == ["timeAllocationMutator.mutationRangeStep=60"]


def test_hermes_mobsim():
    cfg = MatsimConfig.load(EQUIL).set("controller", "mobsim", "hermes").set("hermes", "flowCapacityFactor", 0.5)
    sim = Controller.from_matsim_config(cfg, strict=True).mobsim
    assert isinstance(sim, HermesSim)
    assert sim.params.flow_capacity_factor == 0.5


def test_teleported_modes():
    cfg = MatsimConfig.load(EQUIL)
    routing = cfg._module("routing", create=True)
    ps = ET.SubElement(routing, "parameterset", type="teleportedModeParameters")
    ET.SubElement(ps, "param", name="mode", value="walk")
    ET.SubElement(ps, "param", name="teleportedModeSpeed", value="1.0")
    ET.SubElement(ps, "param", name="beelineDistanceFactor", value="1.5")
    modes = Controller.from_matsim_config(cfg, strict=True).scenario.teleported_modes
    assert (modes["walk"].speed, modes["walk"].beeline_distance_factor) == (1.0, 1.5)
    assert "bike" in modes  # defaults kept


def test_main_modes_and_vehicle_types(tmp_path):
    """qsim.mainMode: the network modes simulated; modeVehicleTypesFromVehiclesData:
    each mode's vehicle type's PCE."""
    vehicles = tmp_path / "vehicleTypes.xml"
    vehicles.write_text(
        '<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd">'
        '<vehicleType id="car"><passengerCarEquivalents pce="1.0"/></vehicleType>'
        '<vehicleType id="truck"><passengerCarEquivalents pce="3.5"/></vehicleType>'
        '<vehicleType id="bike"><passengerCarEquivalents pce="0.2"/></vehicleType>'
        '</vehicleDefinitions>')
    cfg = MatsimConfig.load(EQUIL)
    cfg.set("qsim", "mainMode", "car,truck").set("qsim", "vehiclesSource",
                                                 "modeVehicleTypesFromVehiclesData")
    cfg.set("vehicles", "vehiclesFile", str(vehicles))
    sc = Controller.from_matsim_config(cfg, strict=True).scenario
    assert sc.simulated_modes == {"car", "truck"}
    assert sc.mode_pce == {"car": 1.0, "truck": 3.5}  # bike: not simulated
    plain = Controller.from_matsim_config(EQUIL, strict=True).scenario
    assert plain.simulated_modes == {"car"} and plain.mode_pce == {}  # MATSim's defaults


def test_persons_own_vehicle_types(tmp_path):
    """A ``vehicleTypes`` person attribute (Open Berlin's commercial traffic):
    the person drives that type, of its PCE, in the mobsim too; its maximum
    velocity is reported as not simulated."""
    import dataclasses

    import numpy as np

    from mobslim.config.matsim import person_vehicle_pce
    from mobslim.mobsim._common import person_pce

    vehicles = tmp_path / "vehicleTypes.xml"
    vehicles.write_text(
        '<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd">'
        '<vehicleType id="car"><passengerCarEquivalents pce="1.0"/></vehicleType>'
        '<vehicleType id="van"><maximumVelocity meterPerSecond="22.22"/>'
        '<passengerCarEquivalents pce="2.0"/></vehicleType>'
        '</vehicleDefinitions>')
    cfg = MatsimConfig.load(EQUIL)
    cfg.set("qsim", "vehiclesSource", "modeVehicleTypesFromVehiclesData")
    cfg.set("vehicles", "vehiclesFile", str(vehicles))
    sc = Controller.from_matsim_config(cfg).scenario
    ids = list(sc.population.ids)
    pop = sc.population.with_attributes({ids[1]: {"vehicleTypes": '{"car":"van"}'},
                                         ids[2]: {"vehicleTypes": '{"car":"bus"}'}})
    own, unsupported = person_vehicle_pce(cfg, pop)
    assert own["car"][1] == 2.0 and np.isnan(own["car"][[0, 2]]).all()
    assert unsupported == ["vehicleTypes: unknown vehicle type bus",
                           "vehicleTypes: maximum velocity of vehicle type van (not simulated on links)"]
    sim = dataclasses.replace(sc.with_population(pop), person_pce=own)
    pce = person_pce(sim, pop.selected_batch())
    assert pce[1] == 2.0 and pce[0] == 1.0 and pce[2] == 1.0


def test_unsupported_warns_or_raises(caplog):
    cfg = MatsimConfig.load(EQUIL).set("qsim", "trafficDynamics", "kinematicWaves")
    with pytest.raises(UnsupportedConfigError) as err:
        Controller.from_matsim_config(cfg, strict=True)
    assert err.value.unsupported == ["qsim.trafficDynamics=kinematicWaves"]
    with caplog.at_level("WARNING", logger="mobslim"):
        Controller.from_matsim_config(cfg)
    assert "qsim.trafficDynamics=kinematicWaves" in caplog.text


def test_overrides_replace_translated_arguments():
    sim = QSim()
    controller = Controller.from_matsim_config(EQUIL, mobsim=sim)
    assert controller.mobsim is sim
