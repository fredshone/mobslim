from mobslim.sim import Sim
from mobslim.planners.greedy_trip_planner import TripPlanner, StaticRouter
from mobslim.network import Network
from mobslim.agents import load_xml
from mobslim.expected import SimpleExpectedDurations
from mobslim.listener import EventListener
from mobslim.optimizer import Optimizer

AGENTS = 20
SIZE = 5  # Size of the grid network
MAX_RUNS = 20  # Maximum number of runs for the simulation

# network setup
network = Network()
network.load_xml("scenarios/equil/network.xml")

# agent setup
plans = load_xml("scenarios/equil/plans100.xml")

# event planner setup
listener = EventListener()

# simulation setup
sim = Sim(
    network=network,
    listener=listener,
)

# replanner setup
expected_link_durations = SimpleExpectedDurations(network.G)
router = StaticRouter(network=network, expectations=expected_link_durations)
replanner = TripPlanner()

# optimizer setup
optimizer = Optimizer(sim=sim, plans=plans, router=router, replanner=replanner)
optimizer.run(max_runs=MAX_RUNS)
