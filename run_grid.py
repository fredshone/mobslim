from mobslim.sim import Sim
from mobslim.planner import Planner, StaticRouter
from mobslim.network import Grid
from mobslim.agents import Plan
from mobslim.expected import SimpleExpectedDurations
from mobslim.listener import EventListener
from mobslim.optimizer import Optimizer

AGENTS = 20
SIZE = 5  # Size of the grid network
MAX_RUNS = 20  # Maximum number of runs for the simulation

# network setup
network = Grid(size=SIZE)
o = network.get_start()
d = network.get_end()
print(network)

# agent setup
plans = {}
for i in range(AGENTS):
    plan = Plan()
    plan.add_trip(origin=o, destination=d, start_time=0)
    plans[i] = plan

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
replanner = Planner()

# optimizer setup
optimizer = Optimizer(sim=sim, plans=plans, router=router, replanner=replanner)
optimizer.run(max_runs=MAX_RUNS)
