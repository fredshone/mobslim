"""Correctness against real data: mobslim on TGSIM I-90/I-94, two queues side by side.

TGSIM I-90/I-94 Stationary (FHWA / USDOT ITS JPO) is six runs of about 13
minutes each of the evening peak on northbound I-90/I-94 in Chicago, filmed
from a helicopter and tracked at 10 Hz. Runs 1, 3 and 5 cover a diverge,
runs 2, 4 and 6 a weaving section further on; each run has its own
reference image and coordinates. At both, the three left lanes queue while
the three right lanes flow. The data is not committed; put the trajectories
CSV from
https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-90-i-94-stationary-trajectories
in ``data/tgsim_i90/I90_94_stationary_trajectories.csv``.
"""
