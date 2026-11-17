"""Correctness against real data: mobslim on TGSIM I-395, a freeway weaving section.

TGSIM (Third Generation Simulation Data, FHWA / USDOT ITS JPO) I-395 is two
hours of the morning peak on eastbound I-395 in Washington, DC, filmed from
three overpasses and tracked at 10 Hz: about 450 m of three-lane freeway
with a left off-ramp and a right on-ramp whose added lane drops 200 m later.
The data is not committed; put the trajectories CSV from
https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-395-trajectories
in ``data/tgsim/I395_trajectories.csv``.
"""
