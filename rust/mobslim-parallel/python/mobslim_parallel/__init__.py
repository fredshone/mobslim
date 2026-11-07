"""Compiled kernel for :class:`mobslim.mobsim.parallel.ParallelSim`.

``run`` takes the arguments of ``mobslim.mobsim.parallel.run_kernel`` (the
Python reference) plus ``node_part`` and ``n_parts``, runs one thread per
partition, and returns the same results.
"""

from mobslim_parallel._kernel import run

__all__ = ["run"]
