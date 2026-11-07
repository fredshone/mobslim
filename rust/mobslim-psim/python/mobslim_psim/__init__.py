"""Compiled kernel for :class:`mobslim.mobsim.psim.PSim`.

``run`` takes the arguments of ``mobslim.mobsim.psim.run_kernel`` (the
Python reference) plus ``node_part`` and ``n_parts``, runs one thread per
partition, and returns the same results.
"""

from mobslim_psim._kernel import run

__all__ = ["run"]
