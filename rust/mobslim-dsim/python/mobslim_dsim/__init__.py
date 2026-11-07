"""Compiled kernel for :class:`mobslim.mobsim.dsim.DSim`.

``run`` takes the arguments built by ``mobslim.mobsim.dsim.dsim_inputs``
plus ``node_part`` and ``n_parts``, runs one worker thread per partition,
and returns the same tuple as the other mobsim kernels.
"""

from mobslim_dsim._kernel import run

__all__ = ["run"]
