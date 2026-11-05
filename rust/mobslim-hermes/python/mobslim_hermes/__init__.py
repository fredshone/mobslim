"""Compiled kernel for :class:`mobslim.mobsim.hermes.HermesSim`.

``run`` has the same signature and results as
``mobslim.mobsim.hermes.run_kernel`` (the Python reference).
"""

from mobslim_hermes._kernel import run

__all__ = ["run"]
