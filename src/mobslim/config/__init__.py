"""Run configuration front-ends.

Every front-end translates into the same Python API: keyword arguments for
:class:`~mobslim.controller.Controller` (scenario, mobsim, scorer,
strategies, ``ControllerConfig``). A run uses one of them, never both.

* :mod:`mobslim.config.mobslim`: a mobslim YAML config, with per-person
  values from filters on person attributes (``Controller.from_mobslim_config``;
  parameters in :mod:`mobslim.config.schema`, filters in
  :mod:`mobslim.config.conditions`).
* :mod:`mobslim.config.matsim`: a MATSim ``config.xml``
  (``Controller.from_matsim_config``); :mod:`mobslim.config.convert`
  translates one into a mobslim config.
"""

from mobslim.config.matsim import MatsimConfig, UnsupportedConfigError
from mobslim.config.mobslim import ConfigError, MobslimConfig

__all__ = ["ConfigError", "MatsimConfig", "MobslimConfig", "UnsupportedConfigError"]
