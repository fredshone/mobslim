"""Weave: parallel strands of innovation (see :mod:`mobslim.weave.controller`).

``Weave(scenario, strands=8, innovation_rate=0.1).run()`` runs eight
simulations per iteration, each with a different tenth of the population
innovating, and merges them into one record of everyone's plans and scores.
"""

from mobslim.weave.asynchronous import AsyncWeave
from mobslim.weave.controller import (
    Weave,
    WeaveResult,
    innovation_share,
    weave_rng,
)
from mobslim.weave.ledger import Ledger
from mobslim.weave.loom import (
    Loom,
    ProcessLoom,
    SerialLoom,
    default_workers,
    physical_cores,
)
from mobslim.weave.merge import (
    BACKGROUND,
    CANDIDATE,
    MeanReducer,
    MergeStats,
    Observations,
    PrecisionReducer,
    ScoreReducer,
    ScoreUpdate,
    merge,
)
from mobslim.weave.scheduler import (
    ConstantRate,
    InnovationSchedule,
    InnovationScheduler,
    LinearRate,
    StepRate,
    WeightedStalenessScheduler,
    deal,
    staleness,
)
from mobslim.weave.strand import (
    StrandContext,
    StrandJob,
    StrandResult,
    run_strand,
    strand_population,
)

__all__ = [
    "AsyncWeave", "BACKGROUND", "CANDIDATE", "ConstantRate", "InnovationSchedule", "InnovationScheduler",
    "Ledger", "LinearRate", "Loom", "MeanReducer", "MergeStats", "Observations",
    "PrecisionReducer", "ProcessLoom", "default_workers", "physical_cores", "ScoreReducer", "ScoreUpdate", "SerialLoom", "StepRate",
    "StrandContext", "StrandJob", "StrandResult", "Weave", "WeaveResult",
    "WeightedStalenessScheduler", "deal", "innovation_share", "merge", "run_strand",
    "staleness", "strand_population", "weave_rng",
]
