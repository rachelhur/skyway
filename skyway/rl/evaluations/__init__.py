"""Public API for the evaluations package.

Existing imports of the form

    from skyway.evaluations import build_evaluators, SingleStepEvaluator

continue to work after this refactor. The legacy `DataContainer` name now
points at the abstract base class — new code should construct
`SingleStepDataContainer` or `MultiStepDataContainer` directly (or just call
`build_evaluators`, which does this for you).
"""
from .helpers import (
    calc_airmass,
    calc_moon_dist,
    calc_moon_phase,
    calc_slew_distance,
    calc_sun_and_moon_pos,
)
from .plotters import (
    FILTER_COLORS,
    FILTER_PATCHES,
    EvaluationPlotter,
    PlotStyle,
)
from .data_container import (
    DataContainer,
    MultiStepDataContainer,
    SingleStepDataContainer,
)
from .evaluator import (
    Evaluator,
    MultiStepEvaluator,
    SingleStepEvaluator,
    build_evaluators,
)

# Legacy aliases (private names that some downstream code may import).
_FILTER_COLORS = FILTER_COLORS
_FILTER_PATCHES = FILTER_PATCHES

__all__ = [
    # helpers
    'calc_airmass', 'calc_slew_distance', 'calc_moon_dist',
    'calc_moon_phase', 'calc_sun_and_moon_pos',
    # plotting
    'EvaluationPlotter', 'PlotStyle', 'FILTER_COLORS', 'FILTER_PATCHES',
    # data
    'DataContainer', 'SingleStepDataContainer', 'MultiStepDataContainer',
    # evaluator
    'Evaluator', 'SingleStepEvaluator', 'MultiStepEvaluator', 'build_evaluators',
]
