"""
calibrated_metrics.py
---------------------
Backward-compatibility bridge for the `raid` library.
Deprecated: Please import directly from `raid` instead.
"""

import warnings
from raid import (
    logistic_4p,
    inverse_logistic_4p,
    CARDINAL_DIRECTIONS,
    ROTATION_DIRECTIONS,
    get_translated_crops,
    get_rotated_crops,
    CalibratedMetric,
    DEFAULT_PARAMS_PATH,
    save_calibration_parameters
)

warnings.warn(
    "Importing from `calibrated_metrics` is deprecated. Use `import raid` or `from raid import ...` instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "logistic_4p",
    "inverse_logistic_4p",
    "CARDINAL_DIRECTIONS",
    "ROTATION_DIRECTIONS",
    "get_translated_crops",
    "get_rotated_crops",
    "CalibratedMetric",
    "DEFAULT_PARAMS_PATH",
    "save_calibration_parameters"
]
