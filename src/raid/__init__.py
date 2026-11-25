"""
RAID: Response to Affine Transforms of Image Distance Metrics and Humans
========================================================================

A modular scientific package to benchmark, calibrate, and analyze the response
of image quality metrics (RMSE, SSIM, LPIPS, DISTS, etc.) and human observers to
spatial affine transformations (translation, rotation, scaling).
"""

from .utils import setup_cuda_env, get_device, find_tid2013_reference_images, resolve_image_path
from .psychometrics import (
    logistic_4p,
    inverse_logistic_4p,
    fit_logistic_4p,
    evaluate_predictions,
    compute_jnd_threshold
)
from .transforms import (
    CARDINAL_DIRECTIONS,
    ROTATION_DIRECTIONS,
    DEFAULT_ROTATION_ANGLES,
    get_translated_crops,
    get_rotated_crops
)
from .models import (
    METRIC_REGISTRY,
    register_metric,
    CalibratedMetric,
    DEFAULT_PARAMS_PATH,
    save_calibration_parameters
)
from .datasets import (
    DEFAULT_TID_DIR,
    load_tid2013_jod_dataframe,
    compute_or_load_tid2013_metrics,
    calibrate_all_metrics
)
from .sweeps import (
    get_default_calibrated_models,
    run_translation_sweep_single,
    run_translation_sweeps_all_references,
    run_rotation_sweep_single,
    run_rotation_sweeps_all_references
)
from .visualization import (
    plot_translation_overview,
    plot_cardinal_directions_comparison,
    plot_rotation_overview,
    plot_rotation_directions_comparison,
    METRIC_COLORS,
    DIRECTION_COLORS
)

__version__ = "0.1.0"
__all__ = [
    "setup_cuda_env",
    "get_device",
    "find_tid2013_reference_images",
    "resolve_image_path",
    "logistic_4p",
    "inverse_logistic_4p",
    "fit_logistic_4p",
    "evaluate_predictions",
    "compute_jnd_threshold",
    "CARDINAL_DIRECTIONS",
    "ROTATION_DIRECTIONS",
    "DEFAULT_ROTATION_ANGLES",
    "get_translated_crops",
    "get_rotated_crops",
    "METRIC_REGISTRY",
    "register_metric",
    "CalibratedMetric",
    "DEFAULT_PARAMS_PATH",
    "save_calibration_parameters",
    "DEFAULT_TID_DIR",
    "load_tid2013_jod_dataframe",
    "compute_or_load_tid2013_metrics",
    "calibrate_all_metrics",
    "get_default_calibrated_models",
    "run_translation_sweep_single",
    "run_translation_sweeps_all_references",
    "run_rotation_sweep_single",
    "run_rotation_sweeps_all_references",
    "plot_translation_overview",
    "plot_cardinal_directions_comparison",
    "plot_rotation_overview",
    "plot_rotation_directions_comparison",
    "METRIC_COLORS",
    "DIRECTION_COLORS"
]
