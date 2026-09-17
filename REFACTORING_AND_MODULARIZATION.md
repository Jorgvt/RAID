# Codebase Refactoring and Modular Architecture Guide

## Overview

The repository has been refactored from a collection of standalone script files into a modular Python package located at [`src/raid/`](file:///media/disk/users/vitojor/raid/src/raid) configured with the `uv` workflow.

---

## Package Architecture (`src/raid/`)

```
src/raid/
├── __init__.py          # Public package API exports
├── utils.py             # Hardware environment detection, CUDA runtime library loader, path helpers
├── transforms.py         # Inscribed cropping operators for spatial translation & rotation
├── psychometrics.py     # ITU-R BT.500 4P logistic model, inverse logistic, CV optimization, JND thresholding
├── models.py            # CalibratedMetric class for RMSE, SSIM, LPIPS, DISTS, parameter persistence
├── datasets.py          # TID2013 & RAID loaders, batched GPU metric computer, 25-fold CV calibration
├── sweeps.py            # High-level translation & rotation sweep engines (single scene or full dataset)
└── visualization.py     # Publication plotting routines (response profiles, boxplots, multi-panel figures)
```

---

## Module Breakdown

### 1. `raid.utils` ([`utils.py`](file:///media/disk/users/vitojor/raid/src/raid/utils.py))
- `setup_cuda_env()`: Automatically locates and dynamically loads NVIDIA CUDA `.so` runtime libraries from `.venv` into the current process.
- `get_device()`: Automatically selects `cuda:0` when available, falling back to `cpu`.
- `find_tid2013_reference_images(ref_dir)`: Case-insensitive discovery and numerical sorting of TID2013 references (`I01..I25`).
- `resolve_image_path(base_dir, name)`: Resolves `.bmp` / `.BMP` paths regardless of filesystem case-sensitivity.

### 2. `raid.psychometrics` ([`psychometrics.py`](file:///media/disk/users/vitojor/raid/src/raid/psychometrics.py))
- `logistic_4p(d, b1, b2, b3, b4)`: Standard ITU-R BT.500 4-parameter logistic mapping.
- `inverse_logistic_4p(jod, b1, b2, b3, b4)`: Analytic inversion of the 4P logistic function ($f^{-1}(\text{JOD}) \to d$).
- `fit_logistic_4p(x, y, p0, maxfev)`: Non-linear least squares optimization using Levenberg-Marquardt.
- `evaluate_predictions(y_true, y_pred, raw_d)`: Computes SROCC, KROCC, PLCC, RMSE (JOD), and MAE (JOD).
- `compute_jnd_threshold(stimuli, jod_curve, target_jod=1.0)`: Linear interpolation to extract exact visibility thresholds ($s^*$ or $\theta^*$).

### 3. `raid.transforms` ([`transforms.py`](file:///media/disk/users/vitojor/raid/src/raid/transforms.py))
- `get_translated_crops(arr, shift_px, direction)`: Directional cropping for `right`, `left`, `up`, and `down` translations without border artifacts.
- `get_rotated_crops(img, angle_deg, direction, crop_size=256)`: Centered square cropping strictly within the inscribed radius for `clockwise` and `counter_clockwise` rotations without black border padding.
- `DEFAULT_ROTATION_ANGLES`: Standard angle schedule $\theta \in [0.05^\circ, 30.0^\circ]$.

### 4. `raid.models` ([`models.py`](file:///media/disk/users/vitojor/raid/src/raid/models.py))
- `CalibratedMetric.load(metric_name, standard, device)`: Instant instantiation of pre-calibrated metrics (`RMSE`, `SSIM`, `LPIPS`, `DISTS`) using `calibration_parameters.json`.
- `distance_to_jod(d)` and `predict_jod(ref, dist)`: Maps distances directly to perceptual JOD units.
- `save_calibration_parameters(params_dict, filepath)`: Updates the centralized calibration registry.

### 5. `raid.datasets` ([`datasets.py`](file:///media/disk/users/vitojor/raid/src/raid/datasets.py))
- `compute_or_load_tid2013_metrics(...)`: Batched GPU calculation of RMSE, SSIM, LPIPS, and DISTS across 3,000 TID2013 pairs with CSV caching.
- `calibrate_all_metrics(df, save_params)`: 25-fold Leave-One-Reference-Scene-Out cross-validation across all 4 metrics.

### 6. `raid.sweeps` ([`sweeps.py`](file:///media/disk/users/vitojor/raid/src/raid/sweeps.py))
- `run_translation_sweep_single(...)` and `run_translation_sweeps_all_references(...)`: Executes translation sweeps for single scenes or all 25 TID2013 reference images across 4 cardinal directions.
- `run_rotation_sweep_single(...)` and `run_rotation_sweeps_all_references(...)`: Executes rotation sweeps for single scenes or all 25 reference images across rotation directions.

### 7. `raid.visualization` ([`visualization.py`](file:///media/disk/users/vitojor/raid/src/raid/visualization.py))
- `plot_translation_overview(...)`: 3-panel comparative translation benchmark overview.
- `plot_cardinal_directions_comparison(...)`: 4-panel cardinal direction breakdown.
- `plot_rotation_overview(...)`: 3-panel comparative rotation benchmark overview.
- `plot_rotation_directions_comparison(...)`: 4-panel clockwise vs counter-clockwise breakdown.

---

## Unified Command-Line Interface (`main.py`)

Run commands directly with `uv run`:

```bash
# General CLI help
uv run main.py --help

# Run translation sweep benchmark across all 25 references
uv run main.py translation

# Evaluate translation for a single reference image (e.g. I01)
uv run main.py translation --ref I01 --max_shift 100

# Run rotation sweep benchmark across all 25 references
uv run main.py rotation

# Evaluate rotation for a single reference image (e.g. I01)
uv run main.py rotation --ref I01 --crop_size 256

# Calibrate all metrics against TID2013
uv run main.py calibrate

# Run demo benchmark on RAID dataset
uv run main.py demo
```

---

## Cleaned / Archived Intermediate Scripts

To eliminate clutter and prevent divergence, the following superseded scripts have been removed:
- `add_rmse_baseline.py` $\to$ Replaced by unified [`raid.datasets.compute_or_load_tid2013_metrics`](file:///media/disk/users/vitojor/raid/src/raid/datasets.py#L26-L92)
- `run_lpips_dists_full_pipeline.py` $\to$ Replaced by unified [`run_all_metrics_including_rmse.py`](file:///media/disk/users/vitojor/raid/run_all_metrics_including_rmse.py#L1-L71)
- `calibrate_ssim_tid2013.py` $\to$ Replaced by [`raid.datasets.calibrate_all_metrics`](file:///media/disk/users/vitojor/raid/src/raid/datasets.py#L95-L177)
- `sweep_all_references_translation.py` $\to$ Replaced by [`raid.sweeps.run_translation_sweeps_all_references`](file:///media/disk/users/vitojor/raid/src/raid/sweeps.py#L111-L162)
