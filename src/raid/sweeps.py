"""
raid.sweeps
-----------
Unified sweep engines for spatial translation and rotation across
individual images and full benchmark datasets, with granular per-metric caching.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence, Union, List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from .utils import get_device, find_tid2013_reference_images, resolve_image_path
from .transforms import (
    CARDINAL_DIRECTIONS,
    ROTATION_DIRECTIONS,
    DEFAULT_ROTATION_ANGLES,
    get_translated_crops,
    get_rotated_crops
)
from .psychometrics import compute_jnd_threshold, logistic_4p
from .models import CalibratedMetric, ALL_AVAILABLE_METRICS

DEFAULT_TID_DIR = Path('/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2013')
DEFAULT_GRANULAR_DIR = Path('results/granular')


def get_default_calibrated_models(metric_names: Optional[Sequence[str]] = None,
                                  standard: str = "gaussian_noise_baseline",
                                  device: Optional[Union[str, torch.device]] = None) -> Dict[str, CalibratedMetric]:
    """Loads calibrated models for specified metrics (defaults to all available metrics)."""
    if metric_names is None:
        metric_names = ['RMSE', 'SSIM', 'LPIPS', 'DISTS']
    dev = get_device(device)
    models = {}
    for m in metric_names:
        try:
            models[m.upper()] = CalibratedMetric.load(m.upper(), standard=standard, device=dev)
        except Exception as e:
            print(f"Warning: Could not load calibrated model for '{m}': {e}")
    return models


# ----------------------------------------------------------------------
# 1. Spatial Translation Sweeps
# ----------------------------------------------------------------------

def run_translation_sweep_single(img_input: Union[str, Path, Image.Image, np.ndarray],
                                 ref_name: str = "Image",
                                 calib_models: Optional[Dict[str, CalibratedMetric]] = None,
                                 max_shift: int = 100,
                                 directions: Sequence[str] = CARDINAL_DIRECTIONS,
                                 device: Optional[Union[str, torch.device]] = None) -> pd.DataFrame:
    """
    Evaluates spatial translation sweep for a single image across specified cardinal directions.
    Dynamically computes distances and JOD values for all models in calib_models.
    """
    dev = get_device(device)
    if calib_models is None:
        calib_models = get_default_calibrated_models(device=dev)

    if isinstance(img_input, (str, Path)):
        img_rgb = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, np.ndarray):
        img_rgb = Image.fromarray(img_input).convert('RGB')
    else:
        img_rgb = img_input.convert('RGB')

    arr_rgb = np.array(img_rgb, dtype=float)
    shifts = list(range(1, max_shift + 1))
    records = []

    for direction in directions:
        for s_px in shifts:
            ref_crop, trans_crop = get_translated_crops(arr_rgb, s_px, direction=direction)
            ref_uint8 = ref_crop.astype(np.uint8)
            trans_uint8 = trans_crop.astype(np.uint8)

            row = {
                'reference': ref_name,
                'direction': direction,
                'shift_px': s_px
            }

            for m_name, model in calib_models.items():
                m_key = m_name.lower()
                d_val = model.compute_raw_distance(ref_uint8, trans_uint8)
                jod_val = model.distance_to_jod(d_val)
                row[f"{m_key}_d"] = float(d_val)
                row[f"jod_{m_key}"] = float(jod_val)
                if m_key == 'ssim':
                    row['ssim'] = float(1.0 - d_val)

            records.append(row)

    return pd.DataFrame(records)


def run_translation_sweeps_all_references(tid_dir: Union[str, Path] = DEFAULT_TID_DIR,
                                          calib_models: Optional[Dict[str, CalibratedMetric]] = None,
                                          metric_names: Optional[Sequence[str]] = None,
                                          standard: str = "gaussian_noise_baseline",
                                          max_shift: int = 100,
                                          directions: Sequence[str] = CARDINAL_DIRECTIONS,
                                          results_dir: Optional[Union[str, Path]] = None,
                                          granular_dir: Union[str, Path] = DEFAULT_GRANULAR_DIR,
                                          device: Optional[Union[str, torch.device]] = None,
                                          force_recompute: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes translation sweeps for all 25 TID2013 reference images across cardinal directions.
    Supports modular per-metric caching in results/granular/sweeps_translation/.
    Only missing metrics are computed.
    """
    dev = get_device(device)
    if calib_models is None:
        if metric_names is None:
            metric_names = ['RMSE', 'SSIM', 'LPIPS', 'DISTS']
        calib_models = get_default_calibrated_models(metric_names, standard=standard, device=dev)

    granular_sweep_dir = Path(granular_dir) / 'sweeps_translation'
    granular_sweep_dir.mkdir(parents=True, exist_ok=True)

    # Check which metrics are already cached in granular store
    cached_dfs = {}
    missing_models = {}
    for m_name, model in calib_models.items():
        m_file = granular_sweep_dir / f"{m_name.lower()}_translation_sweep.csv"
        if m_file.exists() and not force_recompute:
            cached_dfs[m_name.upper()] = pd.read_csv(m_file)
        else:
            missing_models[m_name.upper()] = model

    if missing_models:
        print(f"Running translation sweeps for {len(missing_models)} missing metrics: {list(missing_models.keys())}")
        ref_dir = Path(tid_dir) / 'reference_images'
        ref_files = find_tid2013_reference_images(ref_dir)

        all_sweep_records = []
        for ref_path in tqdm(ref_files, desc=f"Translation sweeps ({', '.join(missing_models.keys())})"):
            df_single = run_translation_sweep_single(
                img_input=ref_path,
                ref_name=ref_path.stem.upper(),
                calib_models=missing_models,
                max_shift=max_shift,
                directions=directions,
                device=dev
            )
            all_sweep_records.append(df_single)

        df_missing_sweep = pd.concat(all_sweep_records, ignore_index=True)

        # Save individual granular files for newly computed metrics
        for m_name in missing_models.keys():
            m_key = m_name.lower()
            cols = ['reference', 'direction', 'shift_px', f"{m_key}_d", f"jod_{m_key}"]
            if m_key == 'ssim' and 'ssim' in df_missing_sweep.columns:
                cols.append('ssim')
            df_m = df_missing_sweep[cols].copy()
            df_m.to_csv(granular_sweep_dir / f"{m_key}_translation_sweep.csv", index=False)
            cached_dfs[m_name] = df_m

    # Merge all metric DataFrames into a consolidated df_sweep
    all_metrics = list(calib_models.keys())
    base_df = cached_dfs[all_metrics[0]][['reference', 'direction', 'shift_px']].copy()
    for m_name in all_metrics:
        m_df = cached_dfs[m_name]
        m_key = m_name.lower()
        if f"{m_key}_d" in m_df.columns:
            base_df[f"{m_key}_d"] = m_df[f"{m_key}_d"].values
            base_df[f"jod_{m_key}"] = m_df[f"jod_{m_key}"].values
            if m_key == 'ssim' and 'ssim' in m_df.columns:
                base_df['ssim'] = m_df['ssim'].values

    df_sweep = base_df

    # Compute thresholds per reference, direction, and metric
    thresh_records = []
    metric_keys = [m.lower() for m in calib_models.keys()]

    for ref_name in df_sweep['reference'].unique():
        for direction in directions:
            sub = df_sweep[(df_sweep['reference'] == ref_name) & (df_sweep['direction'] == direction)].sort_values('shift_px')
            shifts_arr = sub['shift_px'].values
            row_res = {'reference': ref_name, 'direction': direction}

            for m in metric_keys:
                if f'jod_{m}' in sub.columns:
                    jod_curve = sub[f'jod_{m}'].values
                    t_val = compute_jnd_threshold(shifts_arr, jod_curve, target_jod=1.0)
                    row_res[f'{m}_thresh_dx_1jnd_px'] = t_val
                    row_res[f'{m}_jod_1px'] = float(jod_curve[0])
                    row_res[f'{m}_jod_2px'] = float(jod_curve[1])
                    row_res[f'{m}_jod_10px'] = float(jod_curve[9])

            thresh_records.append(row_res)

    df_thresh = pd.DataFrame(thresh_records)

    if results_dir is not None:
        res_p = Path(results_dir)
        res_p.mkdir(exist_ok=True, parents=True)
        for direction in directions:
            df_sweep[df_sweep['direction'] == direction].to_csv(res_p / f'all_metrics_sweep_{direction}.csv', index=False)
            df_thresh[df_thresh['direction'] == direction].to_csv(res_p / f'all_metrics_thresholds_{direction}.csv', index=False)
        df_sweep.to_csv(res_p / 'all_metrics_sweep_all_cardinal_directions.csv', index=False)
        df_thresh.to_csv(res_p / 'all_metrics_thresholds_all_cardinal_directions.csv', index=False)

    return df_sweep, df_thresh


# ----------------------------------------------------------------------
# 2. Spatial Rotation Sweeps
# ----------------------------------------------------------------------

def run_rotation_sweep_single(img_input: Union[str, Path, Image.Image, np.ndarray],
                              ref_name: str = "Image",
                              calib_models: Optional[Dict[str, CalibratedMetric]] = None,
                              crop_size: int = 256,
                              angles: Sequence[float] = DEFAULT_ROTATION_ANGLES,
                              directions: Sequence[str] = ROTATION_DIRECTIONS,
                              device: Optional[Union[str, torch.device]] = None) -> pd.DataFrame:
    """
    Evaluates spatial rotation sweep for a single image across specified rotation angles and directions.
    Dynamically computes distances and JOD values for all models in calib_models.
    """
    dev = get_device(device)
    if calib_models is None:
        calib_models = get_default_calibrated_models(device=dev)

    if isinstance(img_input, (str, Path)):
        img_rgb = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, np.ndarray):
        img_rgb = Image.fromarray(img_input).convert('RGB')
    else:
        img_rgb = img_input.convert('RGB')

    records = []

    for direction in directions:
        for angle in angles:
            ref_crop, rot_crop = get_rotated_crops(img_rgb, angle, direction=direction, crop_size=crop_size)

            row = {
                'reference': ref_name,
                'direction': direction,
                'angle_deg': float(angle)
            }

            for m_name, model in calib_models.items():
                m_key = m_name.lower()
                d_val = model.compute_raw_distance(ref_crop, rot_crop)
                jod_val = model.distance_to_jod(d_val)
                row[f"{m_key}_d"] = float(d_val)
                row[f"jod_{m_key}"] = float(jod_val)
                if m_key == 'ssim':
                    row['ssim'] = float(1.0 - d_val)

            records.append(row)

    return pd.DataFrame(records)


def run_rotation_sweeps_all_references(tid_dir: Union[str, Path] = DEFAULT_TID_DIR,
                                       calib_models: Optional[Dict[str, CalibratedMetric]] = None,
                                       metric_names: Optional[Sequence[str]] = None,
                                       standard: str = "gaussian_noise_baseline",
                                       crop_size: int = 256,
                                       angles: Sequence[float] = DEFAULT_ROTATION_ANGLES,
                                       directions: Sequence[str] = ROTATION_DIRECTIONS,
                                       results_dir: Optional[Union[str, Path]] = None,
                                       granular_dir: Union[str, Path] = DEFAULT_GRANULAR_DIR,
                                       device: Optional[Union[str, torch.device]] = None,
                                       force_recompute: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes rotation sweeps for all 25 TID2013 reference images across rotation directions.
    Supports modular per-metric caching in results/granular/sweeps_rotation/.
    Only missing metrics are computed.
    """
    dev = get_device(device)
    if calib_models is None:
        if metric_names is None:
            metric_names = ['RMSE', 'SSIM', 'LPIPS', 'DISTS']
        calib_models = get_default_calibrated_models(metric_names, standard=standard, device=dev)

    granular_sweep_dir = Path(granular_dir) / 'sweeps_rotation'
    granular_sweep_dir.mkdir(parents=True, exist_ok=True)

    # Check which metrics are already cached in granular store
    cached_dfs = {}
    missing_models = {}
    for m_name, model in calib_models.items():
        m_file = granular_sweep_dir / f"{m_name.lower()}_rotation_sweep.csv"
        if m_file.exists() and not force_recompute:
            cached_dfs[m_name.upper()] = pd.read_csv(m_file)
        else:
            missing_models[m_name.upper()] = model

    if missing_models:
        print(f"Running rotation sweeps for {len(missing_models)} missing metrics: {list(missing_models.keys())}")
        ref_dir = Path(tid_dir) / 'reference_images'
        ref_files = find_tid2013_reference_images(ref_dir)

        all_sweep_records = []
        for ref_path in tqdm(ref_files, desc=f"Rotation sweeps ({', '.join(missing_models.keys())})"):
            df_single = run_rotation_sweep_single(
                img_input=ref_path,
                ref_name=ref_path.stem.upper(),
                calib_models=missing_models,
                crop_size=crop_size,
                angles=angles,
                directions=directions,
                device=dev
            )
            all_sweep_records.append(df_single)

        df_missing_sweep = pd.concat(all_sweep_records, ignore_index=True)

        # Save individual granular files for newly computed metrics
        for m_name in missing_models.keys():
            m_key = m_name.lower()
            cols = ['reference', 'direction', 'angle_deg', f"{m_key}_d", f"jod_{m_key}"]
            if m_key == 'ssim' and 'ssim' in df_missing_sweep.columns:
                cols.append('ssim')
            df_m = df_missing_sweep[cols].copy()
            df_m.to_csv(granular_sweep_dir / f"{m_key}_rotation_sweep.csv", index=False)
            cached_dfs[m_name] = df_m

    # Merge all metric DataFrames into a consolidated df_sweep
    all_metrics = list(calib_models.keys())
    base_df = cached_dfs[all_metrics[0]][['reference', 'direction', 'angle_deg']].copy()
    for m_name in all_metrics:
        m_df = cached_dfs[m_name]
        m_key = m_name.lower()
        if f"{m_key}_d" in m_df.columns:
            base_df[f"{m_key}_d"] = m_df[f"{m_key}_d"].values
            base_df[f"jod_{m_key}"] = m_df[f"jod_{m_key}"].values
            if m_key == 'ssim' and 'ssim' in m_df.columns:
                base_df['ssim'] = m_df['ssim'].values

    df_sweep = base_df

    # Compute thresholds per reference, direction, and metric
    thresh_records = []
    metric_keys = [m.lower() for m in calib_models.keys()]

    for ref_name in df_sweep['reference'].unique():
        for direction in directions:
            sub = df_sweep[(df_sweep['reference'] == ref_name) & (df_sweep['direction'] == direction)].sort_values('angle_deg')
            angles_arr = sub['angle_deg'].values
            row_res = {'reference': ref_name, 'direction': direction}

            for m in metric_keys:
                if f'jod_{m}' in sub.columns:
                    jod_curve = sub[f'jod_{m}'].values
                    t_val = compute_jnd_threshold(angles_arr, jod_curve, target_jod=1.0)
                    row_res[f'{m}_thresh_deg_1jnd'] = t_val

                    for landmark in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
                        closest_idx = np.argmin(np.abs(angles_arr - landmark))
                        row_res[f'{m}_jod_{landmark}deg'] = float(jod_curve[closest_idx])

            thresh_records.append(row_res)

    df_thresh = pd.DataFrame(thresh_records)

    if results_dir is not None:
        res_p = Path(results_dir)
        res_p.mkdir(exist_ok=True, parents=True)
        for direction in directions:
            df_sweep[df_sweep['direction'] == direction].to_csv(res_p / f'all_metrics_rotation_sweep_{direction}.csv', index=False)
            df_thresh[df_thresh['direction'] == direction].to_csv(res_p / f'all_metrics_rotation_thresholds_{direction}.csv', index=False)
        df_sweep.to_csv(res_p / 'all_metrics_rotation_sweep_all_directions.csv', index=False)
        df_thresh.to_csv(res_p / 'all_metrics_rotation_thresholds_all_directions.csv', index=False)

    return df_sweep, df_thresh

