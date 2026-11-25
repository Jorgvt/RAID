"""
raid.datasets
-------------
Dataset loaders, batch metric evaluation on TID2013 and RAID benchmarks,
granular per-metric storage, and cross-validated psychometric calibration engines.
"""

import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List, Sequence
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import torch
from torchvision import transforms

from .utils import get_device, resolve_image_path
from .psychometrics import fit_logistic_4p, logistic_4p, inverse_logistic_4p, evaluate_predictions
from .models import (
    METRIC_REGISTRY,
    CalibratedMetric,
    save_calibration_parameters,
    instantiate_metric_model,
    ALL_AVAILABLE_METRICS
)

DEFAULT_TID_DIR = Path('/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2013')
DEFAULT_GRANULAR_DIR = Path('results/granular')


def load_tid2013_jod_dataframe(csv_path: Union[str, Path] = 'tid2013_jod.csv') -> pd.DataFrame:
    """Loads TID2013 JOD CSV and assigns integer reference_id."""
    df = pd.read_csv(csv_path)
    df['reference_id'] = df['reference_file'].apply(lambda x: int(re.search(r'\d+', x).group()))
    return df


def compute_or_load_tid2013_metrics(metrics: Optional[Sequence[str]] = None,
                                    tid_csv_path: Union[str, Path] = 'tid2013_jod.csv',
                                    tid_dir: Union[str, Path] = DEFAULT_TID_DIR,
                                    cache_path: Union[str, Path] = 'tid2013_all_metrics_calibrated.csv',
                                    granular_dir: Union[str, Path] = DEFAULT_GRANULAR_DIR,
                                    batch_size: int = 32,
                                    device: Optional[Union[str, torch.device]] = None,
                                    force_recompute: bool = False) -> pd.DataFrame:
    """
    Incrementally computes metric distances for 3,000 TID2013 pairs.
    Checks granular per-metric store and consolidated CSV cache.
    Only newly requested or missing metric columns are computed and saved.
    """
    if metrics is None:
        metrics = ALL_AVAILABLE_METRICS
    metrics = [m.upper() for m in metrics]

    granular_dist_dir = Path(granular_dir) / 'benchmark_distances'
    granular_dist_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        df = pd.read_csv(cache_path)
    else:
        df = load_tid2013_jod_dataframe(tid_csv_path)

    # Check granular storage for any columns missing in consolidated df
    for m in metrics:
        col = f"{m.lower()}_dist"
        if col not in df.columns:
            m_file = granular_dist_dir / f"{m.lower()}_tid2013.csv"
            if m_file.exists() and not force_recompute:
                df_m = pd.read_csv(m_file)
                if col in df_m.columns and len(df_m) == len(df):
                    df[col] = df_m[col].values

    # Determine which metrics are still missing
    missing_metrics = []
    for m in metrics:
        col = f"{m.lower()}_dist"
        if col not in df.columns or force_recompute:
            missing_metrics.append(m)

    if not missing_metrics:
        df.to_csv(cache_path, index=False)
        return df

    print(f"Incremental computation needed for {len(missing_metrics)} metrics: {missing_metrics}")
    print("Existing cached metrics will NOT be recomputed.")

    dev = get_device(device)
    tid_dir = Path(tid_dir)
    ref_dir = tid_dir / 'reference_images'
    dist_dir = tid_dir / 'distorted_images'

    # Check for official SSIM file if SSIM is requested and missing
    if 'SSIM' in missing_metrics:
        official_ssim_file = tid_dir / 'metrics_values' / 'SSIM.txt'
        if official_ssim_file.exists():
            df['ssim'] = np.loadtxt(official_ssim_file)
            df['ssim_dist'] = 1.0 - df['ssim']
            # Save granular SSIM
            id_cols = [c for c in ['reference_file', 'distorted_file', 'distortion_type', 'distortion_level', 'jod_distortion'] if c in df.columns]
            df[id_cols + ['ssim_dist']].to_csv(
                granular_dist_dir / 'ssim_tid2013.csv', index=False
            )
            missing_metrics.remove('SSIM')

    if not missing_metrics:
        df.to_csv(cache_path, index=False)
        return df

    # Instantiate GPU models for deep / tensor-based metrics
    gpu_models = {}
    cpu_metrics = []
    for m in missing_metrics:
        model = instantiate_metric_model(m, device=dev)
        if model is not None:
            gpu_models[m] = model
        else:
            cpu_metrics.append(m)

    to_tensor = transforms.ToTensor()
    metric_results = {m: [] for m in missing_metrics}

    ref_tensors = []
    dist_tensors = []
    batch_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Computing missing metrics ({len(missing_metrics)})"):
        ref_path = resolve_image_path(ref_dir, row['reference_file'])
        dist_path = resolve_image_path(dist_dir, row['distorted_file'])

        img_ref = Image.open(ref_path).convert('RGB')
        img_dist = Image.open(dist_path).convert('RGB')

        # CPU metrics computation
        for m in cpu_metrics:
            if m == 'RMSE':
                arr_ref = np.array(img_ref, dtype=float)
                arr_dist = np.array(img_dist, dtype=float)
                metric_results['RMSE'].append(float(np.sqrt(np.mean((arr_ref - arr_dist)**2))))
            elif m == 'PSNR':
                import piq
                t_r = to_tensor(img_ref).unsqueeze(0).to(dev)
                t_d = to_tensor(img_dist).unsqueeze(0).to(dev)
                with torch.no_grad():
                    val = piq.psnr(t_r, t_d, data_range=1.0, reduction='none')
                    metric_results['PSNR'].append(float(1.0 / torch.clamp(val, min=1e-3).item()))
            elif m in METRIC_REGISTRY and METRIC_REGISTRY[m]['compute_fn'] is not None:
                fn = METRIC_REGISTRY[m]['compute_fn']
                metric_results[m].append(float(fn(img_ref, img_dist, device=dev)))

        # Batch GPU metrics
        if gpu_models:
            ref_tensors.append(to_tensor(img_ref))
            dist_tensors.append(to_tensor(img_dist))
            batch_indices.append(idx)

            if len(ref_tensors) == batch_size or idx == len(df) - 1:
                batch_ref = torch.stack(ref_tensors).to(dev)
                batch_dist = torch.stack(dist_tensors).to(dev)

                with torch.no_grad():
                    for gm_name, gm_model in gpu_models.items():
                        res_b = gm_model(batch_ref, batch_dist)
                        if isinstance(res_b, torch.Tensor):
                            arr = res_b.squeeze().cpu().numpy()
                            if arr.ndim == 0:
                                arr = [float(arr)]
                            else:
                                arr = arr.tolist()
                        else:
                            arr = [float(res_b)] * len(ref_tensors)
                        metric_results[gm_name].extend(arr)

                ref_tensors = []
                dist_tensors = []
                batch_indices = []

    # Save granular files
    id_cols = [c for c in ['reference_file', 'distorted_file', 'distortion_type', 'distortion_level', 'jod_distortion'] if c in df.columns]
    for m in missing_metrics:
        col = f"{m.lower()}_dist"
        df[col] = np.array(metric_results[m], dtype=float)
        # Save individual granular file
        df_granular = df[id_cols + [col]].copy()
        df_granular.to_csv(granular_dist_dir / f"{m.lower()}_tid2013.csv", index=False)

    df.to_csv(cache_path, index=False)
    print(f"Updated cache: {cache_path} with {missing_metrics}")
    return df


def calibrate_all_metrics(df: pd.DataFrame,
                          metrics: Optional[Sequence[str]] = None,
                          save_params: bool = True,
                          params_path: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs 25-fold Leave-One-Reference-Scene-Out cross validation and psychometric
    calibration for all specified (or detected) metrics against TID2013 JOD scores.
    """
    if metrics is None:
        # Detect all available metric columns ending in _dist
        dist_cols = [c for c in df.columns if c.endswith('_dist')]
        metrics = [c[:-5].upper() for c in dist_cols]

    all_calib_params = {}
    gkf = GroupKFold(n_splits=25)

    for metric_name in metrics:
        col = f"{metric_name.lower()}_dist"
        if col not in df.columns:
            continue

        d_all = df[col].values
        y_all = df['jod_distortion'].values
        formula = METRIC_REGISTRY.get(metric_name.upper(), {}).get('formula', f"{metric_name} distance")

        # A. Gaussian Noise Baseline
        df_gn = df[df['distortion_type'] == 'additive_gaussian_noise'].copy()
        d_gn = df_gn[col].values
        y_gn = df_gn['jod_distortion'].values

        p0_gn = [float(np.max(y_gn)), 0.0, float(np.median(d_gn)), float(np.std(d_gn) if np.std(d_gn) > 1e-4 else 1.0)]
        popt_gn = fit_logistic_4p(d_gn, y_gn, p0=p0_gn)
        b1_g, b2_g, b3_g, b4_g = popt_gn

        y_pred_cv_gn = np.zeros_like(y_gn)
        for train_idx, test_idx in gkf.split(d_gn, y_gn, groups=df_gn['reference_id']):
            popt_fold = fit_logistic_4p(d_gn[train_idx], y_gn[train_idx], p0=p0_gn)
            y_pred_cv_gn[test_idx] = logistic_4p(d_gn[test_idx], *popt_fold)

        res_gn = evaluate_predictions(y_gn, y_pred_cv_gn, d_gn)
        d_jnd_gn = inverse_logistic_4p(1.0, *popt_gn)

        # B. Full TID2013 Standard
        p0_all = [float(np.max(y_all)), 0.0, float(np.median(d_all)), float(np.std(d_all) if np.std(d_all) > 1e-4 else 1.0)]
        popt_all = fit_logistic_4p(d_all, y_all, p0=p0_all)
        b1_a, b2_a, b3_a, b4_a = popt_all

        y_pred_cv_all = np.zeros_like(y_all)
        for train_idx, test_idx in gkf.split(d_all, y_all, groups=df['reference_id']):
            popt_fold = fit_logistic_4p(d_all[train_idx], y_all[train_idx], p0=p0_all)
            y_pred_cv_all[test_idx] = logistic_4p(d_all[test_idx], *popt_fold)

        res_all = evaluate_predictions(y_all, y_pred_cv_all, d_all)
        d_jnd_all = inverse_logistic_4p(1.0, *popt_all)

        df[f'pred_jod_{metric_name.lower()}_cv'] = y_pred_cv_all
        raw_val_jnd = (1.0 - d_jnd_gn) if metric_name == 'SSIM' else d_jnd_gn

        all_calib_params[metric_name] = {
            "gaussian_noise_baseline": {
                "b1": float(b1_g),
                "b2": float(b2_g),
                "b3": float(b3_g),
                "b4": float(b4_g),
                "dataset": "TID2013",
                "subset": "additive_gaussian_noise",
                "distance_formula": formula,
                "jnd_1_0_d": float(d_jnd_gn),
                "jnd_1_0_raw": float(raw_val_jnd),
                "srocc": float(res_gn["SROCC"]),
                "plcc": float(res_gn["PLCC"]),
                "rmse_jod": float(res_gn["RMSE_JOD"]),
                "mae_jod": float(res_gn["MAE_JOD"])
            },
            "full_tid2013": {
                "b1": float(b1_a),
                "b2": float(b2_a),
                "b3": float(b3_a),
                "b4": float(b4_a),
                "dataset": "TID2013",
                "subset": "all_24_distortions",
                "distance_formula": formula,
                "jnd_1_0_d": float(d_jnd_all),
                "jnd_1_0_raw": float((1.0 - d_jnd_all) if metric_name == 'SSIM' else d_jnd_all),
                "srocc": float(res_all["SROCC"]),
                "plcc": float(res_all["PLCC"]),
                "rmse_jod": float(res_all["RMSE_JOD"]),
                "mae_jod": float(res_all["MAE_JOD"])
            }
        }

    if save_params:
        if params_path is not None:
            save_calibration_parameters(all_calib_params, filepath=params_path)
        else:
            save_calibration_parameters(all_calib_params)

    return all_calib_params, df

