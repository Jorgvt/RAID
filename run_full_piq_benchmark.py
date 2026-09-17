#!/usr/bin/env python3
"""
run_full_piq_benchmark.py
-------------------------
Runs the complete full-reference metric evaluation and calibration pipeline
across all PIQ metrics + classical baselines on TID2013 and spatial sweeps.
Saves granular per-metric results and updates calibration parameters.
"""

from pathlib import Path
import time
import pandas as pd
import numpy as np

from raid import (
    compute_or_load_tid2013_metrics,
    calibrate_all_metrics,
    run_translation_sweeps_all_references,
    run_rotation_sweeps_all_references,
    CARDINAL_DIRECTIONS,
    ROTATION_DIRECTIONS
)
from raid.models import ALL_AVAILABLE_METRICS

TARGET_METRICS = [
    'RMSE', 'PSNR', 'SSIM', 'MS_SSIM', 'IW_SSIM',
    'VIF', 'FSIM', 'SRSIM', 'GMSD', 'MS_GMSD',
    'VSI', 'MDSI', 'HAARPSI', 'DSS',
    'LPIPS', 'DISTS', 'PIEAPP'
]


def main():
    t0 = time.time()
    results_dir = Path('results_all_piq_metrics')
    results_dir.mkdir(exist_ok=True, parents=True)
    granular_dir = Path('results/granular')
    granular_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = Path('Figures')
    fig_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("🚀 FULL-REFERENCE PIQ METRIC BENCHMARK & PSYCHOMETRIC CALIBRATION")
    print(f"Target metrics ({len(TARGET_METRICS)}): {', '.join(TARGET_METRICS)}")
    print("=" * 80)

    # 1. Compute or load TID2013 metric distances incrementally
    print("\n[Step 1/4] Incrementally computing TID2013 metric distances...")
    df_tid = compute_or_load_tid2013_metrics(
        metrics=TARGET_METRICS,
        batch_size=32,
        cache_path='tid2013_all_metrics_calibrated.csv',
        granular_dir=granular_dir
    )
    print(f"TID2013 dataset prepared: {len(df_tid)} rows, {len(df_tid.columns)} columns.")

    # 2. Psychometric calibration against TID2013 JOD scale
    print("\n[Step 2/4] Calibrating all metrics against human JOD scale (25-fold Scene CV)...")
    calib_params, df_calib = calibrate_all_metrics(
        df=df_tid,
        metrics=TARGET_METRICS,
        save_params=True
    )

    # Print TID2013 Gaussian Noise and Full TID performance table
    print("\n" + "=" * 92)
    print(f"{'METRIC':<12} | {'GN SROCC':<10} | {'GN PLCC':<10} | {'GN RMSE (JOD)':<14} | {'1.0 JND d':<12} | {'FULL SROCC':<10}")
    print("-" * 92)
    for m in TARGET_METRICS:
        if m in calib_params:
            gn = calib_params[m]['gaussian_noise_baseline']
            ft = calib_params[m]['full_tid2013']
            print(f"{m:<12} | {gn['srocc']:>10.4f} | {gn['plcc']:>10.4f} | {gn['rmse_jod']:>14.4f} | {gn['jnd_1_0_d']:>12.4e} | {ft['srocc']:>10.4f}")
    print("=" * 92)

    # 3. Translation Sweeps across all 25 references
    print("\n[Step 3/4] Running translation sweeps across all 25 reference images...")
    df_trans_sweep, df_trans_thresh = run_translation_sweeps_all_references(
        metric_names=TARGET_METRICS,
        standard="gaussian_noise_baseline",
        max_shift=100,
        directions=CARDINAL_DIRECTIONS,
        results_dir=results_dir / 'translation',
        granular_dir=granular_dir
    )
    df_trans_sweep.to_csv('all_metrics_translation_sweep_all_references.csv', index=False)
    df_trans_thresh.to_csv('per_reference_thresholds_all_metrics.csv', index=False)

    # 4. Rotation Sweeps across all 25 references
    print("\n[Step 4/4] Running rotation sweeps across all 25 reference images...")
    df_rot_sweep, df_rot_thresh = run_rotation_sweeps_all_references(
        metric_names=TARGET_METRICS,
        standard="gaussian_noise_baseline",
        crop_size=256,
        directions=ROTATION_DIRECTIONS,
        results_dir=results_dir / 'rotation',
        granular_dir=granular_dir
    )
    df_rot_sweep.to_csv('all_metrics_rotation_sweep_all_references.csv', index=False)
    df_rot_thresh.to_csv('per_reference_rotation_thresholds_all_metrics.csv', index=False)

    # Summary table of thresholds
    print("\n" + "=" * 90)
    print(f"🎯 COMPARATIVE 1.0 JOD VISIBILITY THRESHOLDS ACROSS ALL {len(TARGET_METRICS)} METRICS")
    print("=" * 90)
    print(f"{'METRIC':<12} | {'TRANS MEAN (px)':<16} | {'TRANS MED (px)':<15} | {'ROT MEAN (deg)':<16} | {'ROT MED (deg)':<15}")
    print("-" * 90)
    for m in TARGET_METRICS:
        m_k = m.lower()
        t_col = f"{m_k}_thresh_dx_1jnd_px"
        r_col = f"{m_k}_thresh_deg_1jnd"

        t_mean = df_trans_thresh[t_col].mean() if t_col in df_trans_thresh.columns else float('nan')
        t_med = df_trans_thresh[t_col].median() if t_col in df_trans_thresh.columns else float('nan')
        r_mean = df_rot_thresh[r_col].mean() if r_col in df_rot_thresh.columns else float('nan')
        r_med = df_rot_thresh[r_col].median() if r_col in df_rot_thresh.columns else float('nan')

        print(f"{m:<12} | {t_mean:>16.2f} | {t_med:>15.2f} | {r_mean:>16.2f} | {r_med:>15.2f}")
    print("=" * 90)

    elapsed = time.time() - t0
    print(f"\n🎉 Completed benchmark for all {len(TARGET_METRICS)} metrics in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes).")


if __name__ == '__main__':
    main()
