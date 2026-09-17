#!/usr/bin/env python3
"""
run_all_metrics_including_rmse.py
---------------------------------
Comprehensive benchmark and psychometric calibration of:
1. RMSE (Pixel Root Mean Squared Error)
2. SSIM (Structural Similarity Index)
3. LPIPS (Learned Perceptual Image Patch Similarity)
4. DISTS (Deep Image Structure and Texture Similarity)

Calibrates metrics against TID2013 human JOD scores, runs translation sweeps
across all 25 reference images in 4 cardinal directions, computes 1.0 JOD visibility
thresholds, and generates publication-quality figures.
"""

from pathlib import Path
import pandas as pd
from raid import (
    compute_or_load_tid2013_metrics,
    calibrate_all_metrics,
    run_translation_sweeps_all_references,
    plot_translation_overview,
    plot_cardinal_directions_comparison,
    CARDINAL_DIRECTIONS
)


def main():
    output_dir = Path('Figures')
    output_dir.mkdir(exist_ok=True)
    results_dir = Path('results_cardinal_directions')
    results_dir.mkdir(exist_ok=True)

    # 1. Compute or load precalculated metrics for TID2013
    print("\n[1/4] Loading / Computing TID2013 metric distances...")
    df_tid = compute_or_load_tid2013_metrics()

    # 2. Calibrate all 4 metrics against human JOD scale
    print("\n[2/4] Calibrating metrics against human JOD scale (TID2013)...")
    calib_params, df_calib = calibrate_all_metrics(df_tid)

    # 3. Run translation sweeps (s = 1..100 px) across 4 cardinal directions
    print("\n[3/4] Running translation sweeps across all 25 reference images...")
    df_sweep, df_thresh = run_translation_sweeps_all_references(
        max_shift=100,
        directions=CARDINAL_DIRECTIONS,
        results_dir=results_dir
    )

    # Save consolidated root datasets for convenience
    df_sweep.to_csv('all_metrics_translation_sweep_all_references.csv', index=False)
    df_thresh.to_csv('per_reference_thresholds_all_metrics.csv', index=False)

    # 4. Generate publication figures
    print("\n[4/4] Generating publication-quality figures...")
    comp_fig_path = output_dir / 'metrics_translation_comparison_rmse_ssim_lpips_dists.png'
    plot_translation_overview(df_sweep, df_thresh, output_path=comp_fig_path)
    print(f"Saved: {comp_fig_path}")

    cardinal_fig_path = output_dir / 'cardinal_directions_all_metrics_comparison.png'
    plot_cardinal_directions_comparison(df_thresh, output_path=cardinal_fig_path)
    print(f"Saved: {cardinal_fig_path}")

    # Summary Table
    print("\n" + "=" * 88)
    print("🎯 COMPARATIVE VISIBILITY THRESHOLD BENCHMARK (ALL 4 CARDINAL DIRECTIONS)")
    print("=" * 88)
    for m in ['rmse', 'ssim', 'lpips', 'dists']:
        print(f"--- Metric: {m.upper()} ---")
        for d in CARDINAL_DIRECTIONS:
            sub = df_thresh[df_thresh['direction'] == d]
            vals = sub[f'{m}_thresh_dx_1jnd_px']
            print(f"  • {d.upper():<5} : Mean = {vals.mean():.2f} ± {vals.std():.2f} px | Median = {vals.median():.2f} px | Range = [{vals.min():.2f}, {vals.max():.2f}] px")
    print("=" * 88)


if __name__ == '__main__':
    main()
