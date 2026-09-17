#!/usr/bin/env python3
"""
run_all_metrics_rotation_benchmark.py
--------------------------------------
Comprehensive benchmark of spatial rotation invariance across 4 perceptual metrics:
1. RMSE (Pixel L2 Baseline)
2. SSIM (Structural Similarity Index)
3. LPIPS (Learned Perceptual Image Patch Similarity)
4. DISTS (Deep Image Structure and Texture Similarity)

Evaluates rotation sweeps theta in [0.05, 30.0] deg across all 25 TID2013 reference images
for both 'clockwise' and 'counter_clockwise' directions using central 256x256 crops.
"""

from pathlib import Path
import pandas as pd
from raid import (
    run_rotation_sweeps_all_references,
    plot_rotation_overview,
    plot_rotation_directions_comparison,
    ROTATION_DIRECTIONS,
    DEFAULT_ROTATION_ANGLES
)


def main():
    output_dir = Path('Figures')
    output_dir.mkdir(exist_ok=True)
    results_dir = Path('results_rotation')
    results_dir.mkdir(exist_ok=True)

    print("\n[1/3] Running rotation sweeps across all 25 reference images...")
    df_sweep, df_thresh = run_rotation_sweeps_all_references(
        crop_size=256,
        angles=DEFAULT_ROTATION_ANGLES,
        directions=ROTATION_DIRECTIONS,
        results_dir=results_dir
    )

    # Save consolidated root datasets for convenience
    df_sweep.to_csv('all_metrics_rotation_sweep_all_references.csv', index=False)
    df_thresh.to_csv('per_reference_rotation_thresholds_all_metrics.csv', index=False)

    print("\n[2/3] Generating publication-quality rotation figures...")
    comp_fig_path = output_dir / 'metrics_rotation_comparison_rmse_ssim_lpips_dists.png'
    plot_rotation_overview(df_sweep, df_thresh, output_path=comp_fig_path)
    print(f"Saved: {comp_fig_path}")

    rot_dir_fig_path = output_dir / 'rotation_directions_comparison_all_metrics.png'
    plot_rotation_directions_comparison(df_thresh, output_path=rot_dir_fig_path)
    print(f"Saved: {rot_dir_fig_path}")

    # Summary Table
    print("\n" + "=" * 88)
    print("🎯 COMPARATIVE ROTATION VISIBILITY THRESHOLD BENCHMARK (ALL 4 METRICS)")
    print("=" * 88)
    for m in ['rmse', 'ssim', 'lpips', 'dists']:
        print(f"--- Metric: {m.upper()} ---")
        for d in ROTATION_DIRECTIONS:
            sub = df_thresh[df_thresh['direction'] == d]
            vals = sub[f'{m}_thresh_deg_1jnd']
            print(f"  • {d.upper():<18} : Mean = {vals.mean():.2f} ± {vals.std():.2f}° | Median = {vals.median():.2f}° | Range = [{vals.min():.2f}°, {vals.max():.2f}°]")
    print("=" * 88)


if __name__ == '__main__':
    main()
