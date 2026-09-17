#!/usr/bin/env python3
"""
evaluate_rotation_sweep.py
--------------------------
Evaluates rotation sweep for a single reference image across angles theta in [0.05, 30.0] deg
for RMSE, SSIM, LPIPS, and DISTS, and generates response plots.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from raid import (
    run_rotation_sweep_single,
    resolve_image_path,
    DEFAULT_TID_DIR,
    ROTATION_DIRECTIONS,
    DEFAULT_ROTATION_ANGLES,
    METRIC_COLORS
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})


def evaluate_single_image(ref_name="I01", crop_size=256, directions=ROTATION_DIRECTIONS):
    tid_dir = DEFAULT_TID_DIR / 'reference_images'
    output_dir = Path('Figures')
    output_dir.mkdir(exist_ok=True)

    if isinstance(directions, str):
        directions = ROTATION_DIRECTIONS if directions.lower() == 'all' else [directions.lower()]

    img_path = resolve_image_path(tid_dir, ref_name)
    df_res = run_rotation_sweep_single(
        img_input=img_path,
        ref_name=ref_name.upper(),
        crop_size=crop_size,
        angles=DEFAULT_ROTATION_ANGLES,
        directions=directions
    )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    for m_key, m_name in [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]:
        mean_c = df_res.groupby('angle_deg')[f'jod_{m_key}'].mean()
        ax1.plot(mean_c.index, mean_c.values, color=METRIC_COLORS[m_name], lw=2.4, label=m_name)

    ax1.axhline(1.0, color='black', ls='--', lw=1.3, label='1.0 JOD (1 JND)')
    ax1.set_xlabel('Rotation Angle $\\theta$ (degrees)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax1.set_title(f'Rotation Response for Reference {ref_name.upper()} ($\Delta\\theta \leq 30^\circ$)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 3.5)
    ax1.legend(loc='lower right', frameon=True)

    # Zoom-in <= 5 deg
    sub_zoom = df_res[df_res['angle_deg'] <= 5.0]
    for m_key, m_name in [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]:
        mean_z = sub_zoom.groupby('angle_deg')[f'jod_{m_key}'].mean()
        ax2.plot(mean_z.index, mean_z.values, marker='o', markersize=3.5, color=METRIC_COLORS[m_name], lw=2.2, label=m_name)

    ax2.axhline(1.0, color='black', ls='--', lw=1.3)
    ax2.set_xlabel('Rotation Angle $\\theta$ (degrees)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax2.set_title('Zoom-in: Small Rotations ($\Delta\\theta \leq 5^\circ$)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 3.5)
    ax2.set_xlim(0, 5.2)
    ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    fig_path = output_dir / f'rotation_sweep_{ref_name.lower()}.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved single reference plot to: {fig_path}")
    return df_res


if __name__ == '__main__':
    ref = sys.argv[1] if len(sys.argv) > 1 else "I01"
    evaluate_single_image(ref)
