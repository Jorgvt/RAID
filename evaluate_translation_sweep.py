#!/usr/bin/env python3
"""
evaluate_translation_sweep.py
-----------------------------
Translates a reference image across cardinal directions ('right', 'left', 'up', 'down')
for shift in [1, 100] pixels, computes metric responses, and plots response curves.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from raid import (
    run_translation_sweep_single,
    resolve_image_path,
    DEFAULT_TID_DIR,
    CARDINAL_DIRECTIONS,
    DIRECTION_COLORS,
    METRIC_COLORS
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})


def evaluate_single_image(ref_name="I01", max_shift=100, directions=CARDINAL_DIRECTIONS):
    tid_dir = DEFAULT_TID_DIR / 'reference_images'
    output_dir = Path('Figures')
    output_dir.mkdir(exist_ok=True)

    if isinstance(directions, str):
        directions = CARDINAL_DIRECTIONS if directions.lower() == 'all' else [directions.lower()]

    img_path = resolve_image_path(tid_dir, ref_name)
    df_res = run_translation_sweep_single(
        img_input=img_path,
        ref_name=ref_name.upper(),
        max_shift=max_shift,
        directions=directions
    )

    # Plot SSIM translation response
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    for d in directions:
        sub_d = df_res[df_res['direction'] == d]
        ax1.plot(sub_d['shift_px'], sub_d['jod_ssim'], color=DIRECTION_COLORS.get(d, 'black'), lw=2.2, label=f'{d.capitalize()}')

    ax1.axhline(1.0, color='black', ls='--', lw=1.5, label='1.0 JOD Threshold (1 JND)')
    ax1.set_xlabel('Spatial Translation Shift (pixels)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax1.set_title(f'SSIM Response for Reference {ref_name.upper()} (Shift $\in [1, {max_shift}]$ px)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(0, 3.5)

    # Zoom-in <= 15 px
    sub_zoom = df_res[df_res['shift_px'] <= 15]
    for d in directions:
        sub_dz = sub_zoom[sub_zoom['direction'] == d]
        ax2.plot(sub_dz['shift_px'], sub_dz['jod_ssim'], marker='o', markersize=4,
                 color=DIRECTION_COLORS.get(d, 'black'), lw=2.0, label=f'{d.capitalize()}')

    ax2.axhline(1.0, color='black', ls='--', lw=1.5, label='1.0 JOD Threshold')
    ax2.set_xlabel('Spatial Translation Shift (pixels)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax2.set_title('Zoom-in: Small Displacements (Shift $\leq 15$ px)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.set_ylim(0, 3.5)
    ax2.set_xlim(0.5, 15.5)

    plt.tight_layout()
    fig_path = output_dir / f'ssim_translation_sweep_{ref_name.lower()}_cardinal_directions.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Results for reference image {ref_name.upper()}:")
    for d in directions:
        sub_d = df_res[df_res['direction'] == d]
        print(f"  • Direction: {d.upper():<5} | 1px: {sub_d['jod_ssim'].iloc[0]:.2f} JOD | 2px: {sub_d['jod_ssim'].iloc[1]:.2f} JOD | 5px: {sub_d['jod_ssim'].iloc[4]:.2f} JOD | 10px: {sub_d['jod_ssim'].iloc[9]:.2f} JOD")
    print(f"Saved plot to: {fig_path}")
    return df_res


if __name__ == '__main__':
    ref = sys.argv[1] if len(sys.argv) > 1 else "I01"
    evaluate_single_image(ref)
