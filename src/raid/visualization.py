"""
raid.visualization
------------------
Publication-quality plotting routines for psychometric calibration curves,
spatial translation and rotation response profiles, and visibility threshold distributions.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Sequence, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

METRIC_COLORS = {
    'RMSE': '#ff7f0e',
    'PSNR': '#8c564b',
    'SSIM': '#1f77b4',
    'MS_SSIM': '#aec7e8',
    'IW_SSIM': '#393b79',
    'VIF': '#17becf',
    'FSIM': '#2ca02c',
    'SRSIM': '#98df8a',
    'GMSD': '#bcbd22',
    'MS_GMSD': '#dbdb8d',
    'VSI': '#e377c2',
    'MDSI': '#f7b6d2',
    'HAARPSI': '#9467bd',
    'DSS': '#c5b0d5',
    'LPIPS': '#d62728',
    'DISTS': '#2ca02c',
    'PIEAPP': '#ff9896',
    'CONTENT_LOSS': '#7f7f7f',
    'STYLE_LOSS': '#c7c7c7'
}

DIRECTION_COLORS = {
    'right': '#1f77b4',
    'left': '#2ca02c',
    'up': '#d62728',
    'down': '#ff7f0e',
    'clockwise': '#1f77b4',
    'counter_clockwise': '#d62728'
}


def get_metric_color(metric_name: str) -> str:
    """Returns assigned color or fallback default color for any metric."""
    m_u = metric_name.upper()
    if m_u in METRIC_COLORS:
        return METRIC_COLORS[m_u]
    # Fallback to palette cycle
    import matplotlib.cm as cm
    return '#333333'



def plot_translation_overview(df_sweep: pd.DataFrame, df_thresh: pd.DataFrame,
                            max_zoom_shift: int = 15,
                            output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Generates the 3-panel translation benchmark overview:
    A. Mean Response Curves across 4 metrics (1..100 px)
    B. Zoom-in on small shifts (1..15 px)
    C. Threshold Boxplots and Swarm comparisons.
    """
    shifts = sorted(df_sweep['shift_px'].unique())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300)

    # Panel A: Mean Response Curves
    for m_key, m_name in [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]:
        mean_c = df_sweep.groupby('shift_px')[f'jod_{m_key}'].mean().values
        std_c = df_sweep.groupby('shift_px')[f'jod_{m_key}'].std().values
        col = METRIC_COLORS[m_name]
        ax1.plot(shifts, mean_c, color=col, lw=2.4, label=m_name)
        ax1.fill_between(shifts, mean_c - std_c, mean_c + std_c, color=col, alpha=0.12)

    ax1.axhline(1.0, color='black', ls='--', lw=1.3, label='1.0 JOD (1 JND)')
    ax1.set_xlabel('Translation Shift (pixels)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax1.set_title('A. Mean Translation Response (All Directions)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 3.5)
    ax1.legend(loc='lower right', frameon=True, fontsize=9.5)

    # Panel B: Zoom-in on small shifts
    sub_zoom = df_sweep[df_sweep['shift_px'] <= max_zoom_shift]
    shifts_z = [s for s in shifts if s <= max_zoom_shift]
    for m_key, m_name in [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]:
        mean_z = sub_zoom.groupby('shift_px')[f'jod_{m_key}'].mean().values
        std_z = sub_zoom.groupby('shift_px')[f'jod_{m_key}'].std().values
        col = METRIC_COLORS[m_name]
        ax2.plot(shifts_z, mean_z, marker='o', markersize=3.5, color=col, lw=2.2, label=m_name)
        ax2.fill_between(shifts_z, mean_z - std_z, mean_z + std_z, color=col, alpha=0.12)

    ax2.axhline(1.0, color='black', ls='--', lw=1.3)
    ax2.set_xlabel('Translation Shift (pixels)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax2.set_title(f'B. Zoom-in: Small Displacements ($\leq {max_zoom_shift}$ px)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 3.5)
    ax2.set_xlim(0.5, max_zoom_shift + 0.5)
    ax2.legend(loc='lower right', frameon=True, fontsize=9.5)

    # Panel C: Threshold Boxplots
    df_box = pd.DataFrame({
        'RMSE': df_thresh['rmse_thresh_dx_1jnd_px'],
        'SSIM': df_thresh['ssim_thresh_dx_1jnd_px'],
        'LPIPS': df_thresh['lpips_thresh_dx_1jnd_px'],
        'DISTS': df_thresh['dists_thresh_dx_1jnd_px']
    })

    pal = [METRIC_COLORS['RMSE'], METRIC_COLORS['SSIM'], METRIC_COLORS['LPIPS'], METRIC_COLORS['DISTS']]
    sns.boxplot(data=df_box, ax=ax3, palette=pal, width=0.45, boxprops=dict(alpha=0.3, lw=1.5), medianprops=dict(color='black', lw=2))
    sns.stripplot(data=df_box, ax=ax3, palette=pal, size=5.5, jitter=0.18, alpha=0.7)

    ax3.set_ylabel('1.0 JOD Visibility Threshold (pixels)', fontsize=11, fontweight='semibold')
    ax3.set_title('C. Visibility Thresholds (1 JND)\n($N=100$ Evaluations across 4 Directions)', fontsize=12, fontweight='bold')

    for idx_col, m_name in enumerate(['RMSE', 'SSIM', 'LPIPS', 'DISTS']):
        mean_val = df_box[m_name].mean()
        ax3.text(idx_col, mean_val + 0.12, f'{mean_val:.2f} px', ha='center', fontsize=9, fontweight='bold', color=METRIC_COLORS[m_name])

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(p, dpi=300)
    return fig


def plot_cardinal_directions_comparison(df_thresh: pd.DataFrame,
                                       output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Generates 2x2 multi-panel figure comparing visibility thresholds across 4 cardinal directions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    metric_list = [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]

    for idx_m, (m_key, m_label) in enumerate(metric_list):
        ax = axes[idx_m // 2, idx_m % 2]
        df_p = df_thresh.pivot(index='reference', columns='direction', values=f'{m_key}_thresh_dx_1jnd_px')
        c_order = [d for d in ['right', 'left', 'up', 'down'] if d in df_p.columns]
        df_p = df_p[c_order]

        pal = [DIRECTION_COLORS[c] for c in c_order]
        sns.boxplot(data=df_p, ax=ax, palette=pal, width=0.45, boxprops=dict(alpha=0.35, lw=1.5), medianprops=dict(color='black', lw=2))
        sns.stripplot(data=df_p, ax=ax, palette=pal, size=6.5, jitter=0.18, alpha=0.85)

        ax.set_ylabel('1.0 JOD Threshold (pixels)', fontsize=11, fontweight='semibold')
        ax.set_title(f'{m_label}: Thresholds Across Cardinal Directions', fontsize=12, fontweight='bold')

        for idx_c, c in enumerate(c_order):
            mv = df_p[c].mean()
            ax.text(idx_c, mv + (mv * 0.05 + 0.05), f'{mv:.2f} px', ha='center', fontsize=9.5, fontweight='bold', color=DIRECTION_COLORS[c])

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(p, dpi=300)
    return fig


def plot_rotation_overview(df_sweep: pd.DataFrame, df_thresh: pd.DataFrame,
                          max_zoom_angle: float = 5.0,
                          output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Generates the 3-panel rotation benchmark overview:
    A. Mean Response Curves across 4 metrics (0.05..30 deg)
    B. Zoom-in on small angles (<= 5 deg)
    C. Rotation Visibility Threshold Boxplots & Swarms.
    """
    angles = sorted(df_sweep['angle_deg'].unique())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300)

    # Panel A: Mean Response Curves
    for m_key, m_name in [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]:
        mean_c = df_sweep.groupby('angle_deg')[f'jod_{m_key}'].mean().values
        std_c = df_sweep.groupby('angle_deg')[f'jod_{m_key}'].std().values
        col = METRIC_COLORS[m_name]
        ax1.plot(angles, mean_c, color=col, lw=2.4, label=m_name)
        ax1.fill_between(angles, mean_c - std_c, mean_c + std_c, color=col, alpha=0.12)

    ax1.axhline(1.0, color='black', ls='--', lw=1.3, label='1.0 JOD Threshold (1 JND)')
    ax1.set_xlabel('Rotation Angle $\\theta$ (degrees)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax1.set_title('A. Rotation Response ($\Delta\\theta \in [0.05^\circ, 30.0^\circ]$)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 3.5)
    ax1.legend(loc='lower right', frameon=True, fontsize=9.5)

    # Panel B: Zoom-in on small angles
    sub_zoom = df_sweep[df_sweep['angle_deg'] <= max_zoom_angle]
    angles_z = [a for a in angles if a <= max_zoom_angle]
    for m_key, m_name in [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]:
        mean_z = sub_zoom.groupby('angle_deg')[f'jod_{m_key}'].mean().values
        std_z = sub_zoom.groupby('angle_deg')[f'jod_{m_key}'].std().values
        col = METRIC_COLORS[m_name]
        ax2.plot(angles_z, mean_z, marker='o', markersize=3.5, color=col, lw=2.2, label=m_name)
        ax2.fill_between(angles_z, mean_z - std_z, mean_z + std_z, color=col, alpha=0.12)

    ax2.axhline(1.0, color='black', ls='--', lw=1.3)
    ax2.set_xlabel('Rotation Angle $\\theta$ (degrees)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Predicted Distortion ($\Delta\mathrm{JOD}$)', fontsize=11, fontweight='semibold')
    ax2.set_title(f'B. Zoom-in: Small Rotations ($\Delta\\theta \leq {max_zoom_angle}^\circ$)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 3.5)
    ax2.set_xlim(0, max_zoom_angle + 0.2)
    ax2.legend(loc='lower right', frameon=True, fontsize=9.5)

    # Panel C: Threshold Boxplots
    df_box = pd.DataFrame({
        'RMSE': df_thresh['rmse_thresh_deg_1jnd'],
        'SSIM': df_thresh['ssim_thresh_deg_1jnd'],
        'LPIPS': df_thresh['lpips_thresh_deg_1jnd'],
        'DISTS': df_thresh['dists_thresh_deg_1jnd']
    })

    pal = [METRIC_COLORS['RMSE'], METRIC_COLORS['SSIM'], METRIC_COLORS['LPIPS'], METRIC_COLORS['DISTS']]
    sns.boxplot(data=df_box, ax=ax3, palette=pal, width=0.45, boxprops=dict(alpha=0.3, lw=1.5), medianprops=dict(color='black', lw=2))
    sns.stripplot(data=df_box, ax=ax3, palette=pal, size=5.5, jitter=0.18, alpha=0.7)

    ax3.set_ylabel('1.0 JOD Visibility Threshold $\\theta^*$ (degrees)', fontsize=11, fontweight='semibold')
    ax3.set_title('C. Rotation Visibility Thresholds (1 JND)\n($N=50$ Evaluations: 25 Scenes $\\times$ 2 Directions)', fontsize=12, fontweight='bold')

    offsets = [0.2, 0.2, 0.3, 0.4]
    for idx_col, m_name in enumerate(['RMSE', 'SSIM', 'LPIPS', 'DISTS']):
        mean_val = df_box[m_name].mean()
        ax3.text(idx_col, mean_val + offsets[idx_col], f'{mean_val:.2f}°', ha='center', fontsize=9, fontweight='bold', color=METRIC_COLORS[m_name])

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(p, dpi=300)
    return fig


def plot_rotation_directions_comparison(df_thresh: pd.DataFrame,
                                       output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Generates 2x2 multi-panel figure comparing CW vs CCW rotation directions across 4 metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    metric_list = [('rmse', 'RMSE'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('dists', 'DISTS')]

    for idx_m, (m_key, m_label) in enumerate(metric_list):
        ax = axes[idx_m // 2, idx_m % 2]
        df_p = df_thresh.pivot(index='reference', columns='direction', values=f'{m_key}_thresh_deg_1jnd')
        c_order = [d for d in ['clockwise', 'counter_clockwise'] if d in df_p.columns]
        df_p = df_p[c_order]

        pal = [DIRECTION_COLORS[c] for c in c_order]
        sns.boxplot(data=df_p, ax=ax, palette=pal, width=0.4, boxprops=dict(alpha=0.35, lw=1.5), medianprops=dict(color='black', lw=2))
        sns.stripplot(data=df_p, ax=ax, palette=pal, size=6.5, jitter=0.18, alpha=0.85)

        ax.set_ylabel('1.0 JOD Threshold (degrees)', fontsize=11, fontweight='semibold')
        ax.set_title(f'{m_label}: Rotation Thresholds by Direction', fontsize=12, fontweight='bold')

        for idx_c, c in enumerate(c_order):
            mv = df_p[c].mean()
            ax.text(idx_c, mv + (mv * 0.05 + 0.05), f'{mv:.2f}°', ha='center', fontsize=9.5, fontweight='bold', color=DIRECTION_COLORS[c])

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(p, dpi=300)
    return fig
