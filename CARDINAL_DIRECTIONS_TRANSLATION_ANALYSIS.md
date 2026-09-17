# Spatial Translation Invariance across Cardinal Directions: Comprehensive Benchmark (RMSE vs. SSIM vs. LPIPS vs. DISTS)

This document presents the systematic extension of the spatial translation experiments to all **four cardinal directions** ($\text{Right}, \text{Left}, \text{Up}, \text{Down}$) across all 25 reference images from the **TID2013** dataset ($384 \times 512$ resolution) evaluated up to 100 pixels of displacement ($s \in [1, 100]$ px).

---

## 1. Geometric and Spatial Crop Formulation

To evaluate pure spatial translation while strictly avoiding boundary padding/step artifacts, metric distances are evaluated on the exact overlapping spatial interior windows between the reference image $I$ and translated image $I_{\text{trans}}$:

| Cardinal Direction | Displacement Vector $(\Delta y, \Delta x)$ | Overlapping Reference Crop ($I_{\text{ref}}$) | Overlapping Shifted Crop ($I_{\text{trans}}$) | Valid Evaluation Area |
| :--- | :---: | :---: | :---: | :---: |
| **Right** | $(0, +s)$ | $I[:, :-s]$ | $I[:, s:]$ | $H \times (W - s)$ |
| **Left** | $(0, -s)$ | $I[:, s:]$ | $I[:, :-s]$ | $H \times (W - s)$ |
| **Down** | $(+s, 0)$ | $I[:-s, :]$ | $I[s:, :]$ | $(H - s) \times W$ |
| **Up** | $(-s, 0)$ | $I[s:, :]$ | $I[:-s, :]$ | $(H - s) \times W$ |

### Mathematical Symmetry
For any symmetric image metric $D(A, B) = D(B, A)$:
- **Horizontal Symmetry:** $D(I[:, :-s], I[:, s:]) = D(I[:, s:], I[:, :-s])$, meaning **Right** $\equiv$ **Left**.
- **Vertical Symmetry:** $D(I[:-s, :], I[s:, :]) = D(I[s:, :], I[:-s, :])$, meaning **Down** $\equiv$ **Up**.
- **Anisotropy (Horizontal vs. Vertical):** Natural images generally exhibit distinct spatial frequency distributions and power spectra along horizontal and vertical axes (e.g. gravitational orientation, horizon lines, vertical tree/structure boundaries), resulting in measurable differences between horizontal ($s_x$) and vertical ($s_y$) translation sensitivity.

---

## 2. Empirical Benchmark Results (1.0 JOD / 1 JND Visibility Thresholds)

All distances were evaluated and mapped to standardized **Just-Objectionable-Difference (JOD)** units via psychometric calibration against human perceptual scores from the TID2013 dataset ($N=25$ scenes, calibrated against the neutral Gaussian noise baseline):

$$f(d; \beta_1, \beta_2, \beta_3, \beta_4) = \beta_2 + \frac{\beta_1 - \beta_2}{1 + \exp\left(-\frac{d - \beta_3}{|\beta_4|}\right)}$$

### A. Visibility Threshold ($s^*$ in Pixels for $\Delta\mathrm{JOD} = 1.0$) by Direction

| Metric | Right (Mean $\pm$ Std) | Left (Mean $\pm$ Std) | Up (Mean $\pm$ Std) | Down (Mean $\pm$ Std) | Horizontal Axis ($s_x^*$) | Vertical Axis ($s_y^*$) | Directional Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSE (Pixel $L_2$)** | $0.68 \pm 0.28\text{ px}$ | $0.68 \pm 0.28\text{ px}$ | $0.64 \pm 0.26\text{ px}$ | $0.64 \pm 0.26\text{ px}$ | **$0.68\text{ px}$** | **$0.64\text{ px}$** | Vertical $+5.9\%$ more sensitive |
| **SSIM (Structural)** | $0.80 \pm 0.27\text{ px}$ | $0.80 \pm 0.27\text{ px}$ | $0.68 \pm 0.22\text{ px}$ | $0.68 \pm 0.22\text{ px}$ | **$0.80\text{ px}$** | **$0.68\text{ px}$** | Vertical $+15.0\%$ more sensitive |
| **LPIPS (Deep $L_2$)** | $5.66 \pm 1.70\text{ px}$ | $5.66 \pm 1.70\text{ px}$ | $5.11 \pm 1.57\text{ px}$ | $5.11 \pm 1.57\text{ px}$ | **$5.66\text{ px}$** | **$5.11\text{ px}$** | Vertical $+9.7\%$ more sensitive |
| **DISTS (Structure + Texture)** | $41.69 \pm 17.64\text{ px}$ | $41.69 \pm 17.64\text{ px}$ | $27.21 \pm 7.29\text{ px}$ | $27.21 \pm 7.29\text{ px}$ | **$41.69\text{ px}$** | **$27.21\text{ px}$** | Vertical $+34.7\%$ more sensitive |

---

## 3. Key Scientific Findings

1. **Exact Empirical Pairwise Symmetry:**
   - As predicted by spatial overlap slice symmetry, $\text{Threshold}(\text{Right}) = \text{Threshold}(\text{Left})$ and $\text{Threshold}(\text{Up}) = \text{Threshold}(\text{Down})$ across all 25 TID2013 scenes with $0.000$ discrepancy across all metrics.

2. **Vertical vs. Horizontal Anisotropy:**
   - Across all 4 metrics, natural images trigger a $1.0\text{ JOD}$ distortion threshold at **smaller vertical shifts than horizontal shifts** ($s_y^* < s_x^*$).
   - This aligns with natural image statistics where vertical luminance gradients (e.g., horizontal edges like ground planes, horizons, shadows) frequently dominate horizontal gradients.
   - The effect is especially pronounced in **DISTS** ($27.21\text{ px}$ vertical vs. $41.69\text{ px}$ horizontal), where deep texture correlations across spatial feature maps detect vertical translations earlier than horizontal sweeps.

3. **Invariance Hierarchy Preservation:**
   - The overall ranking of translation invariance is strictly preserved across all four directions:
     $$\text{DISTS} \gg \text{LPIPS} \gg \text{SSIM} > \text{RMSE}$$

---

## 4. Output Data Structure & Separated Files

All datasets are saved and organized in both consolidated and dedicated per-direction files:

### A. Dedicated Directory: `results_cardinal_directions/`
- **Per-Direction Sweeps:**
  - `all_metrics_sweep_right.csv`, `all_metrics_sweep_left.csv`, `all_metrics_sweep_up.csv`, `all_metrics_sweep_down.csv`
  - `ssim_sweep_right.csv`, `ssim_sweep_left.csv`, `ssim_sweep_up.csv`, `ssim_sweep_down.csv`
- **Per-Direction Thresholds:**
  - `all_metrics_thresholds_right.csv`, `all_metrics_thresholds_left.csv`, `all_metrics_thresholds_up.csv`, `all_metrics_thresholds_down.csv`
  - `ssim_thresholds_right.csv`, `ssim_thresholds_left.csv`, `ssim_thresholds_up.csv`, `ssim_thresholds_down.csv`
- **Consolidated Master CSVs:**
  - `all_metrics_sweep_all_cardinal_directions.csv` (10,000 evaluations: 25 scenes $\times$ 4 directions $\times$ 100 shifts)
  - `all_metrics_thresholds_all_cardinal_directions.csv` (100 condition thresholds)

### B. Generated Visualizations in `Figures/`
- [cardinal_directions_all_metrics_comparison.png](file:///media/disk/users/vitojor/raid/Figures/cardinal_directions_all_metrics_comparison.png): 4-panel threshold distributions across cardinal directions for RMSE, SSIM, LPIPS, and DISTS.
- [ssim_cardinal_directions_translation_benchmark.png](file:///media/disk/users/vitojor/raid/Figures/ssim_cardinal_directions_translation_benchmark.png): Detailed 4-panel SSIM comparison including response curves, zoom-in, directional threshold boxplots, and horizontal vs. vertical axis comparison.
- [metrics_translation_comparison_rmse_ssim_lpips_dists.png](file:///media/disk/users/vitojor/raid/Figures/metrics_translation_comparison_rmse_ssim_lpips_dists.png): Combined 4-metric comparison across the complete evaluation dataset.
