# Comprehensive Perceptual Calibration and Translation Visibility Benchmark: RMSE vs. SSIM vs. LPIPS vs. DISTS

This document provides the complete comparative benchmark across 4 representative perceptual and image distance metrics calibrated against the **TID2013** dataset and evaluated on spatial translations $\Delta x \in [1, 100]$ pixels across all 25 reference scenes:

1. **RMSE (Pixel $L_2$ Baseline):** Root Mean Squared Error across RGB pixel channels ($[0, 255]$ scale).
2. **SSIM (Structural Similarity Index):** Luminance windowed structural similarity ($d = 1 - \text{SSIM}$).
3. **LPIPS (Deep Feature Distance):** Learned Perceptual Image Patch Similarity using VGG backbone (via `piq`).
4. **DISTS (Deep Structure + Texture):** Deep Image Structure and Texture Similarity (via `piq`).

---

## 1. TID2013 Psychometric Calibration Results (25-Fold Out-of-Content CV)

All metrics were mapped into standardized human **Just-Objectionable-Difference (JOD)** units using 4-parameter logistic regression:

$$f(d; \beta_1, \beta_2, \beta_3, \beta_4) = \beta_2 + \frac{\beta_1 - \beta_2}{1 + \exp\left(-\frac{d - \beta_3}{|\beta_4|}\right)}$$

### A. Neutral Baseline Standard: Additive Gaussian Noise ($N=125$)

| Metric | Metric Distance $d$ | $\beta_1$ | $\beta_2$ | $\beta_3$ | $\beta_4$ | 25-Fold SROCC | 25-Fold PLCC | CV RMSE | 1.0 JOD Visibility Threshold ($d^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSE** | Pixel $L_2$ (0–255) | $3.5650$ | $-6.0237$ | $-4.7055$ | $14.4276$ | $0.8954$ | $0.8877$ | $0.345\text{ JOD}$ | **$d = 9.83\text{ intensity levels}$** ($3.85\%$) |
| **SSIM** | $1 - \text{SSIM}$ | $3.3148$ | $-3.6107$ | $-0.0299$ | $0.2721$ | **$0.9247$** | **$0.9232$** | **$0.288\text{ JOD}$** | **$d = 0.1577$** ($\text{SSIM} = 0.8423$) |
| **LPIPS** | Deep $L_2$ | $3.1995$ | $-0.4070$ | $0.3006$ | $0.1527$ | $0.9139$ | $0.9091$ | $0.312\text{ JOD}$ | **$d = 0.2323$** |
| **DISTS** | Texture + Struct | $2.5487$ | $0.0136$ | $0.1200$ | $0.0486$ | $0.8986$ | $0.8837$ | $0.351\text{ JOD}$ | **$d = 0.0981$** |

### B. Multi-Distortion Standard: Full TID2013 ($N=3,000$, 24 Distortions)

| Metric | 25-Fold SROCC | 25-Fold PLCC | CV RMSE (JOD) | CV MAE (JOD) | 1.0 JOD Threshold ($d^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DISTS** | **$0.8285$** | **$0.8729$** | **$0.700\text{ JOD}$** | **$0.528\text{ JOD}$** | $d = 0.0695$ |
| **LPIPS** | $0.7048$ | $0.8126$ | $0.836\text{ JOD}$ | $0.657\text{ JOD}$ | $d = 0.1535$ |
| **SSIM** | $0.6507$ | $0.6957$ | $1.030\text{ JOD}$ | $0.790\text{ JOD}$ | $d = 0.0862$ |
| **RMSE** | $0.6490$ | $0.6165$ | $1.129\text{ JOD}$ | $0.826\text{ JOD}$ | $d = 8.74$ |

---

## 2. Horizontal Translation Sweeps Across All 25 TID2013 References

Horizontal pixel translations $\Delta x \in [1, 100]$ px measured on the overlapping interior sub-arrays:

| Metric Evaluation | RMSE (Pixel $L_2$) | SSIM (Structural) | LPIPS (Deep $L_2$) | DISTS (Struct+Texture) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean 1.0 JOD Threshold ($\Delta x^*$)** | **$0.68 \pm 0.28\text{ px}$** | **$0.80 \pm 0.27\text{ px}$** | **$5.66 \pm 1.70\text{ px}$** | **$41.69 \pm 17.64\text{ px}$** |
| **Median Threshold ($\Delta x^*$)** | **$0.57\text{ px}$** | **$0.79\text{ px}$** | **$5.36\text{ px}$** | **$37.50\text{ px}$** |
| **Range [Min, Max]** | $[0.42\text{ px}, 1.29\text{ px}]$ | $[0.45\text{ px}, 1.35\text{ px}]$ | $[3.39\text{ px}, 9.41\text{ px}]$ | $[18.62\text{ px}, 95.43\text{ px}]$ |
| **Mean Distortion at $\Delta x = 1\text{ px}$** | **$1.72\text{ JOD}$** | **$1.38\text{ JOD}$** | **$0.21\text{ JOD}$** | **$0.27\text{ JOD}$** |
| **Mean Distortion at $\Delta x = 2\text{ px}$** | **$2.32\text{ JOD}$** | **$2.18\text{ JOD}$** | **$0.42\text{ JOD}$** | **$0.32\text{ JOD}$** |
| **Mean Distortion at $\Delta x = 5\text{ px}$** | **$2.81\text{ JOD}$** | **$2.57\text{ JOD}$** | **$0.98\text{ JOD}$** | **$0.40\text{ JOD}$** |
| **Mean Distortion at $\Delta x = 10\text{ px}$** | **$2.98\text{ JOD}$** | **$2.67\text{ JOD}$** | **$1.53\text{ JOD}$** | **$0.48\text{ JOD}$** |
| **Mean Distortion at $\Delta x = 50\text{ px}$** | **$3.19\text{ JOD}$** | **$2.89\text{ JOD}$** | **$2.27\text{ JOD}$** | **$1.07\text{ JOD}$** |
| **Mean Distortion at $\Delta x = 100\text{ px}$** | **$3.24\text{ JOD}$** | **$2.96\text{ JOD}$** | **$2.39\text{ JOD}$** | **$1.39\text{ JOD}$** |

---

## 3. Comparative Invariance Hierarchy

```
  Spatial Translation Robustness Spectrum:
  
  [0.68 px]  RMSE  ──> Pure point-wise pixel error (no spatial pooling / worst shift sensitivity)
  [0.80 px]  SSIM  ──> Local sliding window correlation (sub-pixel grid sensitivity)
  [5.66 px]  LPIPS ──> Deep convolutional hierarchy (moderate invariance via pooling)
  [41.69 px] DISTS ──> Structure + texture spatial statistics (highest geometric invariance)
```

1. **RMSE & SSIM (Sub-pixel Failure):**  
   Both point-wise pixel distances (RMSE: $0.68\text{ px}$) and local window covariance (SSIM: $0.80\text{ px}$) fail under 1-pixel shifts, triggering $>1.3 - 1.7\text{ JOD}$ ($>1\text{ JND}$) and saturating rapidly within $5 - 10\text{ px}$.
2. **LPIPS (Moderate Invariance):**  
   Deep feature hierarchies allow small shifts ($1 - 2\text{ px}$) to remain sub-threshold ($< 0.42\text{ JOD}$), but misalignment in deeper channels triggers $1\text{ JND}$ at $\approx 5.7\text{ pixels}$.
3. **DISTS (High Invariance):**  
   By computing spatial statistics (mean and standard deviation) across feature maps, DISTS demonstrates strong translation tolerance, requiring on average **$\approx 42\text{ pixels}$** to register $1\text{ JND}$.

---

## 4. Associated Files & Visualizations

* **Comparative 4-Metric Figure:** [`Figures/metrics_translation_comparison_rmse_ssim_lpips_dists.png`](file:///media/disk/users/vitojor/raid/Figures/metrics_translation_comparison_rmse_ssim_lpips_dists.png)
* **Local Parameters JSON:** [`calibration_parameters.json`](file:///media/disk/users/vitojor/raid/calibration_parameters.json)
* **All-Metrics Calibrated Dataset:** [`tid2013_all_metrics_calibrated.csv`](file:///media/disk/users/vitojor/raid/tid2013_all_metrics_calibrated.csv)
* **Per-Reference Thresholds:** [`per_reference_thresholds_all_metrics.csv`](file:///media/disk/users/vitojor/raid/per_reference_thresholds_all_metrics.csv)
* **Full Translation Sweep (10,000 points):** [`all_metrics_translation_sweep_all_references.csv`](file:///media/disk/users/vitojor/raid/all_metrics_translation_sweep_all_references.csv)
