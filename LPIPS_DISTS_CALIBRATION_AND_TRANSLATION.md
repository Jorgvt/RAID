# Comparative Psychometric Calibration and Translation Visibility Thresholds: SSIM vs. LPIPS vs. DISTS

This document presents the full psychometric calibration against **TID2013** and the comparative evaluation of translation visibility thresholds for three major perceptual metrics:
1. **SSIM** (Structural Similarity Index, Wang et al., 2004)
2. **LPIPS** (Learned Perceptual Image Patch Similarity, Zhang et al., 2018 - VGG backbone via `piq`)
3. **DISTS** (Deep Image Structure and Texture Similarity, Ding et al., 2020 - via `piq`)

---

## 1. Calibration Standards & Results (TID2013)

All metrics were calibrated to standardized human **Just-Objectionable-Difference (JOD)** units using the 4-parameter logistic mapping:

$$f(d; \beta_1, \beta_2, \beta_3, \beta_4) = \beta_2 + \frac{\beta_1 - \beta_2}{1 + \exp\left(-\frac{d - \beta_3}{|\beta_4|}\right)}$$

Performance was evaluated using **25-fold Leave-One-Reference-Scene-Out (GroupKFold)** cross validation:

### A. Neutral Baseline Standard: Additive Gaussian Noise ($N=125$)

| Metric | Raw Distance $d$ | $\beta_1$ | $\beta_2$ | $\beta_3$ | $\beta_4$ | 25-Fold SROCC | 25-Fold PLCC | CV RMSE | 1.0 JOD Visibility Threshold ($d^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSIM** | $1 - \text{SSIM}$ | $3.3148$ | $-3.6107$ | $-0.0299$ | $0.2721$ | **$0.9247$** | **$0.9232$** | **$0.288\text{ JOD}$** | **$d = 0.1577$** ($\text{SSIM} = 0.8423$) |
| **LPIPS** | Deep $L_2$ distance | $3.1995$ | $-0.4070$ | $0.3006$ | $0.1527$ | **$0.9139$** | **$0.9091$** | **$0.312\text{ JOD}$** | **$d = 0.2323$** |
| **DISTS** | Texture + Struct distance | $2.5487$ | $0.0136$ | $0.1200$ | $0.0486$ | **$0.8986$** | **$0.8837$** | **$0.351\text{ JOD}$** | **$d = 0.0981$** |

### B. Multi-Distortion Standard: Full TID2013 ($N=3,000$, 24 Distortions)

| Metric | 25-Fold SROCC | 25-Fold PLCC | CV RMSE (JOD) | CV MAE (JOD) | 1.0 JOD Threshold ($d^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DISTS** | **$0.8285$** | **$0.8729$** | **$0.700\text{ JOD}$** | **$0.528\text{ JOD}$** | **$d = 0.0695$** |
| **LPIPS** | $0.7048$ | $0.8126$ | $0.836\text{ JOD}$ | $0.627\text{ JOD}$ | $d = 0.1535$ |
| **SSIM** | $0.6507$ | $0.6957$ | $1.030\text{ JOD}$ | $0.790\text{ JOD}$ | $d = 0.0862$ |

---

## 2. Horizontal Translation Sweeps ($\Delta x \in [1, 100]\text{ px}$)

Evaluating horizontal shifts across all 25 TID2013 reference images (measuring overlapping valid interiors to avoid boundary artifacts):

| Translation Metric | SSIM | LPIPS | DISTS |
| :--- | :---: | :---: | :---: |
| **Mean 1.0 JOD Visibility Threshold ($\Delta x^*$)** | **$0.80 \pm 0.27\text{ px}$** | **$5.66 \pm 1.70\text{ px}$** | **$41.69 \pm 17.64\text{ px}$** |
| **Median Visibility Threshold ($\Delta x^*$)** | **$0.79\text{ px}$** | **$5.36\text{ px}$** | **$37.50\text{ px}$** |
| **Range [Min, Max Threshold]** | $[0.45\text{ px}, 1.35\text{ px}]$ | $[3.39\text{ px}, 9.41\text{ px}]$ | $[18.62\text{ px}, 95.43\text{ px}]$ |
| **Mean Predicted Distortion at $\Delta x = 1\text{ px}$** | **$1.38\text{ JOD}$** | **$0.21\text{ JOD}$** | **$0.27\text{ JOD}$** |
| **Mean Predicted Distortion at $\Delta x = 2\text{ px}$** | **$2.18\text{ JOD}$** | **$0.42\text{ JOD}$** | **$0.32\text{ JOD}$** |
| **Mean Predicted Distortion at $\Delta x = 5\text{ px}$** | **$2.57\text{ JOD}$** | **$0.98\text{ JOD}$** | **$0.40\text{ JOD}$** |
| **Mean Predicted Distortion at $\Delta x = 10\text{ px}$** | **$2.67\text{ JOD}$** | **$1.53\text{ JOD}$** | **$0.48\text{ JOD}$** |
| **Mean Predicted Distortion at $\Delta x = 50\text{ px}$** | **$2.89\text{ JOD}$** | **$2.27\text{ JOD}$** | **$1.07\text{ JOD}$** |
| **Mean Predicted Distortion at $\Delta x = 100\text{ px}$** | **$2.96\text{ JOD}$** | **$2.39\text{ JOD}$** | **$1.39\text{ JOD}$** |

---

## 3. Scientific Insights & Comparative Discussion

```
  Metric Translation Invariance Hierarchy:
  
  SSIM:   [ 0.8 px ] ───> Extreme over-sensitivity (pixel grid alignment failure)
  LPIPS:  [ 5.7 px ] ───> Moderate shift sensitivity (pooling provides small invariance)
  DISTS:  [ 41.7 px] ───> High shift invariance (texture/structure spatial mean pooling)
```

1. **SSIM ($\Delta x^* = 0.80\text{ px}$):**  
   SSIM suffers from extreme spatial over-sensitivity due to local sliding pixel windows. Shifting by just 1 pixel immediately drops correlation and triggers $\approx 1.38\text{ JOD}$ ($>1\text{ JND}$), saturating by $5 - 10\text{ px}$.
2. **LPIPS ($\Delta x^* = 5.66\text{ px}$):**  
   LPIPS relies on intermediate VGG convolutional feature maps. While convolutional pooling layers provide small translation tolerance (shifts of $1 - 2\text{ px}$ produce sub-threshold responses $< 0.42\text{ JOD}$), spatial feature misalignments trigger $1\text{ JND}$ at $\sim 5.7\text{ pixels}$.
3. **DISTS ($\Delta x^* = 41.69\text{ px}$):**  
   DISTS explicitly separates structure and texture representations by computing spatial mean and standard deviation pooling of feature activations. Consequently, DISTS is remarkably robust to geometric translation, requiring on average **$\approx 42\text{ pixels}$** to register a $1\text{ JND}$ degradation.

---

## 4. Associated Files & Visualizations

* **Comparative Figure:** [`Figures/metrics_translation_comparison_ssim_lpips_dists.png`](file:///media/disk/users/vitojor/raid/Figures/metrics_translation_comparison_ssim_lpips_dists.png)
* **Saved Parameters:** [`calibration_parameters.json`](file:///media/disk/users/vitojor/raid/calibration_parameters.json)
* **Full Dataset Calibration:** [`tid2013_all_metrics_calibrated.csv`](file:///media/disk/users/vitojor/raid/tid2013_all_metrics_calibrated.csv)
* **Sweep Dataset (7,500 data points):** [`all_metrics_translation_sweep_all_references.csv`](file:///media/disk/users/vitojor/raid/all_metrics_translation_sweep_all_references.csv)
* **Per-Scene Thresholds:** [`per_reference_thresholds_all_metrics.csv`](file:///media/disk/users/vitojor/raid/per_reference_thresholds_all_metrics.csv)
* **Pipeline Script:** [`run_lpips_dists_full_pipeline.py`](file:///media/disk/users/vitojor/raid/run_lpips_dists_full_pipeline.py)
