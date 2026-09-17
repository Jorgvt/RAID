# Comprehensive PIQ Full-Reference Metrics Benchmark and Psychometric Calibration

## 1. Executive Summary

This document details the scientific benchmark, psychometric calibration, and spatial affine invariance evaluation across the complete suite of **17 Full-Reference (FR) Image Quality Assessment (IQA) metrics** available in the `piq` library alongside classical baselines.

All metrics are calibrated into human **Just-Objectionable-Difference (JOD)** units via 4-parameter logistic psychometric mapping on the TID2013 dataset ($N=3{,}000$ pairs) using 25-fold Leave-One-Reference-Scene-Out cross-validation. We evaluate the spatial translation tolerance ($\Delta x^*$) and rotation tolerance ($\Delta\theta^*$) required to exceed $1.0\text{ JND}$ ($1.0\text{ JOD}$) across 25 natural reference scenes.

```
====================================================================================================
METRIC FAMILY          | METRICS INCLUDED                 | AFFINE INVARIANCE PROFILE
====================================================================================================
1. Deep Learned        | DISTS, PieAPP, LPIPS             | High Tolerance (5.4 - 34.5 px / 2.8 - 6.0 deg)
2. Structural/Contrast | SSIM, MS-SSIM, IW-SSIM           | Fragile (0.5 - 1.0 px / 0.3 - 0.6 deg)
3. Gradient & Saliency | FSIM, SRSIM, GMSD, MS-GMSD, VSI  | Extremely Hyper-sensitive (0.4 - 0.5 px / 0.1 - 0.2 deg)
4. Wavelet & Frequency | HaarPSI, DSS, MDSI, VIF          | Fragile (0.5 - 0.6 px / 0.1 - 0.4 deg)
5. Pixel Baselines     | RMSE, PSNR                       | Rigid (0.66 px / 0.37 deg)
====================================================================================================
```

---

## 2. Theoretical Grounding and Metric Inventory

| Metric | Paradigm | Distance Formulation ($d \ge 0$) | Key Scientific Reference |
| :--- | :--- | :--- | :--- |
| **RMSE** | Pixel $L_2$ | $\sqrt{\frac{1}{N}\sum(I_{\text{ref}} - I_{\text{dist}})^2}$ | Classic baseline |
| **PSNR** | Signal Log Ratio | $1 / \max(\text{PSNR}, 10^{-3})$ | Classic baseline |
| **SSIM** | Luminance / Contrast / Structure | $1.0 - \text{SSIM}$ | Wang et al., IEEE TIP 2004 |
| **MS-SSIM** | Multi-Scale Structural Similarity | $1.0 - \text{MS-SSIM}$ | Wang et al., Asilomar 2003 |
| **IW-SSIM** | Information-Weighted SSIM | $1.0 - \text{IW-SSIM}$ | Wang & Li, IEEE TIP 2011 |
| **VIF** | Visual Information Fidelity (Pixel) | $1.0 - \text{VIF}_p$ | Sheikh & Bovik, IEEE TIP 2006 |
| **FSIM** | Phase Congruency & Gradient Magnitude | $1.0 - \text{FSIM}$ | Zhang et al., IEEE TIP 2011 |
| **SRSIM** | Spectral Residual Visual Saliency | $1.0 - \text{SRSIM}$ | Zhang et al., IEEE TIP 2012 |
| **GMSD** | Gradient Magnitude Similarity Dev. | $\text{std}(\text{GMS})$ | Xue et al., IEEE SPL 2014 |
| **MS-GMSD** | Multi-Scale Gradient Deviation | $\sum w_s \cdot \text{GMSD}_s$ | Zhang et al., IEEE SPL 2017 |
| **VSI** | Visual Saliency-induced Index | $1.0 - \text{VSI}$ | Zhang et al., IEEE TIP 2014 |
| **MDSI** | Mean Deviation Similarity Index | $\text{std}(\text{gradient} \otimes \text{chroma})$ | Nafchi et al., IEEE SPL 2016 |
| **HaarPSI** | 2D Haar Wavelet Similarity | $1.0 - \text{HaarPSI}$ | Reisenhofer et al., Signal Processing 2018 |
| **DSS** | DCT Subband Similarity | $1.0 - \text{DSS}$ | Frosio et al., IEEE TIP 2017 |
| **LPIPS** | Deep Feature Distance (AlexNet) | $\sum_l \frac{1}{H_l W_l} \|w_l \odot (\hat{y}^l - \hat{y}_0^l)\|_2^2$ | Zhang et al., CVPR 2018 |
| **DISTS** | Deep Structure & Texture Similarity | $1 - (S_s \cdot S_t)$ (VGG16) | Ding et al., IEEE TPAMI 2020 |
| **PieAPP** | Pairwise Preference Deep Probability | Cross-layer probability error | Prashnani et al., CVPR 2018 |

---

## 3. Modular & Granular Architecture

To prevent redundant computation when extending or visualizing results, data is stored in modular, atomic tables:

```
results/granular/
├── benchmark_distances/
│   ├── dss_tid2013.csv
│   ├── fsim_tid2013.csv
│   ├── gmsd_tid2013.csv
│   ├── haarpsi_tid2013.csv
│   ├── iw_ssim_tid2013.csv
│   ├── lpips_tid2013.csv
│   ├── mdsi_tid2013.csv
│   ├── ms_gmsd_tid2013.csv
│   ├── ms_ssim_tid2013.csv
│   ├── pieapp_tid2013.csv
│   ├── psnr_tid2013.csv
│   ├── rmse_tid2013.csv
│   ├── srsim_tid2013.csv
│   ├── ssim_tid2013.csv
│   ├── vif_tid2013.csv
│   └── vsi_tid2013.csv
├── sweeps_translation/
│   └── <metric>_translation_sweep.csv (17 metrics)
└── sweeps_rotation/
    └── <metric>_rotation_sweep.csv (17 metrics)
```

- **Incremental Computation**: `compute_or_load_tid2013_metrics` and sweep functions inspect disk and evaluate *only* missing metrics on GPU.
- **Decoupled Plotting**: Any visualization function can query arbitrary subsets of metrics without rerunning inference.

---

## 4. Psychometric Calibration Parameters (TID2013)

Evaluated across 25-fold Leave-One-Reference-Scene-Out Cross-Validation:

| Metric | GN SROCC | GN PLCC | GN RMSE (JOD) | 1.0 JND Distance $d^*$ | Full TID SROCC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSE** | 0.9840 | 0.9904 | 0.1709 | 4.881 | 0.7093 |
| **PSNR** | 0.9840 | 0.9904 | 0.1709 | 0.0289 | 0.7093 |
| **SSIM** | 0.9702 | 0.9863 | 0.2037 | 0.0543 | 0.7600 |
| **MS-SSIM** | 0.9877 | 0.9912 | 0.1633 | 0.0163 | 0.8415 |
| **IW-SSIM** | 0.9845 | 0.9908 | 0.1668 | 0.0101 | 0.8522 |
| **VIF** | 0.9839 | 0.9881 | 0.1895 | 0.0469 | 0.7584 |
| **FSIM** | 0.9832 | 0.9909 | 0.1662 | 0.0270 | 0.8805 |
| **SRSIM** | 0.9821 | 0.9902 | 0.1726 | 0.0195 | 0.8712 |
| **GMSD** | 0.9841 | 0.9899 | 0.1754 | 0.0163 | 0.8795 |
| **MS-GMSD** | 0.9848 | 0.9904 | 0.1711 | 0.0167 | 0.8872 |
| **VSI** | 0.9844 | 0.9907 | 0.1683 | 0.0177 | 0.8874 |
| **MDSI** | 0.9836 | 0.9896 | 0.1776 | 0.0447 | 0.8893 |
| **HaarPSI** | 0.9856 | 0.9907 | 0.1684 | 0.0601 | 0.8951 |
| **DSS** | 0.9846 | 0.9907 | 0.1685 | 0.0401 | 0.8703 |
| **LPIPS** | 0.9744 | 0.9888 | 0.1843 | 0.0911 | 0.7699 |
| **DISTS** | 0.9806 | 0.9868 | 0.1994 | 0.0633 | 0.7963 |
| **PieAPP** | 0.9789 | 0.9880 | 0.1906 | 0.3168 | 0.8519 |

---

## 5. Comparative Spatial Affine Invariance Results

Visibility thresholds required to reach $1.0\text{ JOD}$ of perceived distortion:

| Metric Family | Metric | Translation Mean $\Delta x^*$ (px) | Translation Median (px) | Rotation Mean $\Delta\theta^*$ (deg) | Rotation Median (deg) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Deep Learned** | **DISTS** | **34.45** | **30.75** | **5.95°** | **6.00°** |
| | **PieAPP** | **11.68** | **12.07** | **3.95°** | **3.79°** |
| | **LPIPS** | **5.39** | **4.81** | **2.80°** | **2.54°** |
| **Structural** | **SSIM** | 1.01 | 0.87 | 0.45° | 0.41° |
| | **MS-SSIM** | 0.93 | 0.86 | 0.61° | 0.55° |
| | **IW-SSIM** | 0.49 | 0.48 | 0.32° | 0.31° |
| **Frequency/Wavelet**| **DSS** | 0.64 | 0.62 | 0.29° | 0.29° |
| | **VIF** | 0.57 | 0.54 | 0.40° | 0.38° |
| | **MDSI** | 0.52 | 0.48 | 0.14° | 0.13° |
| | **HaarPSI** | 0.51 | 0.50 | 0.24° | 0.22° |
| **Gradient/Saliency**| **VSI** | 0.52 | 0.48 | 0.16° | 0.15° |
| | **MS-GMSD** | 0.47 | 0.46 | 0.19° | 0.17° |
| | **GMSD** | 0.46 | 0.45 | 0.17° | 0.15° |
| | **FSIM** | 0.45 | 0.42 | 0.11° | 0.10° |
| | **SRSIM** | 0.44 | 0.41 | 0.10° | 0.09° |
| **Pixel Baselines** | **RMSE** | 0.66 | 0.58 | 0.37° | 0.30° |
| | **PSNR** | 0.66 | 0.58 | 0.37° | 0.30° |

---

## 6. Key Scientific Findings

1. **Hierarchy of Spatial Invariance**:
   $$\text{DISTS} \gg \text{PieAPP} > \text{LPIPS} \gg \text{MS-SSIM} \approx \text{SSIM} > \text{Pixel/RMSE} > \text{Wavelet/DSS} > \text{Gradient/Saliency (FSIM, SRSIM)}$$
   - Deep learned texture/structure models (**DISTS**) exhibit by far the largest affine tolerance ($\Delta x^* \approx 34.5\text{ px}$, $\Delta\theta^* \approx 6.0^\circ$).
   - Phase-congruency and gradient-saliency metrics (**FSIM**, **SRSIM**) are **hyper-sensitive**, flagging $< 0.5\text{ px}$ translations and $< 0.1^\circ$ rotations as noticeable distortion.
2. **Sub-Pixel Fragility of Hand-Crafted Metrics**:
   Almost all classical and gradient-based metrics exceed $1.0\text{ JND}$ at sub-pixel shifts ($\approx 0.45\text{ px}$), illustrating that hand-crafted metrics confound coordinate displacement with physical signal degradation.
