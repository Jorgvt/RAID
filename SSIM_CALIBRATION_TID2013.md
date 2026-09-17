# Psychometric Calibration of SSIM on TID2013 (JOD Scale)

This document provides the complete theoretical foundation, calibrated parameters, analytical inversion formulas, and benchmark results for calibrating the **Structural Similarity Index (SSIM)** into standardized **Just-Objectionable-Difference (JOD)** units using the **TID2013** dataset.

---

## 1. Mathematical Framework

### A. Raw Distance Definition
Given a reference image $I_{\text{ref}}$ and a distorted/translated image $I_{\text{dist}}$, the raw structural distance $d$ is defined as:

$$d = 1 - \text{SSIM}(I_{\text{ref}}, I_{\text{dist}}), \quad d \in [0, 2]$$

where $d = 0$ corresponds to identical images ($\text{SSIM} = 1.0$).

### B. The 4-Parameter Logistic Psychometric Model (VQEG / ITU-R BT.500)
To convert the artificial non-linear distance $d$ into human perceptual interval units ($\Delta\text{JOD}$), we apply the standard monotonic psychometric function:

$$f(d; \beta_1, \beta_2, \beta_3, \beta_4) = \beta_2 + \frac{\beta_1 - \beta_2}{1 + \exp\left(-\frac{d - \beta_3}{|\beta_4|}\right)}$$

where:
* $\beta_1$: Upper asymptotic perceptual saturation limit.
* $\beta_2$: Lower asymptote / baseline threshold.
* $\beta_3$: Transition inflection point ($d_0$).
* $\beta_4$: Scale growth parameter / transition steepness.

### C. Analytical Inverse Function ($f^{-1}(\text{JOD}) \to d$)
To calculate the raw SSIM distance required to produce a specific human perceptual effect (such as the **$1.0\text{ JOD} = 1\text{ JND}$** visibility threshold):

$$d(\text{JOD}) = f^{-1}(\text{JOD}) = \beta_3 - |\beta_4| \cdot \ln\left(\frac{\beta_1 - \beta_2}{\text{JOD} - \beta_2} - 1\right)$$

$$\text{SSIM}(\text{JOD}) = 1 - d(\text{JOD})$$

---

## 2. Calibration Standards & Results

Two calibration baselines were fitted and validated using **25-fold Leave-One-Reference-Scene-Out (GroupKFold)** cross-validation to guarantee out-of-content generalization:

| Calibration Standard | $\beta_1$ | $\beta_2$ | $\beta_3$ | $\beta_4$ | 25-Fold SROCC | 25-Fold PLCC | CV RMSE (JOD) | 1.0 JOD Threshold ($d$) | 1.0 JOD Threshold (SSIM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral Baseline (Additive Gaussian Noise)** | **3.3148** | **-3.6108** | **-0.0299** | **0.2722** | **0.9247** | **0.9232** | **0.288 JOD** | **$d = 0.1577$** | **$\text{SSIM} = 0.8423$** |
| **Multi-Distortion (Full TID2013 - 24 Types)** | **3.5170** | **0.6783** | **0.2034** | **0.0570** | **0.6507** | **0.6957** | **1.030 JOD** | **$d = 0.0862$** | **$\text{SSIM} = 0.9138$** |

---

## 3. Key Findings & Metric Bias

1. **Exceptional Linearity on Gaussian Noise:**  
   On classic neutral distortions (Additive Gaussian Noise), SSIM aligns closely with human psychophysical scaling ($PLCC = 0.9232$, $RMSE = 0.288\text{ JOD}$). A difference of $1\text{ JND}$ corresponds to $\text{SSIM} = 0.8423$ ($d = 0.1577$).
2. **Sub-threshold Error on Baseline:**  
   An $RMSE$ of $0.288\text{ JOD}$ is well below the sub-threshold limit ($< 0.5\text{ JOD}$), proving that the 4-parameter logistic mapping is an accurate converter for non-geometric degradation.
3. **Multi-Distortion Sensitivity:**  
   Across all 24 heterogeneous distortions, SSIM achieves $PLCC = 0.6957$ and $RMSE = 1.030\text{ JOD}$, demonstrating that SSIM's structural sensitivity varies across high-frequency artifacts vs. contrast/color shifts.

---

## 4. How to Evaluate Translation Visibility Thresholds

When evaluating translated images (shifting a reference image by $\Delta x$ pixels):

```
Reference Image I_ref ──┐
                        ├─> SSIM(I_ref, I_trans(Δx)) ──> d(Δx) = 1 - SSIM ──> JOD_pred = f(d(Δx))
Translated Image I_trans ┘
```

1. **Step 1:** Compute $\text{SSIM}(\Delta x)$ for increasing pixel shifts $\Delta x \in [0, 10]\text{ px}$.
2. **Step 2:** Convert to predicted perceptual distortion using the calibrated neutral converter:
   $$\hat{\text{JOD}}(\Delta x) = -3.6108 + \frac{6.9256}{1 + \exp\left(-\frac{(1 - \text{SSIM}(\Delta x)) + 0.0299}{0.2722}\right)}$$
3. **Step 3:** The metric's predicted **translation visibility threshold** $\Delta x_{\text{SSIM}}^*$ is the shift where:
   $$\hat{\text{JOD}}(\Delta x^*) = 1.0\text{ JOD} \iff \text{SSIM}(\Delta x^*) = 0.8423$$
4. **Step 4:** Compare $\Delta x_{\text{SSIM}}^*$ against the human visibility threshold $\Delta x_{\text{Human}}^*$ to determine the exact metric over-sensitivity factor:
   $$\text{Bias Factor} = \frac{\Delta x_{\text{Human}}^*}{\Delta x_{\text{SSIM}}^*}$$

---

## 5. Artifacts and Diagnostic Figures

* **Calibration Script:** [`calibrate_ssim_tid2013.py`](file:///media/disk/users/vitojor/raid/calibrate_ssim_tid2013.py)
* **Calibrated Dataset:** [`tid2013_ssim_calibrated.csv`](file:///media/disk/users/vitojor/raid/tid2013_ssim_calibrated.csv)
* **Gaussian Noise Calibration Curve:** [`Figures/ssim_calibration_gaussian_noise.png`](file:///media/disk/users/vitojor/raid/Figures/ssim_calibration_gaussian_noise.png)
* **Full TID2013 Calibration Curve:** [`Figures/ssim_calibration_full_tid2013.png`](file:///media/disk/users/vitojor/raid/Figures/ssim_calibration_full_tid2013.png)
* **Per-Distortion Breakdown:** [`Figures/ssim_per_distortion_performance.png`](file:///media/disk/users/vitojor/raid/Figures/ssim_per_distortion_performance.png)

---

## 6. Programmatic Usage (Zero-Recalculation API)

To load and use the pre-calculated calibration parameters without recomputing:

```python
from calibrated_metrics import CalibratedMetric

# Load pre-calibrated SSIM model (Gaussian Noise baseline standard)
ssim_cal = CalibratedMetric.load("SSIM", standard="gaussian_noise_baseline")

# 1. Map raw distance d = 1 - SSIM directly to human JOD
jod = ssim_cal.distance_to_jod(d=0.15)  # Returns ~0.95 JOD

# 2. Compute directly from reference and translated image files
jod_predicted = ssim_cal.predict_jod("reference.png", "translated_dx2.png")

# 3. Retrieve the exact 1 JND visibility threshold
print(f"SSIM 1 JND threshold: {ssim_cal.jnd_1_0_raw:.4f}")
```
