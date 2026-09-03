# Practical Guide: Benchmarking Quality Metrics using JOD Scales

This guide explains how to evaluate pretrained Image Quality Assessment (IQA) and perceptual distance metrics (e.g., **LPIPS**, **DISTS**, **PieAPP**, **SSIM**, **CLIP-IQA**) against the **RAID** and **TID2013** datasets, expressing prediction accuracy directly in **Just-Objectionable-Difference (JOD)** units.

---

## 1. The Core Challenge: Metric Distances vs. JOD Units

Most pretrained image quality metrics output arbitrary values:
* Deep feature distances (e.g., $L_2$ feature space distances in LPIPS/DISTS).
* Mean Opinion Scores (MOS) bounded between $[1, 5]$ or $[0, 100]$ subject to subjective ceiling and floor saturations.

Because **JOD is an interval scale with direct physical/probabilistic meaning** ($1\text{ JOD} =$ 75% human discrimination threshold), evaluating raw uncalibrated distances against JOD directly would distort linear error metrics ($RMSE$, $PLCC$).

---

## 2. Recommended Methodology: Post-Hoc Monotonic Calibration (Approach 1)

In perceptual research and international standards (**ITU-R BT.500** and **VQEG**), the standard protocol for benchmarking IQA metrics is **non-linear psychometric calibration**:

```
 ┌───────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────┐
 │   Reference Image     │         │   Pretrained IQA Metric   │         │    Raw Distance $d$   │
 │   + Distorted Image   │ ──────> │ (LPIPS, DISTS, SSIM, ...) │ ──────> │  (arbitrary scale)    │
 └───────────────────────┘         └───────────────────────────┘         └───────────┬───────────┘
                                                                                     │
                                                                                     ▼
 ┌───────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────┐
 │ Evaluated Performance │         │  Predicted $\hat{\text{JOD}}$    │         │  4-Parameter Logistic │
 │  • RMSE (in JOD)      │ <────── │  (Calibrated JOD units)   │ <────── │      Regression       │
 │  • Pearson $r$ (PLCC) │         └───────────────────────────┘         └───────────────────────┘
 └───────────────────────┘
```

### Why this approach is optimal:
1. **Preserves Native Ranking:** Because the calibration function $f(d)$ is strictly monotonic, **Spearman ($\rho$) and Kendall ($\tau$) rank correlations remain unchanged**.
2. **Physically Meaningful Errors:** Prediction errors ($RMSE$, $MAE$) are measured directly in **JOD units** (e.g., *"Metric A predicts human perception with an average error of $\pm 0.38\text{ JOD}$"*).
3. **Zero Risk of Overfitting:** Fitting a 4-parameter logistic curve across hundreds of conditions requires no model retraining and cannot distort internal visual representations.

---

## 3. Mathematical Formulation: 4-Parameter Logistic Function

The standard mapping function recommended by VQEG is the **4-Parameter Logistic Function**:

$$f(d; \beta_1, \beta_2, \beta_3, \beta_4) = \beta_2 + \frac{\beta_1 - \beta_2}{1 + \exp\left(-\frac{d - \beta_3}{|\beta_4|}\right)}$$

* $d$: Raw distance/score output by the IQA metric.
* $\beta_1, \beta_2$: Asymptotic maximum and minimum values of the target scale.
* $\beta_3$: Midpoint inflection parameter (transition threshold).
* $\beta_4$: Growth rate / transition steepness.

---

## 4. Complete Python Implementation

The following complete routine benchmarks an IQA metric on [`raid_jod.csv`](file:///Users/jorgvt/Developer/RAID/raid_jod.csv):

```python
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import KFold

# 1. Define standard 4-Parameter Logistic Mapping
def logistic_4p(d, b1, b2, b3, b4):
    return b2 + (b1 - b2) / (1.0 + np.exp(-(d - b3) / np.abs(b4)))

def evaluate_metric_jod(raw_distances, y_true_jod, n_splits=5):
    """
    Evaluates raw metric distances against ground-truth JOD scores using
    cross-validated 4-parameter logistic calibration.
    
    Parameters:
    -----------
    raw_distances : array-like, shape (N,)
        Raw distances or scores predicted by the IQA model.
    y_true_jod : array-like, shape (N,)
        Ground truth JOD distortion values (from raid_jod.csv).
    n_splits : int
        Number of cross-validation folds.
    """
    raw_d = np.array(raw_distances, dtype=float)
    y_true = np.array(y_true_jod, dtype=float)
    
    # Cross-validated prediction array
    y_pred_jod = np.zeros_like(y_true)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(raw_d):
        d_train, y_train = raw_d[train_idx], y_true[train_idx]
        d_test = raw_d[test_idx]
        
        # Initial parameter estimates
        p0 = [np.min(y_train), np.max(y_train), np.median(d_train), 1.0]
        
        # Fit 4-parameter logistic curve
        try:
            popt, _ = curve_fit(logistic_4p, d_train, y_train, p0=p0, maxfev=20000)
        except RuntimeError:
            popt = p0
            
        y_pred_jod[test_idx] = logistic_4p(d_test, *popt)
        
    # Compute standard evaluation metrics
    srocc = spearmanr(raw_d, y_true)[0]               # Rank correlation (monotonicity)
    krocc = kendalltau(raw_d, y_true)[0]              # Kendall tau
    plcc  = pearsonr(y_pred_jod, y_true)[0]           # Linear correlation after calibration
    rmse  = np.sqrt(np.mean((y_pred_jod - y_true)**2)) # RMSE in physical JOD units
    mae   = np.mean(np.abs(y_pred_jod - y_true))       # MAE in physical JOD units
    
    print("=" * 45)
    print("🎯 BENCHMARK RESULTS (in JOD units)")
    print("=" * 45)
    print(f"  • Spearman Rho  (SROCC) : {srocc:.4f} (Monotonicity)")
    print(f"  • Kendall Tau   (KROCC) : {krocc:.4f}")
    print(f"  • Pearson r     (PLCC)  : {plcc:.4f}  (Linearity)")
    print(f"  • RMSE (in JOD units)   : {rmse:.3f} JOD")
    print(f"  • MAE  (in JOD units)   : {mae:.3f} JOD")
    print("=" * 45)
    
    return {
        'SROCC': srocc,
        'KROCC': krocc,
        'PLCC': plcc,
        'RMSE_JOD': rmse,
        'MAE_JOD': mae,
        'y_pred_jod': y_pred_jod
    }
```

---

## 5. Interpreting Evaluation Metrics in JOD Units

When reporting your benchmark results, the metrics have direct physical significance:

| Metric | Target | Perceptual Meaning |
| :--- | :---: | :--- |
| **SROCC / KROCC** | $\to 1.0$ | How well the metric ranks relative distortion levels (order accuracy). |
| **PLCC** | $\to 1.0$ | Linear agreement with human perceived distance after calibration. |
| **RMSE (JOD)** | $\to 0.0$ | **Average prediction error in JOD units**: <br>• $\mathbf{< 0.5\text{ JOD}}$: Exceptional accuracy (errors are sub-threshold and imperceptible).<br>• $\mathbf{0.5 - 1.0\text{ JOD}}$: Good accuracy (errors within 1 noticeable step).<br>• $\mathbf{> 1.5\text{ JOD}}$: Poor accuracy (frequent perceptual mismatches). |

---

## 6. Granular Breakdown by Distortion Type

To diagnose a metric's specific strengths and weaknesses across RAID's transformations, compute the calibration per distortion category:

```python
# Load RAID JOD dataset
df = pd.read_csv('raid_jod.csv')

# Benchmark per distortion type
for distortion in ['rotation', 'translation', 'scaling', 'additive_gaussian_noise']:
    sub = df[df['distortion_type'] == distortion]
    print(f"\n--- Results for: {distortion.upper()} ---")
    
    # Replace sub['raw_distance'] with your model's predictions
    # res = evaluate_metric_jod(sub['raw_distance'], sub['jod_distortion'])
```

This reveals whether a metric succeeds on traditional Gaussian noise while failing on affine transformations like rotation or translation.

---

## 7. Relevant References

1. **ITU-R BT.500-14 (2019).** *Methodologies for the subjective assessment of the quality of television pictures.* Recommendation ITU-R.
2. **VQEG (2000).** *Final report from the Video Quality Experts Group on the validation of objective models of video quality assessment.*
3. **Mantiuk, R. K., et al. (2018).** *A Practical Guide and Software for Analysing Pairwise Comparison Experiments.* **ACM Transactions on Applied Perception (TAP)**.
4. **Daudén-Oliver, P. et al. (2025).** *RAID-dataset: Human Responses to Affine Image Distortions and Gaussian noise.* Scientific Data (under review).
