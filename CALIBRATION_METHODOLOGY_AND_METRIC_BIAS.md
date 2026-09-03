# Methodological Guide: Human JOD Invariance vs. Metric Distortion Bias

This document addresses a critical methodological question in perceptual quality assessment:
> *"If human JOD values are absolute and comparable across datasets, why does the choice of calibration dataset influence the evaluation of an Image Quality Assessment (IQA) metric?"*

---

## 1. The Core Distinction: Human Scale vs. Metric Scale

### A. Human JOD is Absolute and Invariant
Human JOD (Just-Objectionable-Difference) values **are universal and directly comparable** across all datasets and distortion types:

$$\Delta \text{JOD} = 1.0 \iff P(\text{Human detection / preference}) = 75.0\%$$

* $1.0\text{ JOD}$ of Gaussian noise in **TID2013** $=$
* $1.0\text{ JOD}$ of image translation in **RAID** $=$
* $1.0\text{ JOD}$ of JPEG compression.

In all cases, $1.0\text{ JOD}$ corresponds to the exact physical limit of **1 Just Noticeable Difference (JND)** of human sensory noise ($\sigma$).

---

### B. AI Metrics Have Unequal Distortion Biases
In contrast, artificial perceptual metrics (e.g., **LPIPS**, **DISTS**, **SSIM**, **MSE**) do **not** naturally treat all distortion types equally:

```
               ┌────────────────────────────────────────────────────────────┐
               │                  HUMAN PERCEPTION (JOD)                    │
               │               (Fixed, Universal "Metal Ruler")             │
               └─────────────────────────────┬──────────────────────────────┘
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
      1.0 JOD of Gaussian Noise                           1.0 JOD of Translation (2px)
   (75% of humans detect noise)                         (75% of humans detect shift)
                   │                                                   │
                   ▼                                                   ▼
     Metric Raw Output: d = 0.10                         Metric Raw Output: d = 0.45
      (Metric sees it as SMALL)                           (Metric overreacts massively!)
```

* **On Gaussian Noise:** The metric's deep feature representations might shift slightly ($d = 0.10$), which correctly corresponds to human threshold ($1.0\text{ JOD}$).
* **On Translation (e.g., 2-pixel shift):** Because convolutional filters are sensitive to spatial alignment, the metric distance blows up to $d = 0.45$—even though human observers still only perceive it as a tiny $1.0\text{ JOD}$ difference!

---

## 2. What the Calibration Function $f(d)$ Does

The calibration function $f(d)$ is the **mathematical converter** (e.g., 4-parameter logistic mapping) that translates the metric's raw artificial distance $d$ into human JOD units:

$$\hat{\text{JOD}} = f(d)$$

The choice of calibration dataset determines **which converter dictionary is learned**:

---

### Scenario 1: Cross-Domain Calibration *(The Gold Standard)*
> **Calibrate on TID2013 / Gaussian Noise $\longrightarrow$ Test on RAID Affine Transformations**

1. **Step 1 (Fit on Noise):**  
   The converter learns: $\text{Raw score } d = 0.10 \implies 1.0\text{ JOD}$.
2. **Step 2 (Evaluate on Translation):**  
   When a 2-pixel shift occurs (human JOD $= 1.0$), the metric outputs $d = 0.45$.  
   The converter maps $0.45 \longrightarrow \mathbf{4.2\text{ JOD}}$.
3. **Scientific Finding:**  
   **The evaluation exposes the metric's flaw:** *"The metric is 4x over-sensitive to translation compared to human vision."*

---

### Scenario 2: In-Domain Calibration *(Masks Metric Flaws)*
> **Calibrate on RAID Translation $\longrightarrow$ Test on RAID Translation**

1. The converter learns specifically for translation: $\text{Raw score } d = 0.45 \implies 1.0\text{ JOD}$.
2. The predicted JOD matches ground truth ($1.0 = 1.0$), producing high $PLCC$ and low $RMSE$ on paper.
3. **The Problem:** You have **stretched the ruler specifically for translation** to compensate for the metric's bug. The metric itself still does not know that a 2-pixel shift is perceptually equal to a small amount of noise.

---

## 3. The "Metal Ruler vs. Rubber Band" Analogy

```
 Human Vision (JOD) :  |---- 1 JOD ----|---- 1 JOD ----|---- 1 JOD ----|  (Rigid Metal Ruler)
 
 Metric (Noise)     :  |--- d = 0.10 --|--- d = 0.10 --|--- d = 0.10 --|  (Normal scale)
 Metric (Shift/Rot) :  |----------------- d = 0.45 -------------------|  (Stretched Rubber Band)
```

* **Human JOD** is a rigid **metal meter stick** (invariant across all phenomena).
* **The Metric** is an **elastic rubber band** whose stretchiness changes depending on the distortion family.
* By calibrating the rubber band on a neutral standard (TID2013/Noise) and measuring translations, **we measure how warped the rubber band is**.

---

## 4. Methodological Guidelines for Research

| Research Objective | Recommended Calibration Set | Evaluation / Test Set | Rationale |
| :--- | :--- | :--- | :--- |
| **Invisibility Thresholds (JND testing)** | **TID2013** (or RAID Gaussian Noise) | **RAID Affine subsets** (`translation`, `rotation`, `scaling`) | Tests whether a general metric naturally predicts human visibility thresholds on unseen affine shifts. |
| **Affine Robustness Diagnosis** | **TID2013** | **RAID** | Quantifies metric over/under-sensitivity to geometric transformations relative to traditional distortions. |
| **Overall Dataset Benchmark** | **`raid_tid2013_aligned.csv`** (Out-of-content CV) | **`raid_tid2013_aligned.csv`** (Held-out scenes) | Standard competitive ranking across all 3,960 image pairs. |

---

## 5. Related Files & Resources

* [`MLDS_to_JOD.md`](file:///Users/jorgvt/Developer/RAID/MLDS_to_JOD.md): Theoretical foundation of MLDS, decision noise, and JOD derivation.
* [`BENCHMARKING_METRICS_WITH_JOD.md`](file:///Users/jorgvt/Developer/RAID/BENCHMARKING_METRICS_WITH_JOD.md): Practical guide and Python code for running 4-parameter logistic calibration.
* [`benchmark_metrics.py`](file:///Users/jorgvt/Developer/RAID/benchmark_metrics.py): Executable evaluation script.
* [`raid_tid2013_aligned.csv`](file:///Users/jorgvt/Developer/RAID/raid_tid2013_aligned.csv): Full aligned 3,960-condition dataset.
