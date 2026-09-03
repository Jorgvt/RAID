# Theoretical & Practical Guide: MLDS, JOD Scaling, and Cross-Dataset Alignment

This document details:
1. The mathematical relationship between **Maximum Likelihood Difference Scaling (MLDS)** (used in the RAID dataset) and the **Just-Objectionable-Difference (JOD)** / **Just-Noticeable-Difference (JND)** scale (from pairwise comparison tools like [`pwcmp`](https://github.com/mantiuk/pwcmp)).
2. How to **interpret JOD values** in practical perceptual contexts.
3. Documentation for all generated CSV files aligning **RAID** and **TID2013**.

---

## 1. Context & Experimental Paradigms

### A. 2AFC Pairwise Comparisons (Thurstone's Case V / `pwcmp`)
In standard 2-Alternative Forced Choice (2AFC) experiments, observers compare two stimuli $A$ and $B$ directly:
> *"Which image has higher quality?"* or *"Which image is more distorted?"*

Under Thurstone’s Case V model, each stimulus evokes a latent psychological response $X_i \sim \mathcal{N}(q_i, \sigma_{\text{stim}}^2)$. The difference between two conditions is normally distributed with decision variance $\sigma_{\Delta}^2 = 2\sigma_{\text{stim}}^2$, yielding the choice probability:

$$P(A > B) = \Phi\left(\frac{q_A - q_B}{\sigma_{\Delta}}\right)$$

where $\Phi(\cdot)$ is the cumulative distribution function (CDF) of the standard normal distribution.

### B. Quadruplet Difference Scaling (MLDS / RAID Dataset)
In the **RAID** experiment (`answers.csv`), observers were presented with **two pairs of images** (a quadruplet: $(I_a, I_b)$ vs. $(I_c, I_d)$) and asked:
> *"Which pair exhibits the larger perceptual difference?"*

The underlying decision model (Maloney & Yang, 2003) assumes that perceived difference is given by $\Delta_1 = |\psi(a) - \psi(b)|$ and $\Delta_2 = |\psi(c) - \psi(d)|$, corrupted by Gaussian decision noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$:

$$P(\text{Pair 1} > \text{Pair 2}) = \Phi\left(\frac{|\psi(a) - \psi(b)| - |\psi(c) - \psi(d)|}{\sigma}\right)$$

> [!NOTE]
> Because `answers.csv` contains interval comparisons between pairs rather than direct 2AFC comparisons between individual stimuli, tools expecting 2AFC inputs (like `pwcmp`) cannot ingest `answers.csv` directly without modifying the underlying likelihood function to MLDS quadruplets.

---

## 2. Scale Indeterminacy & Signal Detection Normalization

In the MLDS probability equation:

$$P(\text{Pair 1} > \text{Pair 2}) = \Phi\left(\frac{\Delta_1 - \Delta_2}{\sigma}\right) = \Phi\left(\frac{k \cdot (\Delta_1 - \Delta_2)}{k \cdot \sigma}\right)$$

Scaling all $\psi$ values by a factor $k$ and $\sigma$ by $k$ leaves choice probabilities unchanged. The scale is therefore an **interval scale** identified up to an arbitrary positive multiplier and origin.

### Two Standard Parameterizations:
1. **Unit-Range Normalization (`Curve_Value` / $\psi$):**  
   Fix $\psi(x_{\min}) = 0$ and $\psi(x_{\max}) = 1$, and estimate the internal decision noise standard deviation $\sigma$ (`Sigmas`).
2. **Standard-Noise / Thurstonian Normalization (`Percept_scale`):**  
   Fix the decision noise to $\sigma = 1$, which transforms the scale into:

   $$\Psi(x) = \frac{\psi(x)}{\sigma} = \frac{\text{Curve\_Value}}{\text{Sigmas}}$$

In psychophysics and Signal Detection Theory (SDT), measuring perceptual distance along $\Psi$ expresses distances in units of **internal noise standard deviations** ($\sigma$), corresponding to the sensitivity index **$d'$ (or standard JND/JOD)**.

---

## 3. Mantiuk's JOD (75% Discrimination Convention)

In image quality literature and the `pwcmp` toolbox (Mantiuk et al., 2018), **$1\text{ JOD}$** is formally defined as:
> *The perceptual difference corresponding to a **75% discrimination/selection probability**.*

For a normal distribution:

$$P = \Phi(z) = 0.75 \implies z = \Phi^{-1}(0.75) \approx 0.67448975$$

Since a difference of $1\text{ JOD}$ corresponds to $z = \Phi^{-1}(0.75) \cdot \sigma$, the scaling factor between the standard-deviation scale (`Percept_scale`) and the 75%-criterion JOD scale is:

$$\text{Scale Factor} = \frac{1}{\Phi^{-1}(0.75)} = \frac{1}{0.67448975} \approx 1.4826022$$

### Final JOD Formula:

$$\text{JOD}(x) = \frac{\text{Percept\_scale}(x)}{\Phi^{-1}(0.75)} = \frac{\text{Curve\_Value}(x)}{\text{Sigmas} \cdot \Phi^{-1}(0.75)} \approx \text{Percept\_scale}(x) \times 1.4826022$$

---

## 4. 🧠 How to Interpret JOD Values

Unlike ordinal Mean Opinion Scores (MOS), **JOD is an interval scale with physical probabilistic meaning**:

### A. The Probabilistic Meaning of Differences ($\Delta \text{JOD}$)
The difference in JOD ($\Delta = |\text{JOD}_A - \text{JOD}_B|$) directly predicts the probability that an average human observer will detect a difference or prefer condition $A$ over $B$:

$$P(A > B) = \Phi\left(\Delta \cdot \Phi^{-1}(0.75)\right) = \Phi(\Delta \times 0.6745)$$

| $\Delta \text{JOD}$ | Detection / Preference Probability | Perceptual Meaning |
| :---: | :---: | :--- |
| **$0.0\text{ JOD}$** | **$50.0\%$** (chance) | Identical or completely imperceptible difference |
| **$0.5\text{ JOD}$** | **$63.2\%$** | Sub-threshold difference; subtle, unnoticed by most |
| **$1.0\text{ JOD}$** | **$75.0\%$** (threshold) | **1 Just-Noticeable Difference (JND)**; reliably detected by 3 out of 4 observers |
| **$2.0\text{ JOD}$** | **$91.1\%$** | Obvious difference; almost all observers notice the degradation |
| **$3.0\text{ JOD}$** | **$97.8\%$** | Large perceptual difference; strong degradation |
| **$4.0+\text{ JOD}$** | **$> 99.7\%$** | Severe degradation; obvious to virtually any viewer |

### B. Quality vs. Distortion Distance Conventions
Depending on the objective, JOD values are expressed in one of two directions (both anchored at $0$ for the pristine reference image):

1. **Distortion Distance Scale (`jod_distortion` $\ge 0$):**
   * Reference image $= 0.0\text{ JOD}$
   * Values increase positively with distortion intensity (e.g. $+1.5\text{ JOD}, +4.0\text{ JOD}$).
   * Ideal for perceptual distance metrics and MLDS difference scales.

2. **Perceptual Quality Scale (`jod_quality` $\le 0$):**
   * Reference image $= 0.0\text{ JOD}$
   * Values decrease into negative numbers as quality drops (e.g. $-1.5\text{ JOD}, -4.0\text{ JOD}$).
   * Standard convention in Cambridge UPIQ, `pwcmp`, and HDR/SDR quality benchmarks.

$$\text{jod\_distortion} = - \text{jod\_quality} = |\text{JOD}|$$

---

## 5. 📂 Repository CSV Files Reference

| File | Rows | Description | Key Columns |
| :--- | :---: | :--- | :--- |
| **[`perceptual_scales.csv`](file:///Users/jorgvt/Developer/RAID/perceptual_scales.csv)** | 960 | Primary RAID MLDS scales | `Reference`, `Distorted`, `Curve_Value`, `Sigmas`, `Percept_scale`, `Estimated_MOS`, `JOD` |
| **[`raid_jod.csv`](file:///Users/jorgvt/Developer/RAID/raid_jod.csv)** | 960 | Standalone RAID dataset with clean schema | `reference_file`, `distorted_file`, `distortion_type`, `distortion_level`, `mlds_curve_value`, `mlds_sigma`, `percept_scale`, `estimated_mos`, `jod_distortion`, `jod_quality` |
| **[`tid2013_jod.csv`](file:///Users/jorgvt/Developer/RAID/tid2013_jod.csv)** | 3,000 | Standalone TID2013 dataset scaled to JOD | `reference_file`, `distorted_file`, `distortion_type`, `distortion_level`, `mos`, `jod_quality`, `jod_distortion` |
| **[`raid_tid2013_aligned.csv`](file:///Users/jorgvt/Developer/RAID/raid_tid2013_aligned.csv)** | 3,960 | Unified dataset combining RAID & TID2013 | `dataset`, `reference_id`, `reference_file`, `distorted_file`, `distortion_type`, `distortion_level`, `raw_metric_type`, `raw_metric_value`, `mos`, `native_jod`, `aligned_jod_distortion`, `aligned_jod_quality` |
| **[`shared_gn_raid_tid2013.csv`](file:///Users/jorgvt/Developer/RAID/shared_gn_raid_tid2013.csv)** | 96 | Paired 96 shared Gaussian Noise conditions | `reference_id`, `raid_reference_file`, `raid_distorted_file`, `tid2013_test_file`, `raid_gn_level`, `tid2013_gn_level`, `raid_jod_distortion`, `tid2013_jod_distortion`, `tid2013_mos` |

---

## 6. Key References

1. **Maloney, L. T., & Yang, J. N. (2003).** *Maximum likelihood difference scaling.* **Journal of Vision**, 3(8):5, 573–585. [DOI: 10.1167/3.8.5](https://doi.org/10.1167/3.8.5)
2. **Knoblauch, K., & Maloney, L. T. (2008).** *MLDS: Maximum likelihood difference scaling in R.* **Journal of Statistical Software**, 25(2), 1–26. [DOI: 10.18637/jss.v025.i02](https://doi.org/10.18637/jss.v025.i02)
3. **Knoblauch, K., & Maloney, L. T. (2012).** *Modeling Psychophysical Data in R.* Springer.
4. **Mantiuk, R. K., et al. (2018).** *A Practical Guide and Software for Analysing Pairwise Comparison Experiments.* **ACM Transactions on Applied Perception (TAP)**. [arXiv:1712.03686](https://arxiv.org/abs/1712.03686)
5. **Mikhailiuk, A., Pérez-Ortiz, M., & Mantiuk, R. K. (2018).** *Psychometric scaling of TID2013 dataset.* **QoMEX 2018**. [DOI: 10.1109/QoMEX.2018.8463403](https://doi.org/10.1109/QoMEX.2018.8463403)
6. **Pérez-Ortiz, M., et al. (2020).** *From Pairwise Comparisons and Rating to a Unified Quality Scale.* **IEEE Transactions on Image Processing**, 29, 2557–2568.
7. **Daudén-Oliver, P. et al. (2025).** *RAID-dataset: Human Responses to Affine Image Distortions and Gaussian noise.* Scientific Data (under review). [GitHub](https://github.com/paudauo/BBDD_Affine_Transformations)
