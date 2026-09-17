# SSIM Translation Visibility Threshold Benchmark (TID2013 References)

This document presents the complete benchmark of **SSIM's response to horizontal translation** across all 25 reference scenes in the **TID2013** dataset, evaluating shifts $\Delta x \in [1, 100]$ pixels and calculating the metric's predicted **$1.0\text{ JOD}$ (1 JND)** visibility threshold.

---

## 1. Experimental Methodology

* **Stimuli:** All 25 pristine reference images from TID2013 (`I01.BMP` to `I25.BMP`, $512 \times 384$ pixels).
* **Transformation:** Pure horizontal translation to the right for integer displacements $\Delta x \in \{1, 2, \dots, 100\}\text{ px}$.
* **Artifact Isolation:** To evaluate true spatial alignment sensitivity without confounding edge-step discontinuities, metrics are computed on the valid overlapping interior sub-arrays:
  $$\text{SSIM}_{\text{eval}}(\Delta x) = \text{SSIM}(I_{\text{ref}}[:, :-\Delta x], I_{\text{ref}}[:, \Delta x:])$$
* **Psychometric Conversion:** Raw structural distances $d(\Delta x) = 1 - \text{SSIM}_{\text{eval}}(\Delta x)$ are mapped into human interval units ($\Delta\text{JOD}$) using the pre-calibrated neutral Gaussian noise baseline standard:
  $$\hat{\text{JOD}}(\Delta x) = -3.6108 + \frac{6.9256}{1 + \exp\left(-\frac{d(\Delta x) + 0.0299}{0.2722}\right)}$$
* **1 JND Threshold Criterion:** The visibility threshold $\Delta x^*$ is the displacement satisfying $\hat{\text{JOD}}(\Delta x^*) = 1.0\text{ JOD}$ (75% human discrimination probability).

---

## 2. Summary Statistics Across 25 Reference Scenes

| Metric Parameter | Mean $\pm$ Std | Median | Range [Min, Max] |
| :--- | :---: | :---: | :---: |
| **1.0 JOD Visibility Threshold ($\Delta x^*$)** | **$0.80 \pm 0.27\text{ px}$** | **$0.79\text{ px}$** | **$[0.45\text{ px}, 1.35\text{ px}]$** |
| **Predicted Distortion at $\Delta x = 1\text{ px}$** | **$1.377 \pm 0.403\text{ JOD}$** | $1.257\text{ JOD}$ | $[0.749\text{ JOD}, 2.215\text{ JOD}]$ |
| **Predicted Distortion at $\Delta x = 2\text{ px}$** | **$2.176 \pm 0.447\text{ JOD}$** | $2.184\text{ JOD}$ | $[1.428\text{ JOD}, 2.950\text{ JOD}]$ |
| **Predicted Distortion at $\Delta x = 5\text{ px}$** | **$2.569 \pm 0.334\text{ JOD}$** | $2.665\text{ JOD}$ | $[1.851\text{ JOD}, 3.048\text{ JOD}]$ |
| **Predicted Distortion at $\Delta x = 10\text{ px}$** | **$2.672 \pm 0.286\text{ JOD}$** | $2.786\text{ JOD}$ | $[2.061\text{ JOD}, 3.088\text{ JOD}]$ |
| **Predicted Distortion at $\Delta x = 50\text{ px}$** | **$2.892 \pm 0.179\text{ JOD}$** | $2.946\text{ JOD}$ | $[2.443\text{ JOD}, 3.120\text{ JOD}]$ |
| **Predicted Distortion at $\Delta x = 100\text{ px}$** | **$2.957 \pm 0.138\text{ JOD}$** | $2.993\text{ JOD}$ | $[2.563\text{ JOD}, 3.136\text{ JOD}]$ |

---

## 3. Per-Reference Visibility Thresholds

| Scene ID | Description / Dominant Texture | 1 JND Threshold ($\Delta x^*$) | SSIM at 1px | $\hat{\text{JOD}}$ at 1px | $\hat{\text{JOD}}$ at 2px | $\hat{\text{JOD}}$ at 10px |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `I01` | High-frequency city / trees | **$0.55\text{ px}$** | $0.6781$ | $1.823\text{ JOD}$ | $2.655\text{ JOD}$ | $2.852\text{ JOD}$ |
| `I02` | Building / geometric lines | **$0.67\text{ px}$** | $0.7475$ | $1.503\text{ JOD}$ | $2.234\text{ JOD}$ | $2.513\text{ JOD}$ |
| `I03` | Smooth sky / mountains | **$1.35\text{ px}$** | $0.8856$ | $0.749\text{ JOD}$ | $1.462\text{ JOD}$ | $2.172\text{ JOD}$ |
| `I04` | Lighthouse / water | **$0.83\text{ px}$** | $0.8052$ | $1.205\text{ JOD}$ | $2.017\text{ JOD}$ | $2.524\text{ JOD}$ |
| `I05` | Dense foliage / parrots | **$0.60\text{ px}$** | $0.7149$ | $1.659\text{ JOD}$ | $2.731\text{ JOD}$ | $3.088\text{ JOD}$ |
| `I06` | Boat / harbor | **$0.79\text{ px}$** | $0.7927$ | $1.272\text{ JOD}$ | $2.184\text{ JOD}$ | $2.826\text{ JOD}$ |
| `I07` | Smooth church dome | **$1.21\text{ px}$** | $0.8720$ | $0.829\text{ JOD}$ | $1.644\text{ JOD}$ | $2.553\text{ JOD}$ |
| `I08` | Fine rock / gravel texture | **$0.51\text{ px}$** | $0.6425$ | $1.970\text{ JOD}$ | $2.752\text{ JOD}$ | $3.039\text{ JOD}$ |
| `I09` | Sailboat / ocean | **$0.80\text{ px}$** | $0.7967$ | $1.251\text{ JOD}$ | $1.979\text{ JOD}$ | $2.522\text{ JOD}$ |
| `I10` | Airplane / runway | **$0.80\text{ px}$** | $0.7958$ | $1.256\text{ JOD}$ | $2.036\text{ JOD}$ | $2.569\text{ JOD}$ |
| `I11` | Flower / bee | **$0.66\text{ px}$** | $0.7456$ | $1.512\text{ JOD}$ | $2.398\text{ JOD}$ | $2.828\text{ JOD}$ |
| `I12` | Portrait / woman | **$0.98\text{ px}$** | $0.8382$ | $1.023\text{ JOD}$ | $1.820\text{ JOD}$ | $2.549\text{ JOD}$ |
| `I13` | Dense fabric texture | **$0.45\text{ px}$** | $0.5762$ | $2.215\text{ JOD}$ | $2.950\text{ JOD}$ | $3.076\text{ JOD}$ |
| `I14` | Fruit basket | **$0.69\text{ px}$** | $0.7593$ | $1.444\text{ JOD}$ | $2.436\text{ JOD}$ | $2.973\text{ JOD}$ |
| `I15` | Window / brick wall | **$0.80\text{ px}$** | $0.7956$ | $1.257\text{ JOD}$ | $2.019\text{ JOD}$ | $2.435\text{ JOD}$ |
| `I16` | Smooth landscape | **$1.13\text{ px}$** | $0.8586$ | $0.907\text{ JOD}$ | $1.619\text{ JOD}$ | $2.365\text{ JOD}$ |
| `I17` | Buildings / lake | **$0.98\text{ px}$** | $0.8378$ | $1.026\text{ JOD}$ | $1.860\text{ JOD}$ | $2.669\text{ JOD}$ |
| `I18` | Trees / forest | **$0.60\text{ px}$** | $0.7132$ | $1.666\text{ JOD}$ | $2.662\text{ JOD}$ | $3.000\text{ JOD}$ |
| `I19` | Architecture / statue | **$0.59\text{ px}$** | $0.7054$ | $1.702\text{ JOD}$ | $2.506\text{ JOD}$ | $2.838\text{ JOD}$ |
| `I20` | Smooth sky / sunset | **$1.28\text{ px}$** | $0.8717$ | $0.831\text{ JOD}$ | $1.428\text{ JOD}$ | $2.061\text{ JOD}$ |
| `I21` | Motorcycle | **$0.80\text{ px}$** | $0.7966$ | $1.252\text{ JOD}$ | $2.062\text{ JOD}$ | $2.629\text{ JOD}$ |
| `I22` | Flowers / leaves | **$0.57\text{ px}$** | $0.6955$ | $1.747\text{ JOD}$ | $2.571\text{ JOD}$ | $2.795\text{ JOD}$ |
| `I23` | Smooth color patches | **$1.28\text{ px}$** | $0.8738$ | $0.818\text{ JOD}$ | $1.459\text{ JOD}$ | $2.216\text{ JOD}$ |
| `I24` | Peppers | **$0.62\text{ px}$** | $0.7267$ | $1.603\text{ JOD}$ | $2.575\text{ JOD}$ | $2.917\text{ JOD}$ |
| `I25` | Tools / machinery | **$0.53\text{ px}$** | $0.6610$ | $1.895\text{ JOD}$ | $2.336\text{ JOD}$ | $2.786\text{ JOD}$ |

---

## 4. Key Scientific Insights

1. **Sub-Pixel Sensitivity Across 72% of Scenes:**  
   For $18$ out of the $25$ reference images ($72\%$), the $1.0\text{ JOD}$ threshold is strictly **sub-pixel** ($\Delta x^* < 1.0\text{ px}$). The global median threshold is **$0.79\text{ pixels}$**.
2. **Spatial Frequency Dependency:**  
   * **High-frequency / fine textures** (`I13`, `I08`, `I25`, `I01`) trigger $1\text{ JND}$ at only $\mathbf{\Delta x^* \approx 0.45 - 0.55\text{ px}}$, with a 1-pixel shift perceived as $\mathbf{\approx 1.8 - 2.2\text{ JOD}}$ (strong distortion).
   * **Low-frequency / smooth scenes** (`I03`, `I20`, `I23`, `I07`, `I16`) have larger thresholds of $\mathbf{\Delta x^* \approx 1.1 - 1.35\text{ px}}$.
3. **Rapid Perceptual Saturation:**  
   Across all 25 scenes, SSIM rapidly saturates by $\Delta x = 5 - 10\text{ pixels}$ ($\hat{\text{JOD}} \approx 2.6 - 2.7\text{ JOD}$), with minimal gradient between $10\text{ px}$ and $100\text{ px}$.

---

## 5. Artifacts and Files

* **Full Sweep CSV (2,500 data points):** [`ssim_translation_sweep_all_references.csv`](file:///media/disk/users/vitojor/raid/ssim_translation_sweep_all_references.csv)
* **Threshold Summary CSV (25 scenes):** [`ssim_translation_thresholds_per_reference.csv`](file:///media/disk/users/vitojor/raid/ssim_translation_thresholds_per_reference.csv)
* **Sweep Script:** [`sweep_all_references_translation.py`](file:///media/disk/users/vitojor/raid/sweep_all_references_translation.py)
* **Multi-Panel Combined Figure:** [`Figures/ssim_translation_sweeps_all_references.png`](file:///media/disk/users/vitojor/raid/Figures/ssim_translation_sweeps_all_references.png)
