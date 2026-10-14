 Continuous Black-Box Optimization (BBO) Capstone

This repository documents a 10-week capstone campaign in continuous black-box optimization (BBO) for eight synthetic objective functions ranging from $2\text{D}$ to $8\text{D}$. The project explores the empirical transition from global Gaussian Process (GP) surrogate modeling to localized Trust-Region Bayesian Optimization (TuRBO) under strict query budget constraints ($N \le 49$).

---

## Executive Summary

* **Objective:** Maximize eight unknown continuous black-box functions $f: [0.000000, 0.999999]^d \to \mathbb{R}$ across varying input dimensionalities ($2\text{D}, 2\text{D}, 3\text{D}, 4\text{D}, 4\text{D}, 5\text{D}, 6\text{D}, 8\text{D}$) using initial dataset seeds and 9 sequential feedback query submissions.
* **Core Strategy:**
  * **Phase 1 (Weeks 1–8):** Global surrogate modeling via `scikit-learn.optimize` Gaussian Processes. Used Expected Improvement (EI) for lower dimensions ($2\text{D}$–$3\text{D}$) and Lower Confidence Bound (LCB) with dynamic $\kappa$ decay for higher dimensions ($4\text{D}$–$8\text{D}$).
  * **Phase 2 (Weeks 9–10):** High-dimensional functions ($\ge 4\text{D}$) transitioned to Trust-Region Bayesian Optimization (TuRBO). Assigned `failure_tolerance=2` for $4\text{D}$ functions (Func 4 & 5) and `failure_tolerance=3` for $5\text{D}+$ functions (Func 6, 7 & 8).
  * **Alternative Framework Trials:** Evaluated HEBO in Week 9 (blocked by PyTorch/GPyTorch environment version mismatches) and `tidyHEBO` in Week 10 (abandoned due to `divide-by-zero` runtime warnings during GP matrix inversion on clustered samples).
* **Key Highlights:** Achieved significant local convergence in Func 5 ($Y > 1400$), broke Func 4 out of deep negative valleys (reaching $+0.4124$), and reached global campaign maxima in Func 6 ($-0.5442$) and Func 8 ($9.8310$) on the final evaluation rounds.

---

## Optimization Results Summary

| Function | Dimension | Initial Samples | Evaluated Queries | Total Points | Initial Best $Y$ | Final Maximum $Y$ | Peak Iteration / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Func 1** | 2D | 10 | 9 | 19 | $7.71 \times 10^{-16}$ | **$7.71 \times 10^{-16}$** | Initial Pt 2 (Scale ill-conditioning) |
| **Func 2** | 2D | 10 | 9 | 19 | $0.5390$ | **$0.6736$** | Pt 11 (Steady EI convergence) |
| **Func 3** | 3D | 15 | 9 | 24 | $-0.0348$ | **$-0.0310$** | Pt 20 (Incremental local progress) |
| **Func 4** | 4D | 30 | 9 | 39 | $-4.0255$ | **$+0.4124$** | Pt 37 (TuRBO valley escape) |
| **Func 5** | 4D | 20 | 9 | 29 | $1088.86$ | **$1428.31$** | Pt 21 (High-yield local basin) |
| **Func 6** | 5D | 20 | 9 | 29 | $-0.7143$ | **$-0.5442$** | Pt 28 (Active upward trajectory) |
| **Func 7** | 6D | 30 | 9 | 39 | $1.3650$ | **$1.3650$** | Initial Pt 6 (Curse of dimensionality) |
| **Func 8** | 8D | 40 | 9 | 49 | $9.5985$ | **$9.8310$** | Pt 48 (Final round maximum) |

---

## Project Artifacts & Documentation

Detailed analytical documentation, dataset characteristics, and algorithmic mechanics are available in the repository files below:

* 📄 **[DATASHEET.md](./DATASHEET.md):** Complete dataset documentation covering motivation, composition, collection timelines, sampling biases, scale ill-conditioning, and preprocessing requirements.
* 📋 **[MODELCARD.md](./MODELCARD.md):** Comprehensive model card outlining the Multi-Phase Trust-Region Bayesian Optimization (MP-TRBO) framework, strategy evolutions across 10 weeks, software fallback decisions, underlying assumptions, and failure modes.

---

## Technical Stack & Dependencies

* **Language:** Python 3.9+
* **Primary Optimization Libraries:** `scikit-learn` (`sklearn.gaussian_process`), `scipy.optimize`, `numpy`, `pandas`, `matplotlib`
* **Experimental Modules Evaluated:** `TuRBO` (Trust-Region BO), `HEBO` / `tidyHEBO`

---

## Repository Structure

```text
├── DATASHEET.md          # Dataset documentation & domain characteristics
├── MODELCARD.md          # Model card for MP-TRBO strategy & benchmarking
├── README.md             # Project overview and executive summary
├── data/                 # Function dataset CSVs (func1-data.csv to func8-data.csv)
└── notebooks/            # Jupyter notebooks detailing weekly query generation logic
