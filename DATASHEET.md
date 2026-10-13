# Datasheet: Black-Box Optimization (BBO) Capstone Dataset

This datasheet documents the cumulative dataset created across 9 rounds of iterative black-box optimization (BBO) feedback. It integrates empirical findings, methodology evolutions, and reflections developed during Stage 2 of the Imperial College Executive Education Capstone Project.

---

## 1. Motivation

* **Task & Purpose:** This dataset supports sequential black-box function maximization over continuous hypercubes $[0.000000, 0.999999]^d$. It was generated to benchmark surrogate-based optimization strategies—specifically Expected Improvement (EI), Lower Confidence Bound (LCB), and Trust-Region Bayesian Optimization (TuRBO)—under strict query budgets.


* **Creators:** Created by the student researcher as part of the Imperial College ML Black-Box Optimization Capstone Project.


* **Funding & Support:** Imperial College Executive Education Business School.


* **Gaps Filled:** Serves as an empirical benchmark for evaluating how surrogate models transition from global exploration to local trust-region exploitation across varying dimensions ($2\text{D}$ to $8\text{D}$).



---

## 2. Composition

* **Content:** The dataset consists of 8 separate CSV files (`func1-data.csv` through `func8-data.csv`), corresponding to 8 synthetic black-box objective functions. Each row represents a single function evaluation consisting of input coordinates ($X_0, \dots, X_{d-1}$) and the resulting scalar output score.


* **Instance Count & Structure (Initial vs. Evaluated):**
* **Func 1 (2D):** 19 instances (10 initial, 9 evaluated). Output range: $[-0.00361, 7.71 \times 10^{-16}]$.


* **Func 2 (2D):** 19 instances (10 initial, 9 evaluated). Output range: $[-0.1428, 0.6736]$.


* **Func 3 (3D):** 24 instances (15 initial, 9 evaluated). Output range: $[-0.3989, -0.0310]$.


* **Func 4 (4D):** 39 instances (30 initial, 9 evaluated). Output range: $[-35.9228, 0.4124]$.


* **Func 5 (4D):** 29 instances (20 initial, 9 evaluated). Output range: $[0.1129, 1428.31]$.


* **Func 6 (5D):** 29 instances (20 initial, 9 evaluated). Output range: $[-2.5712, -0.5442]$.


* **Func 7 (6D):** 39 instances (30 initial, 9 evaluated). Output range: $[0.0027, 1.3650]$.


* **Func 8 (8D):** 49 instances (40 initial, 9 evaluated). Output range: $[5.5922, 9.8310]$.




* **Format:** Comma-Separated Values (`.csv`) with 6-decimal floating-point precision for inputs and double-precision floats for outputs.


* **Data Gaps & Sampling Biases:**
* **Extreme Volumetric Sparsity:** In 6D–8D space (Func 7 & 8), 39–49 points cover less than $0.01\%$ of the volumetric domain, leaving large interior sub-volumes unsampled.


* **Local Clustering Bias:** The transition to TuRBO in Weeks 9–10 heavily clustered sampling in Func 5 around $X \approx [0.21, 0.87, 0.86, 0.92]$, leaving $95\%$ of the search domain unexplored.


* **Scale Ill-Conditioning:** Func 1 exhibits near-zero values ($10^{-16}$ to $10^{-164}$), creating a numerical gap that prevents standard stationary Gaussian Process (GP) kernels from discerning gradient signals.




* **Privacy & Confidentiality:** No human subjects or sensitive demographic data are present; all data are synthetic black-box evaluations.



---

## 3. Collection Process

* **Sampling Strategy & Phase Evolution:**
* **Phase 1 (Weeks 1–8):** Global surrogate modeling using `scikit-learn.optimize` Gaussian Processes. Lower dimensions (2D–3D) used Expected Improvement (EI). Higher dimensions (4D–8D) used Lower Confidence Bound (LCB) with dynamic $\kappa$ parameter tuning.


* **Phase 2 (Weeks 9–10):** High-dimensional functions ($\ge 4\text{D}$) transitioned to Trust-Region Bayesian Optimization (TuRBO). A failure tolerance of `failure_tolerance=2` was assigned to 4D functions (Func 4 & 5) and `failure_tolerance=3` to 5D+ functions (Func 6, 7 & 8) to control hyper-rectangle contraction speed.


* **Failed Iteration Trials:** Evaluated HEBO in Week 9 (blocked by PyTorch/GPyTorch dependency version mismatches) and `tidyHEBO` in Week 10 (abandoned due to `divide-by-zero` runtime warnings during GP covariance inversion on clustered samples).


* **Collection Time Frame:** Collected sequentially across 9 query submission rounds in the capstone portal.


* **Input Validation Constraints:** Inputs strictly adhere to the capstone formatting specification $x_1-x_2-\dots-x_n$ bounded within $[0.000000, 0.999999]$.



---

## 4. Preprocessing & Uses

* **Preprocessing & Transformations:**
* **Raw Input Preservation:** Raw submitted query coordinates and feedback outputs are preserved in full precision.


* **Transformations Required for Modeling:** Target variables require scaling before surrogate fitting. Func 1 requires logarithmic target transformation $y' = \log(y - y_{\min} + \epsilon)$ to normalize outputs. Func 5 requires standard $z$-score normalization due to output magnitudes exceeding $1000$.




* **Intended Uses:**
* Benchmarking sequential decision-making under low evaluation budgets.


* Demonstrating trust-region Bayesian optimization performance in non-convex black-box settings.


* Analyzing surrogate model degradation in high-dimensional spaces.




* **Inappropriate Uses:**
* Evaluating global convergence for non-stationary or discontinuous functions without re-scaling.


* Training deep neural networks or offline reinforcement learning agents that require large sample sizes ($N > 10^4$).



---

## 5. Distribution & Maintenance

* **Distribution & Repository:** Publicly available on GitHub as a portfolio artifact under the capstone repository submission. Raw cumulative data points are maintained in primary `.csv` and `.npy` formats.


* **Terms of Use & License:** Distributed under the MIT Open Source License for educational and research purposes.


* **Maintenance & Archiving:** Maintained solely by the project creator. The dataset is static following Round 10 completion, serving as a reproducible artifact of the optimization campaign.
