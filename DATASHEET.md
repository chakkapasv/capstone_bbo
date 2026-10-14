# Datasheet: Black-Box Optimization (BBO) Capstone Dataset

---

## 1. Motivation (Purpose & Task Supported)

* **Task & Purpose:** This dataset was created to support continuous black-box function maximization over bounded hypercubes $[0.000000, 0.999999]^d$. It serves as an empirical record for benchmarking sequential decision-making algorithms under strict evaluation budgets ($N \le 49$ total points).


* **Creators:** Developed by the student researcher as part of the Imperial College Executive Education Machine Learning Capstone Project.


* **Funding & Support:** Imperial College Executive Education Business School.


* **Problem Addressed:** Evaluates how surrogate-based optimization algorithms transition from global exploration to local trust-region exploitation when optimizing unknown non-linear functions across varying dimensions ($2\text{D}$ to $8\text{D}$).



---

## 2. Composition (Contents, Size, Format, and Gaps)

* **Content:** Consists of 8 separate CSV files (`func1-data.csv` through `func8-data.csv`), corresponding to 8 synthetic black-box objective functions. Each row contains $d$ input feature columns ($X_0, \dots, X_{d-1}$) and one continuous scalar output column ($Y$).


* **Size & Structure (Initial Seeds vs. Evaluated Queries):**
* **Func 1 (2D):** 19 instances (10 initial seed points, 9 evaluated queries). Output range: $[-0.00361, 7.71 \times 10^{-16}]$.


* **Func 2 (2D):** 19 instances (10 initial seed points, 9 evaluated queries). Output range: $[-0.1428, 0.6736]$.


* **Func 3 (3D):** 24 instances (15 initial seed points, 9 evaluated queries). Output range: $[-0.3989, -0.0310]$.


* **Func 4 (4D):** 39 instances (30 initial seed points, 9 evaluated queries). Output range: $[-35.9228, 0.4124]$.


* **Func 5 (4D):** 29 instances (20 initial seed points, 9 evaluated queries). Output range: $[0.1129, 1428.31]$.


* **Func 6 (5D):** 29 instances (20 initial seed points, 9 evaluated queries). Output range: $[-2.5712, -0.5442]$.


* **Func 7 (6D):** 39 instances (30 initial seed points, 9 evaluated queries). Output range: $[0.0027, 1.3650]$.


* **Func 8 (8D):** 49 instances (40 initial seed points, 9 evaluated queries). Output range: $[5.5922, 9.8310]$.




* **Format:** Comma-Separated Values (`.csv`) with 6-decimal precision for inputs and double-precision floats for outputs.


* **Data Gaps & Sampling Biases:**
* **High-Dimensional Volumetric Sparsity:** In $6\text{D}$–$8\text{D}$ space (Func 7 & 8), 39–49 points cover less than $0.01\%$ of the volumetric domain, leaving massive interior sub-volumes unsampled.


* **Local Clustering Bias:** Transitioning to TuRBO in Weeks 9–10 heavily clustered sampling in Func 5 around $X \approx [0.21, 0.87, 0.86, 0.92]$, leaving $95\%$ of the search space unexplored.


* **Scale Ill-Conditioning:** Func 1 features outputs near zero ($10^{-16}$ to $10^{-164}$), creating a numerical gap that prevents standard stationary Gaussian Process kernels from computing gradient signals.




* **Privacy & Demographics:** No human subjects, personal data, or offensive content are present; all data are synthetic black-box evaluations.



---

## 3. Collection Process (Generation, Strategy, and Time Frame)

* **Query Generation & Optimization Strategy:**
* **Phase 1 (Weeks 1–8):** Global surrogate modeling using `scikit-learn.optimize` Gaussian Processes. Lower dimensions ($2\text{D}$–$3\text{D}$) used Expected Improvement (EI). Higher dimensions ($4\text{D}$–$8\text{D}$) used Lower Confidence Bound (LCB) with dynamic $\kappa$ parameter decay.


* **Phase 2 (Weeks 9–10):** High-dimensional functions ($\ge 4\text{D}$) transitioned to Trust-Region Bayesian Optimization (TuRBO). Assigned `failure_tolerance=2` for 4D functions (Func 4 & 5) and `failure_tolerance=3` for 5D+ functions (Func 6, 7 & 8) to control bounding box contraction speed.


* **Alternative Trialed Strategies:** Evaluated HEBO in Week 9 (blocked by PyTorch/GPyTorch environment version mismatches) and `tidyHEBO` in Week 10 (abandoned due to `divide-by-zero` runtime warnings during GP covariance matrix inversion on clustered samples).


* **Collection Time Frame:** Queries were generated sequentially over 9 query submission rounds during a 10-week capstone evaluation period.


* **Validation Rules:** All queries strictly conform to the portal input format $x_1-x_2-\dots-x_n$ bounded within $[0.000000, 0.999999]$.



---

## 4. Preprocessing & Uses (Transformations, Intended vs. Inappropriate Uses)

* **Preprocessing & Target Transformations:**
* **Raw Data Preservation:** Submitted query coordinates and raw output feedback are preserved without modification.


* **Required Surrogate Transformations:** Target variables require scaling before fitting GP surrogates. Func 1 requires logarithmic target transformation $y' = \log(y - y_{\min} + \epsilon)$ to normalize outputs across extreme exponent scales. Func 5 requires standard $z$-score normalization due to output magnitudes exceeding $1000$.




* **Intended & Suitable Uses:**
* Benchmarking sequential decision-making under low evaluation budgets ($N \le 49$).


* Demonstrating trust-region Bayesian optimization mechanics in continuous non-convex search spaces.


* Analyzing surrogate model degradation and variance over-estimation in high dimensions.




* **Inappropriate & Out-of-Scope Uses:**
* Evaluating global convergence for non-stationary or discontinuous functions without output re-scaling.


* Training offline deep reinforcement learning agents or deep neural networks that require large dataset volumes ($N > 10^4$).



---

## 5. Distribution & Maintenance (Availability, Terms of Use, and Maintainer)

* **Dataset Availability & Location:** Publicly hosted on GitHub within the student's capstone repository portfolio (`/data/` folder in `.csv` format and raw `.npy` arrays).


* **Terms of Use & Licensing:** Distributed under the MIT Open Source License for open educational, research, and benchmarking purposes.


* **Maintenance & Version Control:** Maintained exclusively by the student project creator. The dataset is static following Round 10 completion, serving as a reproducible record of the optimization campaign.
