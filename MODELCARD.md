# Model Card: Multi-Phase Trust-Region Bayesian Optimization (MP-TRBO)

---

## 1. Overview (Name, Type, and Version)

* **Model Name:** Multi-Phase Trust-Region Bayesian Optimization (MP-TRBO)


* **Type:** Sequential Black-Box Optimization (BBO) Framework / Surrogate-Based Optimization


* **Version:** 2.1 (Capstone Final Phase)


* **Developer:** Student Researcher, Imperial College Executive Education Business School


* **License:** Open Access (Capstone Educational Portfolio Release)


* **Description:** MP-TRBO is an adaptive Bayesian optimization framework designed to maximize unknown continuous black-box functions $f: [0.000000, 0.999999]^d \to \mathbb{R}$ across varying input dimensionalities ($2\text{D}$ to $8\text{D}$) under strict evaluation budgets ($N \le 49$ total points). The system transitions dynamically from global Gaussian Process (GP) exploration to localized Trust-Region Bayesian Optimization (TuRBO) in higher dimensions.



---

## 2. Intended Use (Suitable Tasks & Cases to Avoid)

### Suitable Tasks & Applications

* **Continuous Black-Box Optimization:** Ideal for expensive, noisy, or non-linear objective functions where internal analytical formulas or gradients are unavailable.


* **Low-Budget Experimental Loops:** Suitable for physical experimentation or hyperparameter tuning with constrained evaluation budgets ($N < 100$), such as chemical formulation testing or automated machine learning (AutoML).



### Out-of-Scope & Use Cases to Avoid

* **High-Budget / High-Throughput Optimization:** Inappropriate for cheap-to-evaluate functions where direct Monte Carlo, evolutionary algorithms, or gradient-based methods are significantly faster ($N > 10^4$).


* **Discrete & Combinatorial Search Spaces:** Unsuitable for categorical, integer-only, or graph-structured search domains without custom continuous embedding representations.


* **High-Dimensional Unbounded Domains:** Unsuitable for dimensions $d \gg 10$ without specialized high-dimensional projection or additive GP architectures.



---

## 3. Optimization Strategy, Techniques, and Evolution Across Rounds

The optimization campaign evolved across 10 project weeks (spanning 9 query submission rounds):

### Strategy Progression

1. **Phase 1: Global Surrogate Search (Rounds 1–8 / Weeks 1–8)**
* **2D & 3D Functions (Func 1–3):** Employed Matérn 5/2 Gaussian Process surrogates paired with **Expected Improvement (EI)** as the acquisition function. Focused on exploratory sampling before tightening acquisition bounds for local refinement.


* **4D to 8D Functions (Func 4–8):** Applied **Lower Confidence Bound (LCB)** with a decreasing $\kappa$ parameter ($\kappa_t$) to enforce global exploration early and exploitation late.




2. **Phase 2: Local Trust-Region Transition (Rounds 9–10 / Weeks 9–10)**
* High-dimensional functions ($\ge 4\text{D}$) transitioned to **Trust-Region Bayesian Optimization (TuRBO)** to combat the curse of dimensionality and over-smoothing in global GP surrogates.


* **4D Configuration (Func 4 & 5):** Assigned `failure_tolerance=2` for aggressive trust-region contraction around high-yielding local basins.


* **5D–8D Configuration (Func 6, 7 & 8):** Assigned `failure_tolerance=3` to allow necessary dimensionality flexibility before shrinking the bounding hyper-rectangle.




3. **Alternative Framework Execution & Fallbacks**
* **HEBO (Week 9):** Evaluated Heteroscedastic Evolutionary Bayesian Optimization to model non-stationary output variance, but execution was blocked by PyTorch/GPyTorch environment version mismatches.
* **tidyHEBO (Week 10):** Trialed in Round 10, but produced runtime `divide-by-zero` warnings during lengthscale covariance inversion when samples clustered closely, prompting a fallback to the stable TuRBO implementation.



---

## 4. Performance Summary Across Functions & Metrics

### Evaluation Metrics

* **Primary Metric:** Cumulative Maximum Output Score achieved per function ($\max_{t} Y_t$).


* **Secondary Metrics:** Trajectory gain relative to initial dataset seed ($\Delta Y$) and final round convergence behavior.


* **Evaluation Budget:** 9 feedback rounds appending 10 new evaluations to initial datasets.



| Objective Function | Dim | Initial Points | Evaluated Points | Total Points | Initial Best $Y$ | Final Maximum $Y$ | Trajectory Gain ($\Delta Y$) | Peak Iteration & Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Func 1** | 2D | 10 | 9 | 19 | $7.71 \times 10^{-16}$ | **$7.71 \times 10^{-16}$** | $0.00$ | Initial Pt 2 (Stagnated / Float precision limit)

 |
| **Func 2** | 2D | 10 | 9 | 19 | $0.5390$ | **$0.6736$** | $+0.1346$ | Pt 11 (Steady EI convergence)

 |
| **Func 3** | 3D | 15 | 9 | 24 | $-0.0348$ | **$-0.0310$** | $+0.0038$ | Pt 20 (Incremental local progress)

 |
| **Func 4** | 4D | 30 | 9 | 39 | $-4.0255$ | **$+0.4124$** | $+4.4379$ | Pt 37 (TuRBO breakthrough out of deep valley)

 |
| **Func 5** | 4D | 20 | 9 | 29 | $1088.86$ | **$1428.31$** | $+339.45$ | Pt 21 (High-yield local basin cluster)

 |
| **Func 6** | 5D | 20 | 9 | 29 | $-0.7143$ | **$-0.5442$** | $+0.1701$ | Pt 28 (Active upward trajectory at run end)

 |
| **Func 7** | 6D | 30 | 9 | 39 | $1.3650$ | **$1.3650$** | $0.00$ | Initial Pt 6 (Stagnated / High-dim over-exploration)

 |
| **Func 8** | 8D | 40 | 9 | 49 | $9.5985$ | **$9.8310$** | $+0.2325$ | Pt 48 (Global maximum achieved on final round)

 |

---

## 5. Key Assumptions, Constraints, and Potential Failure Modes

### Underlying Assumptions

1. **Spatial Continuity & Smoothness:** Assumes objective function values vary smoothly across continuous hypercube space $[0, 1]^d$, accurately modeled by stationary Matérn 5/2 GP kernels.


2. **Homoscedastic Noise:** Assumes observation noise variance is constant across the entire search space.



### Constraints & Failure Modes

* **Scale-Induced Model Collapse (Func 1 Failure Mode):** When output magnitudes drop to near-zero scales ($10^{-16}$ to $10^{-164}$), stationary GP kernels fail to compute valid gradient signals, interpreting variations as zero and freezing search progress.


* **Volumetric Curse of Dimensionality (Func 7 Constraint):** In 6D+ spaces, $39$ evaluations cover less than $0.01\%$ of the volumetric space. Global GP variance remains artificially high, causing acquisition functions to query boundary regions rather than refining known peak basins.


* **Covariance Matrix Instability:** Dense local sample clustering inside TuRBO trust regions can cause singular matrix inversion errors (`divide-by-zero`), requiring explicit jitter regularizers (`alpha > 1e-6`).



---

## 6. Transparency, Reproducibility, and Real-World Adaptation

* **Supporting Reproducibility:** By documenting exact algorithm parameters (`failure_tolerance`, $\kappa$ decay, kernel selections), random state seeds, and package version constraints (PyTorch/GPyTorch conflicts), external researchers can reproduce identical sampling trajectories or audit model behavior.


* **Real-World Adaptation & Deployment:** Explaining failure modes (e.g., scale collapse and zero-division warnings) provides crucial guidance for adapting BBO frameworks to real-world industrial systems (e.g., automated chemical synthesis or materials discovery), where unmonitored model breakdowns could lead to unsafe or wasted physical experiments.
