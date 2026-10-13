# Model Card: Multi-Phase Trust-Region Bayesian Optimisation (MP-TRBO)

---

## 1. Overview

* **Model Name:** Multi-Phase Trust-Region Bayesian Optimisation (MP-TRBO)


* **Type:** Sequential Black-Box Optimisation (BBO) Framework / Surrogate-Based Optimisation


* **Version:** 2.1 (Capstone Final Phase)


* **Developer:** Student Researcher, Imperial College Executive Education Business School


* **License:** Open Access (Capstone Educational Portfolio Release)


* **Description:** MP-TRBO is a adaptive Bayesian optimisation strategy designed to maximise unknown continuous black-box functions $f: [0.000000, 0.999999]^d \to \mathbb{R}$ across varying input dimensionalities ($2\text{D}$ to $8\text{D}$) under strict evaluation budgets ($N \le 49$ total points). The system transitions dynamically from global Gaussian Process (GP) exploration to localised Trust-Region Bayesian Optimisation (TuRBO) in higher dimensions.



---

## 2. Intended Use

### Primary Tasks

* **Continuous Black-Box Optimisation:** Suitable for expensive, noisy, or non-linear objective functions where internal analytical formulas are unavailable.


* **Hyperparameter Tuning & Physical Experimentation:** Designed for low-budget experimental loops, such as tuning ML hyperparameters, material synthesis, or engineering prototype testing.



### Out-of-Scope & Inappropriate Use Cases

* **High-Budget / High-Throughput Optimisation:** Inappropriate for cheap-to-evaluate functions where direct Monte Carlo, evolutionary algorithms, or gradient-based methods are faster ($N > 10^4$).


* **Combinatorial / Discrete Spaces:** Not designed for discrete or graph-structured search spaces without custom continuous embedding.


* **High-Dimensional Unbounded Spaces:** Unsuitable for dimensions $d \gg 10$ without specialised high-dimensional projection or additive GP architectures.



---

## 3. Methodological Details & Evolutionary Strategy

The optimisation campaign evolved across 10 project weeks (spanning 9 query submission rounds):

```
+-----------------------------------------------------------------------------------+
|                            MP-TRBO OPTIMIsation PIPELINE                          |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Weeks 1-8: Global Surrogate Phase]                                              |
|  * Lower Dim (2D-3D):  GP + Expected Improvement (EI)                             |
|  * Higher Dim (4D-8D): GP + Lower Confidence Bound (LCB) with decay kappa         |
|                                                                                   |
|                                     ||                                            |
|                                     \/                                            |
|                                                                                   |
|  [Weeks 9-10: Trust-Region Phase]                                                 |
|  * Higher Dim (4D-8D): Switched to TuRBO (Trust-Region BO)                        |
|    - 4D (Func 4, 5): failure_tolerance = 2 (Tight regional contraction)           |
|    - 5D-8D (Func 6-8): failure_tolerance = 3 (Dimensional flexibility)            |
|                                                                                   |
|                                     ||                                            |
|                                     \/                                            |
|                                                                                   |
|  [Alternative Fallbacks Tested]                                                   |
|  * Week 9 HEBO: Blocked by PyTorch/GPyTorch environment dependency conflicts      |
|  * Week 10 tidyHEBO: Abandoned due to divide-by-zero GP matrix inversion errors   |
+-----------------------------------------------------------------------------------+

```

### Strategy Progression across Rounds

1. **Rounds 1–8 (Global Search):**
* **2D & 3D (Func 1–3):** Employed Matérn 5/2 Gaussian Process surrogates paired with **Expected Improvement (EI)**. Focused on exploratory sampling across $X_0, X_1, X_2$ before tightening acquisition bounds.


* **4D to 8D (Func 4–8):** Applied **Lower Confidence Bound (LCB)** with a decreasing $\kappa$ parameter ($\kappa_t$) to enforce exploration early and exploitation late.




2. **Rounds 9–10 (Local Trust-Region Shift):**
* Transitioned 4D+ functions to **Trust-Region Bayesian Optimisation (TuRBO)**. Global GP models suffer from over-smoothing in higher dimensions; TuRBO restricts candidate queries to a localised bounding hyper-rectangle that expands on successes and contracts on failures.


* **4D Configuration (Func 4 & 5):** Assigned `failure_tolerance=2` for rapid contraction around high-yielding local basins.


* **5D–8D Configuration (Func 6, 7, 8):** Assigned `failure_tolerance=3` to allow additional exploration steps before shrinking the trust-region volume.




3. **Alternative Framework Executions:**
* **HEBO (Week 9):** Evaluated Heteroscedastic Evolutionary Bayesian Optimisation to model non-stationary output variance, but execution was blocked by PyTorch/GPyTorch environment version conflicts.
* **tidyHEBO (Week 10):** Trialed in Round 10, but produced runtime `divide-by-zero` warnings during lengthscale covariance inversion when samples clustered closely, prompting a fallback to the stable TuRBO implementation.



---

## 4. Performance & Evaluation Results

### Evaluation Metric

* **Primary Metric:** Cumulative Maximum Output Score achieved per function ($\max_{t} Y_t$).


* **Benchmark Budget:** 9 feedback rounds appending 10 new evaluations to initial datasets.



| Objective Function | Dim | Initial Points | Evaluated Points | Total Points | Initial Best $Y$ | Final Best $Y$ | Performance Trend & Behavior |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Func 1** | 2D | 10 | 9 | 19 | $7.71 \times 10^{-16}$ | **$7.71 \times 10^{-16}$** | Stagnated due to extreme float scale ($10^{-164}$)

 |
| **Func 2** | 2D | 10 | 9 | 19 | $0.5390$ | **$0.6736$** | Steady gain under EI refinement

 |
| **Func 3** | 3D | 15 | 9 | 24 | $-0.0348$ | **$-0.0310$** | Incremental upward convergence

 |
| **Func 4** | 4D | 30 | 9 | 39 | $-4.0255$ | **$+0.4124$** | Major breakthrough via TuRBO (out of deep valley)

 |
| **Func 5** | 4D | 20 | 9 | 29 | $1088.86$ | **$1428.31$** | Strong local convergence inside high-yield basin

 |
| **Func 6** | 5D | 20 | 9 | 29 | $-0.7143$ | **$-0.5442$** | Actively climbing; achieved max on final round

 |
| **Func 7** | 6D | 30 | 9 | 39 | $1.3650$ | **$1.3650$** | Stagnated; early initial peak un-surpassed

 |
| **Func 8** | 8D | 40 | 9 | 49 | $9.5985$ | **$9.8310$** | Achieved global maximum on final iteration

 |

---

## 5. Key Assumptions, Constraints, and Failure Modes

### Underlying Assumptions

1. **Spatial Smoothness & Stationarity:** Assumes that function outputs vary smoothly across continuous space according to a stationary Matérn 5/2 kernel.


2. **Noise Homoscedasticity:** Assumes observation noise is uniform across the hypercube $[0, 1]^d$.



### Limitations & Failure Modes

* **Scale-Induced Collapse (Func 1 Failure Mode):** When function output magnitudes drop to near-zero scales ($10^{-16}$ to $10^{-164}$), standard stationary GP kernels fail to compute meaningful gradients. The surrogate treats variations as numerical zero, freezing search progress.


* **Volumetric Curse of Dimensionality (Func 7 Limitation):** In 6D+ spaces, 39 evaluations cover less than $0.01\%$ of the space. Global variance estimates remain artificially high, causing acquisition functions to query boundary regions rather than refining known peaks.


* **Numerical Covariance Instability:** Highly clustered sampling inside TuRBO trust regions can cause singular matrix inversion errors (`divide-by-zero`), requiring jitter regularisers (`alpha > 1e-6`).



---

## 6. Ethical Considerations & Governance

* **Transparency & Reproducibility:** By documenting algorithm choices, random seed states, failure modes (e.g., package dependency failures and divide-by-zero warnings), and hyperparameter settings, external researchers can audit, replicate, or improve the strategy.


* **Accountability in Autonomous Optimisation:** Demonstrates why automated optimisers must incorporate fallback mechanisms and safety bounds when running in real-world physical systems (e.g., chemical synthesis or manufacturing) to prevent run-away queries caused by numerical model breakdown.



---

## 7. Model Card Design Reflections

### How MP-TRBO Makes Decisions

MP-TRBO makes decisions by balancing probabilistic surrogate uncertainty against predicted expectation. In early rounds, decisions are driven globally by Gaussian Process uncertainty bounds. In later rounds, decisions are bounded within hyper-rectangular trust regions centered strictly around historical maximums, prioritizing localised gradient exploitation over global uncertainty.

### Strengths & Limitations

* **Strengths:** Rapid recovery from deep negative valleys (Func 4), strong performance in high-yielding local basins (Func 5), and robustness against dimensionality stagnation in 8D space via TuRBO.


* **Limitations:** Vulnerable to non-stationary scale variance (Func 1) and premature local convergence when failure tolerances are set too aggressively.



### Structure Assessment: Is Current Detail Sufficient?

The current structure—modeled directly after the **Google DeepMind Gemma model card format**—is fully sufficient for this project. It captures:

1. Architectural and strategy progression (Global GP $\to$ Local TuRBO).


2. Quantitative benchmark tables detailing initial vs. final outcomes.


3. Technical edge cases (HEBO dependencies and `tidyHEBO` zero-division errors).
4. Explicit failure mode breakdowns (Func 1 scale collapse and Func 7 curse of dimensionality).



Adding excessive further detail would create cognitive clutter without providing additional analytical value for portfolio review.
