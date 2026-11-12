"""
Local (trust-region) Bayesian optimisation for the last few weeks.

Idea
----
Fit a noise-aware GP to all the data, then look only inside a small,
fixed box around the current best point (the "centre"):

    box   = [centre - radius, centre + radius], clipped to [0, 1]
    score = posterior_mean + beta * posterior_sd      (UCB, maximisation)

beta = 0   -> pure exploitation (go where the model predicts the highest value)
beta = 0.5 -> mostly exploitation with a little curiosity (default)
beta = 2   -> exploratory (similar to your LCB with kappa=2)

Unlike TuRBO, the box size here can be manually fixed, not by a success/failure
history. That makes it predictable, which is what you want with 1-2
queries left.

Needs only numpy / scipy / scikit-learn (no torch).
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning


def _fit_gp(X, y_std, seed=0):
    d = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=[0.3] * d, length_scale_bounds=(1e-2, 20.0), nu=2.5)
              + WhiteKernel(1e-2, (1e-6, 0.5)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=10, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y_std)
    return gp


def suggest_next_point_local(X_past, y_past, radius=0.1, beta=0.5,
                             center="best_obs", log_y=False,
                             n_candidates=20000, min_dist=1e-3,
                             seed=42, verbose=False):
    """
    X_past  : (n, d) array in [0, 1]
    y_past  : (n,) RAW outputs to MAXIMISE (do not negate)
    radius  : half-width of the box around the centre (0.05-0.1 is a good choice now)
    beta    : weight on the posterior sd (0 = exploit, 0.5 = default, 2 = explore)
    center  : "best_obs"  -> the point with the highest observed y
              "best_mean" -> the observed point the GP thinks is truly best
                             (better for noisy functions such as 2 and 3,
                              where the top observation may be a lucky draw)
    log_y   : model log(y) instead of y (only if all y > 0, e.g. function 5)
    Returns (next_point_list, info_dict)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X_past, dtype=float)
    y = np.asarray(y_past, dtype=float).ravel()
    n, d = X.shape

    # --- transform and standardise y -----------------------------------
    if log_y:
        assert np.all(y > 0), "log_y=True needs all outputs > 0"
        yt = np.log(y)
    else:
        yt = y.copy()
    mu, sd = yt.mean(), yt.std() + 1e-12
    ys = (yt - mu) / sd

    gp = _fit_gp(X, ys, seed=seed)

    def to_y(m):                      # back to the original units
        v = m * sd + mu
        return np.exp(v) if log_y else v

    # --- choose the centre ---------------------------------------------
    if center == "best_mean":
        c_idx = int(np.argmax(gp.predict(X)))
    else:
        c_idx = int(np.argmax(y))
    x_c = X[c_idx]

    lb = np.clip(x_c - radius, 0.0, 1.0)
    ub = np.clip(x_c + radius, 0.0, 1.0)

    # --- candidate search inside the box -------------------------------
    sobol = qmc.Sobol(d, scramble=True, seed=seed)
    m_pow = int(np.ceil(np.log2(n_candidates)))
    cand = lb + (ub - lb) * sobol.random_base2(m_pow)
    # also try each corner of the box along every single axis
    for k in range(d):
        for v in (lb[k], ub[k]):
            p = x_c.copy(); p[k] = v; cand = np.vstack([cand, p])

    def score(Z):
        m, s = gp.predict(np.atleast_2d(Z), return_std=True)
        return m + beta * s

    sc = score(cand)
    top = cand[np.argsort(sc)[-10:]]          # polish the best 10 with L-BFGS-B
    best_x, best_s = None, -np.inf
    for x0 in top:
        r = minimize(lambda z: -score(z)[0], x0, method="L-BFGS-B",
                     bounds=list(zip(lb, ub)))
        if -r.fun > best_s:
            best_x, best_s = r.x, -r.fun

    # avoid re-submitting (almost) the same point
    if np.min(np.linalg.norm(X - best_x, axis=1)) < min_dist:
        order = np.argsort(sc)[::-1]
        for i in order:
            if np.min(np.linalg.norm(X - cand[i], axis=1)) >= min_dist:
                best_x = cand[i]; break

    best_x = np.clip(best_x, 0.0, 1.0)

    m_next, s_next = gp.predict(best_x[None, :], return_std=True)
    m_c, _ = gp.predict(x_c[None, :], return_std=True)
    info = dict(
        center=x_c, center_idx=c_idx, center_y=y[c_idx],
        lb=lb, ub=ub, kernel=str(gp.kernel_),
        pred_mean=float(to_y(m_next[0])),
        pred_sd=float(s_next[0] * sd) if not log_y else None,
        pred_at_center=float(to_y(m_c[0])),
        best_observed=float(y.max()),
        step=best_x - x_c,
    )

    if verbose:
        np.set_printoptions(precision=4, suppress=True)
        print(f"  GP kernel   : {info['kernel']}")
        print(f"  centre ({center}): row {c_idx}, y={y[c_idx]:.6g}, x={x_c}")
        print(f"  box lb      : {lb}")
        print(f"  box ub      : {ub}")
        print(f"  step        : {info['step']}")
        sd_txt = f" +/- {info['pred_sd']:.4g}" if info['pred_sd'] is not None else ""
        print(f"  predicted y : {info['pred_mean']:.6g}{sd_txt}   "
              f"(model at centre {info['pred_at_center']:.6g}, best observed {info['best_observed']:.6g})")

    return best_x.tolist(), info
