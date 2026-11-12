"""
TuRBO-1 (Eriksson et al., 2019) for a one-point-per-week black-box setting.

Fixes compared with the previous version
-----------------------------------------
1. The trust-region state persists between weeks. Each week runs as a
   fresh process, so the state is rebuilt by replaying the weekly results
   in submission order (success -> grow, failure -> shrink). No state file
   is needed and the result is the same every time for the same data.
2. failure_tolerance / success_tolerance are used as given. Before, they
   were overwritten in __post_init__ and never used anyway.
3. y is standardised before the GP fit (function 5 has values ~4000).
4. The GP lengthscale and noise have sensible bounds, and the per-dimension
   trust-region weights are clipped. This stops the box from covering the
   whole range in one dimension (the cause of function 7's jumps to
   x1 ~ 0.95).
5. torch is seeded, so the suggestion can be reproduced.
6. float64 throughout. Candidates that duplicate an earlier point are dropped.
7. Optional restart. True TuRBO restarts with a new random design when the
   region collapses. That makes no sense with 1-2 queries left, so by
   default the length is held at length_min instead.
"""

from dataclasses import dataclass
import math
import numpy as np
import torch
import gpytorch
from gpytorch.constraints import Interval
from torch.quasirandom import SobolEngine

torch.set_default_dtype(torch.float64)


# =====================================================================
# 1. Trust-region state
# =====================================================================

@dataclass
class TurboState:
    dim: int
    batch_size: int = 1
    length_init: float = 0.4
    length_min: float = 0.02
    length_max: float = 1.0
    failure_tolerance: int = None      # None -> paper default ceil(max(4, dim))
    success_tolerance: int = 2
    allow_restart: bool = False

    # running values (set while replaying history)
    length: float = None
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    restart_triggered: bool = False
    n_restarts: int = 0

    def __post_init__(self):
        if self.failure_tolerance is None:
            self.failure_tolerance = math.ceil(
                max(4.0 / self.batch_size, float(self.dim) / self.batch_size))
        if self.length is None:
            self.length = self.length_init


def update_state(state, y_next):
    """Standard TuRBO update after observing y_next (a scalar or a batch)."""
    y_next = float(np.max(y_next))
    if y_next > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:      # expand
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:    # shrink
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, y_next)

    if state.length < state.length_min:
        if state.allow_restart:
            state.restart_triggered = True
            state.n_restarts += 1
            state.length = state.length_init
            state.success_counter = state.failure_counter = 0
        else:
            state.length = state.length_min
    return state


def replay_state(y_past, n_init, dim, verbose=False, **state_kwargs):
    """
    Rebuild the state from history: the first n_init points are the initial
    design, and every later point is one TuRBO step taken in order.
    """
    y_past = np.asarray(y_past, dtype=float).ravel()
    state = TurboState(dim=dim, **state_kwargs)
    state.best_value = float(np.max(y_past[:n_init]))
    for k, y in enumerate(y_past[n_init:]):
        prev_best = state.best_value
        update_state(state, y)
        if verbose:
            tag = "SUCCESS" if y > prev_best + 1e-3 * abs(prev_best) else "fail"
            print(f"  step {k+1:2d}: y={y: .5g}  {tag:7s} -> length={state.length:.4f} "
                  f"(succ={state.success_counter}, fail={state.failure_counter})")
    return state


# =====================================================================
# 2. GP model
# =====================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        d = train_x.shape[1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=d,
                lengthscale_constraint=Interval(0.005, 4.0)),
            outputscale_constraint=Interval(0.05, 20.0))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def fit_gp(train_x, train_y, training_iter=200, lr=0.1, verbose=False):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=Interval(1e-6, 0.2))       # y is standardised
    model = ExactGPModel(train_x, train_y, likelihood)
    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()
    if verbose:
        ls = model.covar_module.base_kernel.lengthscale.detach().squeeze().numpy()
        print(f"  GP fit: loss={loss.item():.3f} noise={likelihood.noise.item():.4f} "
              f"lengthscales={np.round(ls, 3)}")
    model.eval(); likelihood.eval()
    return model, likelihood


# =====================================================================
# 3. Candidate generation (Thompson sampling inside the trust region)
# =====================================================================

def generate_candidate(state, model, X, Y, n_candidates=None,
                       weight_clip=(0.5, 2.0), min_dist=1e-3, verbose=False):
    dim = X.shape[-1]
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * dim))

    x_center = X[Y.argmax(), :].clone()

    # trust-region shape from the ARD lengthscales (geometric mean = 1), clipped
    w = model.covar_module.base_kernel.lengthscale.squeeze().detach().reshape(-1)
    w = w / w.mean()
    w = w / torch.prod(w.pow(1.0 / len(w)))
    w = w.clamp(*weight_clip)

    tr_lb = torch.clamp(x_center - w * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + w * state.length / 2.0, 0.0, 1.0)
    if verbose:
        print(f"  TR center={np.round(x_center.numpy(), 4)}")
        print(f"  TR lb    ={np.round(tr_lb.numpy(), 4)}")
        print(f"  TR ub    ={np.round(tr_ub.numpy(), 4)}")

    sobol = SobolEngine(dim, scramble=True, seed=int(torch.randint(0, 10**6, (1,))))
    pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_candidates).to(X.dtype)

    # perturb only a subset of dimensions (matters for dim > 20)
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),))] = True   # was dim-1 (off by one)

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # drop candidates that (almost) repeat an existing point
    keep = torch.cdist(X_cand, X).min(dim=1).values > min_dist
    if keep.sum() > 0:
        X_cand = X_cand[keep]

    with torch.no_grad(), gpytorch.settings.max_cholesky_size(10000):
        sample = model(X_cand).sample()
    return X_cand[torch.argmax(sample)]


# =====================================================================
# 4. Wrapper used by func-turbo.py
# =====================================================================

def suggest_next_point_turbo(X_past, y_past, dimension, n_init,
                             failure_tolerance=None, success_tolerance=2,
                             length_init=0.4, length_min=0.02, length_max=1.0,
                             allow_restart=False, weight_clip=(0.5, 2.0),
                             seed=42, verbose=False):
    """
    X_past : (n, d) array in [0, 1]
    y_past : (n,) RAW values to MAXIMISE (do not negate)
    n_init : number of initial-design points at the start of X_past/y_past;
             everything after them is treated as TuRBO steps, in order
    Returns (next_point_list, state)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_np = np.asarray(X_past, dtype=float)
    y_np = np.asarray(y_past, dtype=float).ravel()
    assert X_np.shape == (len(y_np), dimension)
    assert 0 < n_init <= len(y_np)

    if verbose:
        print("Replaying trust-region history:")
    state = replay_state(y_np, n_init, dimension, verbose=verbose,
                         failure_tolerance=failure_tolerance,
                         success_tolerance=success_tolerance,
                         length_init=length_init, length_min=length_min,
                         length_max=length_max, allow_restart=allow_restart)
    if verbose:
        print(f"  => length={state.length:.4f}, failure_tolerance={state.failure_tolerance}, "
              f"best={state.best_value:.5g}, restarts={state.n_restarts}")

    X = torch.tensor(X_np)
    y_std = (y_np - y_np.mean()) / (y_np.std() + 1e-12)
    Y = torch.tensor(y_std)

    model, _ = fit_gp(X, Y, verbose=verbose)
    x_next = generate_candidate(state, model, X, Y,
                                weight_clip=weight_clip, verbose=verbose)
    return x_next.numpy().tolist(), state
