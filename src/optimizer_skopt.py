import numpy as np
from skopt import gp_minimize
from skopt.space import Real

def suggest_next_point(X_past, y_past, dimension, n_restarts_optimizer=50, 
  acq_func="EI", xi=0.01, kappa=1.96):
    """
    X_past: 2D array-like (shape: n_samples, dimension)
    y_past: 1D array-like (shape: n_samples,). Note: skopt MINIMIZES by default.
            If the goal is to MAXIMIZE, pass `-y_past`.
    """
    # Define bounds [0, 1] for each dimension
    bounds = [Real(0.0, 1.0) for _ in range(dimension)]

    #x0=list(X_past)
    #print('x0:', x0)
   
    # We set n_calls equal to the existing data + 1.
    # By passing x0 and y0, skopt skips building a new random design
    # and immediately fits the GP to suggest the next single point.
    res = gp_minimize(
        func=lambda x: 0.0,  # Dummy function (not evaluated, since we only want 1 step)
        dimensions=bounds,
        x0=np.array(X_past).tolist(),       # FIX: Fully converts the 2D array to a list of lists
        y0=np.array(y_past).flatten().tolist(),  # FIX: Ensures y0 is a flat,
        #x0=list(X_past),
        #y0=list(y_past),
        n_calls=len(X_past) + 1,
        n_random_starts=0,
        n_restarts_optimizer=n_restarts_optimizer, # Restarts to avoid local minima
        acq_func=acq_func,                    # Expected Improvement
        xi=xi,
        kappa=kappa,
        #acq_func_kwargs=acq_func_kwargs,
        acq_optimizer="lbfgs",            # Uses L-BFGS-B under the hood
        noise="gaussian",                 # Handles data noise robustly
        random_state=42
    )

    # The next optimal point to submit
    return res.x_iters[-1]

