#!/usr/bin/env python3
"""
Usage: func-argmax.py  <week>  <function_no>

This script uses scipy-minimize to suggest the next optimum point given week and function_no.
"""

import sys

if not len(sys.argv) >= 3:
  print(__doc__)
  sys.exit()
    
week_no = int(sys.argv[1])
func_no = int(sys.argv[2])

print("week_no:", week_no) 
print("func_no:", func_no)

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 1. Load data
np.random.seed(42)

from extra_points import *

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)

# 2. Fit 3D Gaussian Process Model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(data, output)

# 3. Define 3D Expected Improvement (EI) Function
def negative_ei(x, gp, y_max, xi=0.01):
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    
    improvement = mu - y_max - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    return -ei[0]  # Scipy minimizes, so return negative EI

# 4. Find the global maximum of EI in {data.shape[1]}D space
y_max = np.max(output)
bounds = [(0.0, 1.0)] * data.shape[1]

# Start from multiple random points to avoid local minima in optimization
best_ei = float('inf')
best_next_point = None

for start_point in np.random.uniform(0.0, 1.0, size=(data.shape[1]*10, data.shape[1])):
    res = minimize(
        negative_ei, start_point, 
        args=(gp, y_max, 0.01), 
        method='L-BFGS-B', 
        bounds=bounds
    )
    if res.fun < best_ei:
        best_ei = res.fun
        best_next_point = res.x

print()

print("scipy.optimize.minimize - GaussianProgressRegression EI / xi=0.01")
print()
print(f"Expected Improvement: {-best_ei:.6f}")

next_point = best_next_point

print()

print("-----")
print("The suggested next point\n")

printNextPoint(next_point, week_no, func_no)
