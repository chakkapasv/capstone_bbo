#!/usr/bin/env python3
"""
Usage: func-argmax.py  <week>  <function_no>

This script runs Gaussian Process with EI / xi=0.1 on function_no of week.
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
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

np.random.seed(42)

# 1. Load data
from extra_points import *

data, output, n_init  = loadDataWithExtraPoints(week=week_no, func=func_no)

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)
print("n_init"+':', n_init)

# 2. Fit Gaussian Process
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(data, output)

# 3. Define Expected Improvement (EI) Function
def expected_improvement(X, gp, y_max, xi=0.01):
    """
    Computes Expected Improvement at points X given a fitted GP model.
    xi: Exploration-exploitation trade-off parameter (e.g., 0.01 for moderate exploration)
    """

    mu, sigma = gp.predict(X, return_std=True)
    
    # Avoid divide-by-zero errors where uncertainty is zero
    sigma = np.maximum(sigma, 1e-9)
    
    improvement = mu - y_max - xi
    Z = improvement / sigma
    
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# 4. Create 2D Prediction Grid (100x100 resolution)
x1 = np.linspace(0.0, 1.0, 100)
x2 = np.linspace(0.0, 1.0, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])

# Compute EI across the grid
output_max = np.max(output)
ei_values = expected_improvement(X_grid, gp, output_max, xi=0.1)
Z_ei = ei_values.reshape(X1.shape)

# Find the point that maximizes Expected Improvement
next_point_idx = np.argmax(ei_values)
next_point = X_grid[next_point_idx]
max_ei_val = ei_values[next_point_idx]

# --- Print Recommended Coordinates ---
print()
print("GaussianProcessRegressor - EI / xi=0.1")
print()
print(f"Expected Improvement Value at Point: {max_ei_val:.6f}")
print()

print("-----")
print("The suggested next point\n")

printNextPoint(next_point, week_no, func_no)
