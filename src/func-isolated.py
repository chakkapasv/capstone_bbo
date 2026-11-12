#!/usr/bin/env python3
"""
Usage: func-isolated.py  <week>  <function_no>

This script searchs for the point that further most from other points in search space.

Instead of using GP as normal we started by looking for a centre of the largest void
and test for output, in case it might be a useful data point to cover part of unknown
function.
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
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

from extra_points import *

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)
print("n_init"+':', n_init)

print()

data_dim = data.shape[1]

# Fit a 1-NN model to quickly find the absolute closest point
nbrs = NearestNeighbors(n_neighbors=1).fit(data)

# 2. Define the objective function
# We want to MAXIMIZE the distance to the nearest neighbor.
# Scipy minimizes, so we return the negative distance.
def objective_function(x):
    # x is the candidate point shape (data_dim,)
    dist, _ = nbrs.kneighbors(x.reshape(1, -1))
    return -dist[0][0]  # Negative because we want to maximize this distance

# 3. Define the bounds for the full possible data space [0.0, 1.0] for all inputs_dim dimensions
#bounds = [(0.0, 0.9)] * data_dim
bounds = [(0.0, 1.0)] * data_dim

# 4. Run the global optimization
result = differential_evolution(objective_function, bounds, seed=42)

# --- Results ---
most_isolated_point = result.x
max_distance = -result.fun

print(f"Most Isolated Point in Full Space ({data_dim}D):")
print(f"\nDistance to its closest neighbor: {max_distance:.4f}")
print()

next_point = most_isolated_point 

print("-----")
print("The suggested next point\n")

printNextPoint(next_point, week_no, func_no)
