#!/usr/bin/env python3
"""
Usage: func-dirichlet.py  <week>  <function_no>

This script find the most isolated point within the dataset from function_no of week. 

In large dimension data with limited number of initial points searching through the 
whole space might not be efficient way, so we start from finding the most isolated
point inside the dataset to check whether it is possible to help getting useful data.
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
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# 1. Load dataset
from extra_points import *

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)
print("n_init"+':', n_init)

print()
print("Most isolated point in dataset using Dirichlet distribution")

num_points = data.shape[0]
data_dim = data.shape[1]

# 2. Generate candidate points via Dirichlet distribution
num_candidates = 100000

# Dirichlet parameters: alpha controls where candidates cluster.
# alpha = 1.0 gives a uniform distribution across the convex hull space.
# alpha < 1.0 pushes points toward vertices; alpha > 1.0 pushes points toward the center.
weights = np.random.dirichlet(alpha=np.ones(num_points), size=num_candidates)

# Matrix multiplication creates the weighted combinations (100000, data_dim)
candidates = weights @ data

# 3. Fit a k-Nearest Neighbors model on original data to measure local density
k = 3  # Look at distance to 3 nearest original points
nbrs = NearestNeighbors(n_neighbors=k).fit(data)

# Measure distances from each candidate to its k nearest original points
distances, _ = nbrs.kneighbors(candidates)

# Use mean distance to k-NN as the "emptiness" metric (higher = less dense)
sparsity_scores = distances.mean(axis=1)

# 4. Find the candidate with maximum distance to its neighbors
best_idx = np.argmax(sparsity_scores)
least_dense_point = candidates[best_idx]

print(f"\nAverage distance to {k} nearest original points: {sparsity_scores[best_idx]:.4f}")
print(f"Distance to absolute nearest original point: {distances[best_idx, 0]:.4f}")
print()

next_point = least_dense_point

print("-----")
print("The suggested next point\n")

printNextPoint(next_point, week_no, func_no)
