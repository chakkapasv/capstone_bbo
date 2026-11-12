#!/usr/bin/env python3
"""
Usage: func-hebo.py  <week>  <function_no>

This script runs HEBO algorithm for global maximum search given week and function_no.
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
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from optimizer_hebo import suggest_next_point_hebo

np.random.seed(42)

# Load data set
from extra_points import *

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)

print()
print("Using HEBO")
print()

dimension = data.shape[1]

# Run the global optimization

# To maximize y, pass the negative of the past data to hebo
y_past_for_hebo = -np.array(output)


next_point = suggest_next_point_hebo(data, y_past_for_hebo, dimension)
  
print("-----")
print("The suggested next point\n")

printNextPoint(next_point, week_no, func_no)

#print(f"\nDistance to its closest neighbor: {max_distance:.4f}")
