#!/usr/bin/env python3
"""
Usage: func-turbo.py <week>  <function_no>  [failure_tolerance]

failure_tolerance is optional. If it is left out, the paper default
ceil(max(4, dim)) is used.
"""
import sys

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit()

week_no = int(sys.argv[1])
func_no = int(sys.argv[2])

failure_tolerance = int(sys.argv[3]) if len(sys.argv) >= 4 else None

print("week_no:", week_no)
print("func_no:", func_no)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

from extra_points import *
from optimizer_turbo import suggest_next_point_turbo

from class_config import WEEK_START

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)
#n_init = len(np.load("data/function_%d/initial_outputs.npy" % func_no))

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)
print("n_init"+',', n_init)

print()
print("Using TuRBO\n")
print("failure_tolerance:", failure_tolerance if failure_tolerance else "default")

# Guard: week N must include every result from week WEEK_START to week N-1.
expected = n_init + (week_no - WEEK_START)
assert len(output) == expected, (
    f"Expected {expected} points (initial {n_init} + {week_no-WEEK_START} weekly) "
    f"but loaded {len(output)}. Update data_inputs/data_outputs first!")

print()

# RAW outputs (TuRBO maximises)
next_point, state = suggest_next_point_turbo(
    data, np.array(output), data.shape[1], n_init=n_init,
    failure_tolerance=failure_tolerance, verbose=True)

print("-----")
print("The suggested next point\n")
printNextPoint(next_point, week_no, func_no)
