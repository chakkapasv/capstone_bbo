#!/usr/bin/env python3
"""
Usage: func-local.py  <week>  <function_no>  [radius]  [beta]  [center]  [log]

  radius : half-width of the box around the centre        (default 0.1)
  beta   : weight on the uncertainty, mean + beta*sd       (default 0.5)
  center : best_obs | best_mean                           (default best_obs)
  log    : write "log" to model log(y) (function 5 only)

Examples
  ./func-local.py 23 4                       # radius 0.1, beta 0.5
  ./func-local.py 23 4 0.05 0                # tighter box, pure exploitation
  ./func-local.py 23 2 0.05 0.5 best_mean    # noisy function: start from the GP's best
  ./func-local.py 23 5 0.1 0.5 best_obs log  # function 5 modelled on a log scale
"""
import sys

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit()

week_no = int(sys.argv[1])
func_no = int(sys.argv[2])
radius  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
beta    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
center  = sys.argv[5]        if len(sys.argv) > 5 else "best_obs"
log_y   = len(sys.argv) > 6 and sys.argv[6].lower() == "log"

print("week_no:", week_no)
print("func_no:", func_no)
print()
print("Local region search")
print(f"radius: {radius}  beta: {beta}  "
      f"center: {center}  log_y: {log_y}")

import numpy as np

from class_config import *

from extra_points import *
from optimizer_local import suggest_next_point_local

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)

# Guard: week N must include every result from week WEEK_START to week N-1.
expected = n_init + (week_no - WEEK_START)
assert len(output) == expected, (
    f"Expected {expected} points (initial {n_init} + {week_no-WEEK_START} weekly) "
    f"but loaded {len(output)}. Update data_inputs/data_outputs first!")

print("data.shape:", data.shape)

next_point, info = suggest_next_point_local(
    data, np.array(output), radius=radius, beta=beta,
    center=center, log_y=log_y, verbose=True)

print("-----")
print("The suggested next point\n")
printNextPoint(next_point, week_no, func_no)
