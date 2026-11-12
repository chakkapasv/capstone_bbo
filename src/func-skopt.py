#!/usr/bin/env python3
"""
Usage: func-skopt.py  <week>  <function_no>  <acq_func>  <acq_param>  [n_restarts_optimizer]

acq_func             :  EI or LCB
acq_param            :  xi=<value> for EI or kappa=<value> for LCB
n_restarts_optimizer :  default = 30

Example: func-skopt.py 19 6 LCB kappa=2.0 130
"""
import sys

if not len(sys.argv) >= 3:
  print(__doc__)
  sys.exit()

week_no = int(sys.argv[1])
func_no = int(sys.argv[2])

print("week_no:", week_no)
print("func_no:", func_no)

acq_func = sys.argv[3]
acq_param = sys.argv[4]
acq_p = acq_param.split('=')
acq_param_val = float(acq_p[1])
acq_param_name = acq_p[0]

if (acq_func=='EI' and acq_param_name=='xi'):
    xi = acq_param_val
    kappa = None
elif (acq_func=='LCB' and acq_param_name=='kappa'):
    xi = None
    kappa = acq_param_val
else:
  print("acq_func=%s and acq_param_name=%s are not compatible!" % \
    (acq_func, acq_param_name))
  sys.exit()

if len(sys.argv) >= 6:
  n_restarts_optimizer = int(sys.argv[5])
else:
  n_restarts_optimizer = 30

import numpy as np

# Load dataset

np.random.seed(42)

from extra_points import *

data, output, n_init = loadDataWithExtraPoints(week=week_no, func=func_no)

print()
print("data.shape"+':', data.shape)
print("output.shape"+':', output.shape)
print("n_init"+':', n_init)

dimension = data.shape[1]

print()
print("Using skopt.gp_minimize")
print()
print(f"acq_func: {acq_func} / {acq_param_name}={acq_param_val}")
print()
print(f"n_restarts_optimizer: {n_restarts_optimizer}")
print()

# Run the global optimization

from skopt import gp_minimize
from skopt.space import Real

from optimizer_skopt import suggest_next_point

# To maximize y, pass the negative of past data to skopt
y_past_for_skopt = -np.array(output)

next_point = suggest_next_point(data, y_past_for_skopt, dimension,
  n_restarts_optimizer=n_restarts_optimizer, acq_func=acq_func, kappa=kappa, xi=xi )

print("-----")
print("The suggested next point\n")

printNextPoint(next_point, week_no, func_no)
