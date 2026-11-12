#!/usr/bin/env python3
"""
This is a module for loading dataset
"""

import numpy as np
import os

from class_config import *

def loadWeekData(path):
    text = open(path).read().strip()
    # Separate the records with commas and wrap everything in one outer list
    text = "[" + text.replace("]\n[", "],\n[") + "]"
    return eval(text, {"array": np.array})

data_inputs = loadWeekData(os.path.join(DATA_DIR, WEEKLY_INPUT)) 
data_outputs = np.array(loadWeekData(os.path.join(DATA_DIR, WEEKLY_OUTPUT)))   


def loadDataWithExtraPoints(week, func):
  """
  Because of submission is at the end of the week while the data loaded for
  calculation is at the begin of the week,
  so the submission and results of that week are not included.
 
  Therefore extra inputs/outputs start from week WEEK_START+1.
  """

  #week=week-13
  #func=func-1

  data = np.load(os.path.join(DATA_DIR, "function_%d/initial_inputs.npy" % (func) ))
  output = np.load(os.path.join(DATA_DIR, "function_%d/initial_outputs.npy" % (func) ))

  n_init = len(output)

  if week >= WEEK_START+1:
    for week_i in range(0, week-WEEK_START):

      data = np.append(data, [ data_inputs[week_i][func-1] ], axis=0)
      output = np.append(output, [data_outputs[week_i][func-1]])

  return data, output, n_init

def printNextPoint(next_point, week_no, func_no):

  next_point = [0.999999 if x==1.0 else x for x in next_point]

  formatted_output = [f"{x:.6f}" for x in next_point]
  print(formatted_output)
  print()
  print("To-submit-w%d-f%d:" % (week_no, func_no), '-'.join(formatted_output))
