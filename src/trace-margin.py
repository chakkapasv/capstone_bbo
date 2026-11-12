#!/usr/bin/env python3
"""
Usage: ./trace-margin.py  <week>  <function_no>  [explore_margin] [point-to-compare]

This script uses for calculating:
- delta : output value difference from max point
- dist  : Euclidean distance from max point
"""

import csv
import sys
import math
import os

from class_config import *

week_first = WEEK_START

if len(sys.argv) < 3:
  print(__doc__)
  sys.exit()

week_no = int(sys.argv[1])
func_no = int(sys.argv[2])


if len(sys.argv) > 3:
  explore_margin = float(sys.argv[3])
else:
  explore_margin = 0.15

def output_delta(x, y):
  #
  # assuming the last item in the list X and Y is output
  #
  # Z = [Z0, Z1, Z2, ..., Zn, Output]
  #

  delta = x[-1] - y[-1]

  return math.fabs(delta)

def euclidean_dist(x, y):
  #
  # Z = [Z0, Z1, Z2, ..., Zn]
  #

  sum_sqr = 0.0
  for i in range(len(x)):
    sum_sqr += (x[i]-y[i])*(x[i]-y[i])

  dist = math.sqrt(sum_sqr)

  #print("x[:-1]:", x[:-1])
  #print("y[:-1]:", y[:-1])

  #delta = x[-1] - y[-1]
  ##print("Delta = x[-1] - y[-1]:", x[-1], y[-1], delta)
  #delta = math.fabs(delta)

  return dist


idx = 0
max_output_idx = 0
max_output = -10e4
data_l = []

fn = os.path.join(DATA_DIR, f'func{func_no}-data.csv')

with open(fn, "r") as f:
    data = csv.reader(f)
    header = next(data)
    print(header)
    for row in data:
        float_row = [float(x) for x in row]
        output = float_row[-1]
        if output > max_output:
          max_output = output
          max_input = row[:-1]
          max_output_idx = idx
        
        print(idx+1, float_row)

        data_l.append(float_row)

        idx += 1

max_idx = idx

week_delta = max_idx - max_output_idx
week_count_as = week_no - week_delta + 1
if week_count_as >= week_first:
  as_week = str(week_count_as)
else:
  as_week = 'Initial data - row '+ str(max_output_idx+1)

print()
print(max_output_idx+1, max_input, max_output, '    # week-'+as_week)
print()

dist_l = []
delta_l = []

print()
print("row_i+1, data_l[row_i], '-->', delta, '-', dist")
print()

for i in range(len(data_l)):

  # dist - Euclidean distance from the optimum point
  # delta - output value difference from the optimum point

  dist = euclidean_dist(data_l[i][:-1], data_l[max_output_idx][:-1])
  delta = output_delta(data_l[i], data_l[max_output_idx])

  print(i+1, data_l[i], '-->', delta, '-', dist)

  data_l[i].append(i)      # Add reference row number as the last element

  dist_l.append(dist)
  delta_l.append(delta)

# Now sort by nearest distance

data_by_dist_l = [x for _, x in sorted(zip(dist_l, data_l))]

print()
print("Data sorted by distance from max point:")
print()

print("data_by_dist, '-->', dist_l[data_by_dist[-1]]")
print()

#print("data_by_dist_l:", data_by_dist_l)

for i in range(len(data_by_dist_l)):
  data_by_dist = data_by_dist_l[i] 
  print(data_by_dist[-1]+1, data_by_dist[:-1], '-->', dist_l[data_by_dist[-1]])

print()
print("Data sorted by output delta from max point:")
print()

# Now sort by nearest distance

print("data_by_delta, '-->', delta_l[data_by_delta[-1]]")
print()

data_by_delta_l = [x for _, x in sorted(zip(delta_l, data_l))]

for i in range(len(data_by_delta_l)):
  data_by_delta = data_by_delta_l[i] 
  print(data_by_delta[-1]+1, data_by_delta[:-1], '-->', delta_l[data_by_delta[-1]])

optimum_point = data_l[max_output_idx]

lower_bound_l = []
upper_bound_l = []
for i in range(len(optimum_point)-1):
  lower_bound_l.append(optimum_point[i] - explore_margin)
  upper_bound_l.append(optimum_point[i] + explore_margin)

# Cap upper and lower bounds
for i in range(len(optimum_point)-1):
  if lower_bound_l[i] < 0:
    lower_bound_l[i] = 0
  if upper_bound_l[i] > 1.0:
    upper_bound_l[i] = 1.0

print()
print("Set explore boundary (margin=%.3f) for each dimensions:" % explore_margin)
print()
print("optimum_point:", optimum_point[:-1])
print("lower_bound_l:", lower_bound_l)
print("upper_bound_l:", upper_bound_l)
print("boundary cross distance:", euclidean_dist(upper_bound_l, lower_bound_l) )

print()

print("These point(s) are already in the new search boundary:")
print()

for i in range(len(data_l)):
  point = data_by_delta_l[i][:-1] 
  row_i = data_by_delta_l[i][-1]
  in_boundary = True
  #if row_i == max_output_idx:
  #  print("XX:", point)
  for j in range(len(lower_bound_l)-1):
    #print(point[j])
    if point[j] < lower_bound_l[j] or point[j] > upper_bound_l[j]:
      in_boundary = False
      break 

  if in_boundary:
    print(row_i+1, point, '-->', delta_l[row_i], '-', dist_l[row_i])
 #for i in range(len(data_by_delta_l)):
 # data_by_delta = data_by_delta_l[i]
 # print(data_by_delta, '-->', delta_l[data_by_delta[-1]])
    

if len(sys.argv) > 4:
  comp_point_idx = int(sys.argv[4])-1

  comp_point = data_l[comp_point_idx][:-2]

  max_point = data_l[max_output_idx][:-2]


  diff_dist_l = []
  for j in range(len(comp_point)):
    diff_dist = math.fabs(comp_point[j] - max_point[j])
    diff_dist_l.append(diff_dist)

  print()
  print("Diff the boundary:")
  print()
  print("max (row=%d):" % (max_output_idx+1), max_point)
  print("comp (row=%d):" % (comp_point_idx+1), comp_point)
  print("diff_dist_l:", diff_dist_l)
  print("max(diff_dist_l):", max(diff_dist_l))
  print("distance: ", euclidean_dist(max_point, comp_point)) 
