#!/usr/bin/env python
"""
Usage: dataset_csv.py  [--run]

This script uses to mix and convert existing datasets of the capstone's:
- initial datasets under ${DATA_DIR}/function_?/(initial_inputs.npy, initial_outputs.npy)
- weekly datasets under ${DATA_DIR}/(inputs.txt, outputs.txt)

to CSV files for each functions:

${DATA_DIR}/func?-data.csv

This CSV data files can be used for plotting with the script: plot-fdc.py

Configure file:

- class_config.py
"""

import sys
import csv
import os

if not '--run' in sys.argv:
  print(__doc__)
  sys.exit()

from extra_points import *
from class_config import *

data_outputs = np.array(loadWeekData(os.path.join(DATA_DIR, WEEKLY_OUTPUT)))

points = len(data_outputs)

print(points)


for i in range(8):

  print()
  print(f"i={i}")

  func_no = i+1

  data, output, n_init = loadDataWithExtraPoints(WEEK_START+points, func_no)

  mixed_data = []
  for i in range(len(data)):
   #data_new = data[i].append(output[i])

   data_new = np.append(data[i], output[i])

   mixed_data.append(data_new)
    

  header_l = []
  for j in range(len(data[0])):
    header_l.append(f"X{j}")
  header_l.append("Output")

  header_s = ','.join(header_l)

  fn = os.path.join(DATA_DIR, f'func{func_no}-data.csv')

  f = open(fn, 'w')
  f.write(header_s+"\n")
  for i in range(len(mixed_data)):
    s = str(mixed_data[i])
    s = s[1:-1]
    p = s.split()
    sout = ','.join(p)

    f.write(sout+"\n")

  f.close()
