#!/usr/bin/env python3
"""
Usage: plot-fdc.py  <week>  <function_no>  [--save]

Left : output of every point, initial data (blue) then weekly submissions (green)
Right: |output - max output| vs Euclidean distance to the max point
       (a "fitness-distance" plot). Weekly points are labelled with their week.

This script plot data from the start week up until the specified week. Data includes
inputs and outputs up to the week. It should run after output of that week available,
not before.

It requires ${DATA_DIR}/func?-data.csv exist before running this script. In case of
missing run the below command to generate them:

- dataset_csv.py

Output:

- output images func?-delta-dist.png will be saved to the folder: ${OUTPUT}

Configure file:

- class_config.py
"""

"""
Changes from the previous version
- FIX: the delta is now computed from the output list. Before, line
  `data_l[i].append(i)` added the row index to the max row, so every row
  AFTER the max point was compared with that index instead of the max output
  (e.g. function 4 rows 39-41 showed 43.8 / 43.5 / 39.1 instead of 7.2 / 6.9 / 2.5).
- The x-axis shows real week numbers (12, 13, ...) instead of overlapping 0s.
- The title shows the fitness-distance correlation r (see below).
- Distance is also divided by sqrt(d), so functions of different dimension
  can be compared (0 = same point, 1 = opposite corners of the cube).
- --save writes func<N>-delta-dist.png instead of opening a window.

Fitness-distance correlation (Jones & Forrest, 1995):
  r = corr(distance to best point, |output gap to best point|)
  r close to 1   -> one smooth basin: the closer you are, the better (easy, exploit)
  r close to 0   -> distance tells you little: many peaks, or noise (explore)

"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from class_config import *

week_first = WEEK_START

if len(sys.argv) < 3:
    print(__doc__); sys.exit()

week_no = int(sys.argv[1])
func_no = int(sys.argv[2])
save = "--save" in sys.argv

df = pd.read_csv(os.path.join(DATA_DIR, f"func{func_no}-data.csv"))
X = df.iloc[:, :-1].to_numpy(dtype=float)
y = df.iloc[:, -1].to_numpy(dtype=float)
n, d = X.shape

n_weekly = week_no - week_first + 1          # weeks WEEK_START..week_no
n_init = n - n_weekly
assert n_init > 0, f"CSV has {n} rows but week {week_no} implies {n_weekly} weekly rows"

i_best = int(np.argmax(y))
delta = np.abs(y - y[i_best])                       # value gap to the best point
dist = np.linalg.norm(X - X[i_best], axis=1)        # Euclidean distance to the best point
dist_n = dist / np.sqrt(d)                          # normalised to [0, 1]

others = np.arange(n) != i_best
fdc = np.corrcoef(dist[others], delta[others])[0, 1]

best_when = (f"week {week_first + i_best - n_init}" if i_best >= n_init
             else f"initial row {i_best + 1}")
print(f"Function {func_no}: d={d}, n={n}, best={y[i_best]:.6g} ({best_when}), FDC r={fdc:.2f}")
print("No,Week,Output,Delta,Dist,DistNorm")
for i in range(n):
    wk = week_first + i - n_init if i >= n_init else "init"
    print(f"{i+1},{wk},{y[i]:.6g},{delta[i]:.6g},{dist[i]:.4f},{dist_n[i]:.4f}")

# ------------------------------------------------------------------ plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

x_init = np.arange(-n_init + 1, 1) + week_first - 1     # ... 9, 10, 11
x_week = np.arange(week_first, week_first + n_weekly)    # WEEK_START ... week_no
ax1.plot(x_init, y[:n_init], "o-", color="tab:blue", label="initial data")
ax1.plot(x_week, y[n_init:], "o-", color="tab:green", label="weekly submissions")
ax1.axhline(y[:n_init].max(), ls="--", color="grey", lw=1, label="best initial value")
ax1.axvline(week_first - 0.5, color="grey", lw=0.8)
ax1.set_title(f"Function {func_no}: outputs")
ax1.set_xlabel("Initial points (index) | weekly submission (week)")
ax1.set_ylabel("Output")
ax1.legend(fontsize=8)

ax2.scatter(dist_n[:n_init], delta[:n_init], color="tab:blue", label="initial data")
ax2.scatter(dist_n[n_init:], delta[n_init:], color="tab:red", label="weekly submissions")
for i in range(n_init, n):
    ax2.annotate(str(week_first + i - n_init), (dist_n[i], delta[i]),
                 textcoords="offset points", xytext=(4, 3), fontsize=8, color="tab:red")
ax2.set_title(f"Function {func_no}: gap vs distance to best   (FDC r = {fdc:.2f})")
ax2.set_xlabel("Distance from best point / sqrt(d)")
ax2.set_ylabel("|Output - best output|")
ax2.legend(fontsize=8)
fig.tight_layout()

if save:
    out = os.path.join(OUTPUT_DIR, f"func{func_no}-delta-dist.png")
    fig.savefig(out, dpi=120)
    print("saved", out)
else:
    plt.show()
