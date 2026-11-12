#!/usr/bin/env python3
"""
Usage: run-week.py  <week>  [--dry-run] [---show-command]

--dry-run       : print what to run without actual running it
--show-command  : show what command and parameters to run

This script will search for information of what function to run together
with predefined parameters by looking up in ${WEEK_RUN_PARAMETERS} record.

The output is limited to the suggested point to submit.  For the full output
please use: 

- func-generic.py 

Configure file:

- class_config.py

"""

import sys
import csv

import subprocess

from class_config import *

fn = WEEKLY_RUN_PARAMETERS

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()

week_no = sys.argv[1]
#func_no = sys.argv[2]

is_dryrun = '--dry-run' in sys.argv
is_showcommand = '--show-command' in sys.argv

#len(sys.argv) > 2 and sys.argv[2]=='--dry-run'

# Load all running parameters

run_params = []
with open(fn, "r") as f:
  data = csv.reader(f)
  header = next(data)
  for row in data:
    run_params.append(row)

for row in run_params:
  if week_no==row[0]:
    if is_dryrun or is_showcommand:
      print(f"./func-{row[2]}.py {row[0]} {row[1]} {row[3]}")
    run_argv = row[3].strip().split(" ")
    run_command = ["python3", f"./func-{row[2]}.py", week_no, row[1]]+run_argv

    #print(run_command)
 
    if not is_dryrun:
      result = subprocess.run(run_command, capture_output=True, text=True, check=False)

      #print(result.stdout)

      lines = result.stdout.split("\n")

      for line in lines:
        if line[:9]=="To-submit":
          print(line)
          break

