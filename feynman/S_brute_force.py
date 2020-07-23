# runs BF on data and saves the best RPN expressions in results.dat
# all the .dat files are created after I run this script
# the .scr are needed to run the fortran code

import csv
import os
import shutil
import subprocess
import sys
from subprocess import call

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .resources import _get_resource

# sep_type = 3 for add and 2 for mult and 1 for normal


def brute_force(pathdir, filename, BF_try_time, BF_ops_file_type, sep_type="*"):
    try_time = BF_try_time
    try_time_prefactor = BF_try_time
    file_type = BF_ops_file_type

    try:
        os.remove("results.dat")
        os.remove("brute_solutions.dat")
        os.remove("brute_constant.dat")
        os.remove("brute_formulas.dat")
    except:
        pass

    print("Trying to solve mysteries with brute force...")
    print("Trying to solve {}".format(pathdir+filename))

    shutil.copy2(pathdir+filename, "mystery.dat")

    data = "'{}' '{}' mystery.dat results.dat ".format(_get_resource(file_type),
                                                       _get_resource("arity2templates.txt"))

    with open("args.dat", 'w') as f:
        f.write(data)

    if sep_type == "*":
        try:
            subprocess.call(["feynman_sr2"], timeout=try_time)
        except:
            pass
    if sep_type == "+":
        try:
            subprocess.call(["feynman_sr3"], timeout=try_time)
        except:
            pass
