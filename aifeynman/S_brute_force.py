# runs BF on data and saves the best RPN expressions in results.dat
# all the .dat files are created after I run this script
# the .scr are needed to run the fortran code

import csv
import os
import shutil
import subprocess
from tqdm import tqdm
from time import time
import sys
from subprocess import call

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .resources import _get_resource

# sep_type = 3 for add and 2 for mult and 1 for normal


def brute_force(pathdir, filename, BF_try_time, BF_ops_file_type, sep_type="*", sigma=10, band=0):
    try_time = BF_try_time
    try_time_prefactor = BF_try_time
    file_type = BF_ops_file_type

    try:
        os.remove("results.dat")
    except:
        pass

    try:
        os.remove("brute_solutions.dat")
    except:
        pass

    try:
        os.remove("brute_constant.dat")
    except:
        pass

    try:
        os.remove("brute_formulas.dat")
    except:
        pass

    print("Trying to solve mysteries with brute force...")
    print("Trying to solve {}".format(pathdir+filename))

    shutil.copy2(pathdir+filename, "mystery.dat")

    data = "'{}' '{}' mystery.dat results.dat {:f} {:f}".format(_get_resource(file_type),
                                                                _get_resource(
                                                                    "arity2templates.txt"),
                                                                sigma,
                                                                band)

    with open("args.dat", 'w') as f:
        f.write(data)

    if sep_type == "*":
        try:
            # this might be what i need for redirecting the stdout
            # https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
            # Reference:
            # https://fabianlee.org/2019/09/15/python-getting-live-output-from-subprocess-using-poll/
            # subprocess.call(["feynman_sr2"], timeout=try_time)

            with tqdm(total=try_time, desc="Running BF with operator *", position=2, leave=False) as bf_bar:
                start_time = time()
                proc = subprocess.Popen(["timeout", str(try_time), "feynman_sr_mdl_mult"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                while True:
                    bf_bar.update(n=time()-start_time)
                    output = proc.stdout.readline()
                    if proc.poll() is not None:
                        break
                    if output:
                        print(output.strip().decode())
            #subprocess.call(["feynman_sr_mdl_mult"], timeout=try_time)
        except Exception as e:
            print("Non-fatal error occurred while running brute force process:\n{}\nContinuing.".format(e))
    if sep_type == "+":
        try:
            with tqdm(total=try_time, desc="Running BF with operator +", position=2, leave=False) as bf_bar:
                start_time = time()
                proc = subprocess.Popen(["timeout", str(try_time), "feynman_sr_mdl_plus"], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                while True:
                    bf_bar.update(n=time() - start_time)
                    output = proc.stdout.readline()
                    if proc.poll() is not None:
                        break
                    if output:
                        print(output.strip().decode())
            # subprocess.call(["feynman_sr3"], timeout=try_time)
            #subprocess.call(["feynman_sr_mdl_plus"], timeout=try_time)
        except Exception as e:
            print("Non-fatal error occurred while running brute force process:\n{}\nContinuing.".format(e))
