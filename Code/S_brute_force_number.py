# runs BF on data and saves the best RPN expressions in results.dat
# all the .dat files are created after I run this script
# the .scr are needed to run the fortran code

import numpy as np
import os
import shutil
import subprocess
from subprocess import call
import sys
import csv
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def brute_force_number(pathdir,filename):
    try_time = 2
    file_type = "10ops.txt"

    try:
        os.remove("results.dat")
    except:
        pass
    
    subprocess.call(["./brute_force_oneFile_v1.scr", file_type, "%s" %try_time, pathdir+filename])

    return 1

