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

# sep_type = 3 for add and 2 for mult and 1 for normal
def brute_force(pathdir,filename,BF_try_time,BF_ops_file_type,sep_type="*"):
    try_time = BF_try_time
    try_time_prefactor = BF_try_time
    file_type = BF_ops_file_type
    try:
        os.remove("results.dat")
    except:
        pass
    if sep_type=="*":
        subprocess.call(["./brute_force_oneFile_v2.scr", file_type, "%s" %try_time, pathdir+filename])
        #subprocess.call(["./brute_force_oneFile_mdl_v3.scr", file_type, "%s" %try_time, pathdir+filename, "10", "0"])
    if sep_type=="+":
        subprocess.call(["./brute_force_oneFile_v3.scr", file_type, "%s" %try_time, pathdir+filename])
        #subprocess.call(["./brute_force_oneFile_mdl_v2.scr", file_type, "%s" %try_time, pathdir+filename, "10", "0"])
    return 1

