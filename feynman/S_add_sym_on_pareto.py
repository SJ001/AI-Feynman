# Combines 2 pareto fromtier obtained from the separability test into a new one.

from .get_pareto import Point, ParetoSet
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from sympy import Symbol, lambdify, N
from .get_pareto import Point, ParetoSet
from .S_get_expr_complexity import get_expr_complexity

def add_sym_on_pareto(pathdir,filename,PA1,idx1,idx2,PA,sym_typ):
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    PA1 = np.array(PA1.get_pareto_points()).astype('str')
    for i in range(len(PA1)):
        exp1 = PA1[i][2]
        for j in range(len(possible_vars)-2,idx2-1,-1):
            exp1 = exp1.replace(possible_vars[j],possible_vars[j+1])
        exp1 = exp1.replace(possible_vars[idx1],"(" + possible_vars[idx1] + sym_typ + possible_vars[idx2] + ")")
        compl = get_expr_complexity(exp1)
        PA.add(Point(x=compl,y=float(PA1[i][1]),data=str(exp1)))

    return PA




