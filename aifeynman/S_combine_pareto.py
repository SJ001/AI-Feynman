# Combines 2 pareto fromtier obtained from the separability test into a new one.

from .get_pareto import Point, ParetoSet
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from sympy import Symbol, lambdify, N
from .get_pareto import Point, ParetoSet
from .S_get_expr_complexity import get_expr_complexity

def combine_pareto(input_data,PA1,PA2,idx_list_1,idx_list_2,PA,sep_type = "+"):
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    PA1 = np.array(PA1.get_pareto_points()).astype('str')
    PA2 = np.array(PA2.get_pareto_points()).astype('str')
    for i in range(len(PA1)):
        for j in range(len(PA2)):
            try:
                # replace the variables from the separated parts with the variables reflecting the new combined equation
                exp1 = PA1[i][2]
                exp2 = PA2[j][2]
                for k in range(len(idx_list_1)-1,-1,-1):
                    exp1 = exp1.replace(possible_vars[k],possible_vars[idx_list_1[k]])
                for k in range(len(idx_list_2)-1,-1,-1):
                    exp2 = exp2.replace(possible_vars[k],possible_vars[idx_list_2[k]])
                new_eq = "(" + exp1 + ")" + sep_type + "(" + exp2 + ")"
                compl = get_expr_complexity(new_eq)
                PA.add(Point(x=compl,y=get_symbolic_expr_error(input_data,new_eq),data=new_eq))
            except:
                continue
    return PA




