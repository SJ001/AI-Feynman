# Calculates the error of a given symbolic expression applied to a dataset. The input should be a string of the mathematical expression

from get_pareto import Point, ParetoSet
from RPN_to_pytorch import RPN_to_pytorch
from RPN_to_eq import RPN_to_eq
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from sympy import Symbol, lambdify, N

def get_symbolic_expr_error(pathdir,filename,expr):
    data = np.loadtxt(pathdir+filename)
    N_vars = len(data[0])-1
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    variables = []
    for i in range(N_vars):
        variables = variables + [possible_vars[i]]
    eq = parse_expr(expr)
    f = lambdify(variables, N(eq))
    real_variables = []

    for i in range(len(data[0])-1):
        check_var = "x"+str(i)
        if check_var in np.array(variables).astype('str'):
            real_variables = real_variables + [data[:,i]]

    # use this to get rid of cases where the loss gets complex because of transformations of the output variable
    if isinstance(np.mean((f(*real_variables)-data[:,-1])**2), complex):
        return 1000000
    else:
        return np.sqrt(np.mean((f(*real_variables)-data[:,-1])**2))/np.sqrt(np.mean(data[:,-1]**2))
