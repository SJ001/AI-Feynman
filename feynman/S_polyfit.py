import numpy as np
import os
from .S_polyfit_utils import getBest
from .S_polyfit_utils import basis_vector
import itertools
import sys
import csv
import sympy
from sympy import symbols, Add, Mul, S, simplify
from scipy.linalg import fractional_matrix_power

def mk_sympy_function(coeffs, num_covariates, deg):
    generators = [basis_vector(num_covariates+1, i) for i in range(num_covariates+1)]
    powers = map(sum, itertools.combinations_with_replacement(generators, deg))

    coeffs = np.round(coeffs,2)

    xs = (S.One,) + symbols('z0:%d'%num_covariates)
    if len(coeffs)>1:
        return Add(*[coeff * Mul(*[x**deg for x, deg in zip(xs, power)])
                     for power, coeff in zip(powers, coeffs)])
    else:
        return coeffs[0]

def polyfit(maxdeg, filename):
    n_variables = np.loadtxt(filename, dtype='str').shape[1]-1
    variables = np.loadtxt(filename, usecols=(0,))
    means = [np.mean(variables)]

    for j in range(1,n_variables):
        v = np.loadtxt(filename, usecols=(j,))
        means = means + [np.mean(v)]
        variables = np.column_stack((variables,v))

    f_dependent = np.loadtxt(filename, usecols=(n_variables,))

    if n_variables>1:
        C_1_2 = fractional_matrix_power(np.cov(variables.T),-1/2)
        x = []
        z = []
        for ii in range(len(variables[0])):
            variables[:,ii] = variables[:,ii] - np.mean(variables[:,ii])
            x = x + ["x"+str(ii)]
            z = z + ["z"+str(ii)]

        if np.isnan(C_1_2).any()==False:
            variables = np.matmul(C_1_2,variables.T).T
            res = getBest(variables,f_dependent,maxdeg)
            parameters = res[0]
            params_error = res[1]
            deg = res[2]

            x = sympy.Matrix(x)
            M = sympy.Matrix(C_1_2)
            b = sympy.Matrix(means)
            M_x = M*(x-b)

            eq = mk_sympy_function(parameters,n_variables,deg)
            symb = sympy.Matrix(z)

            for i in range(len(symb)):
                eq = eq.subs(symb[i],M_x[i])

            eq = simplify(eq)

        else:
            res = getBest(variables,f_dependent,maxdeg)
            parameters = res[0]
            params_error = res[1]
            deg = res[2]

            eq = mk_sympy_function(parameters,n_variables,deg)
            for i in range(len(x)):
                eq = eq.subs(z[i],x[i])
            eq = simplify(eq)

    else:
        res = getBest(variables,f_dependent,maxdeg)
        parameters = res[0]
        params_error = res[1]
        deg = res[2]
        eq = mk_sympy_function(parameters,n_variables,deg)
        try:
            eq = eq.subs("z0","x0")
        except:
            pass

    return (eq, params_error)

