# Adds on the pareto all the snapped versions of a given expression (all paramters are snapped in the end)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.autograd import Variable
import copy
import warnings
warnings.filterwarnings("ignore")
import sympy
from .S_snap import integerSnap
from .S_snap import zeroSnap
from .S_snap import rationalSnap
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .get_pareto import Point, ParetoSet
from .S_brute_force_number import brute_force_number

from sympy import preorder_traversal, count_ops
from sympy.abc import x,y
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, lambdify, N, simplify, powsimp
from .RPN_to_eq import RPN_to_eq

from .S_get_number_DL_snapped import get_number_DL_snapped

# parameters: path to data, math (not RPN) expression
def add_bf_on_numbers_on_pareto(pathdir, filename, PA, math_expr):
    input_data = np.loadtxt(pathdir+filename)
    def unsnap_recur(expr, param_dict, unsnapped_param_dict):
        """Recursively transform each numerical value into a learnable parameter."""
        import sympy
        from sympy import Symbol
        if isinstance(expr, sympy.numbers.Float) or isinstance(expr, sympy.numbers.Integer) or isinstance(expr, sympy.numbers.Rational) or isinstance(expr, sympy.numbers.Pi):
            used_param_names = list(param_dict.keys()) + list(unsnapped_param_dict)
            unsnapped_param_name = get_next_available_key(used_param_names, "p", is_underscore=False)
            unsnapped_param_dict[unsnapped_param_name] = float(expr)
            unsnapped_expr = Symbol(unsnapped_param_name)
            return unsnapped_expr
        elif isinstance(expr, sympy.symbol.Symbol):
            return expr
        else:
            unsnapped_sub_expr_list = []
            for sub_expr in expr.args:
                unsnapped_sub_expr = unsnap_recur(sub_expr, param_dict, unsnapped_param_dict)
                unsnapped_sub_expr_list.append(unsnapped_sub_expr)
            return expr.func(*unsnapped_sub_expr_list)


    def get_next_available_key(iterable, key, midfix="", suffix="", is_underscore=True):
        """Get the next available key that does not collide with the keys in the dictionary."""
        if key + suffix not in iterable:
            return key + suffix
        else:
            i = 0
            underscore = "_" if is_underscore else ""
            while "{}{}{}{}{}".format(key, underscore, midfix, i, suffix) in iterable:
                i += 1
            new_key = "{}{}{}{}{}".format(key, underscore, midfix, i, suffix)
            return new_key

    eq = parse_expr(str(math_expr))
    expr = eq
    # Get the numbers appearing in the expression
    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    eq_numbers = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]
    # Do bf on one parameter at a time
    bf_on_numbers_expr = []
    for w in range(len(eq_numbers)):
        try:
            param_dict = {}
            unsnapped_param_dict = {'p':1}
            eq_ = unsnap_recur(expr,param_dict,unsnapped_param_dict)
            eq = eq_

            np.savetxt(pathdir+"number_for_bf_%s.txt" %w, [eq_numbers[w]])
            brute_force_number(pathdir,"number_for_bf_%s.txt" %w)
            # Load the predictions made by the bf code
            bf_numbers = np.loadtxt("results.dat",usecols=(1,),dtype="str")
            new_numbers = copy.deepcopy(eq_numbers)

            # replace the number under consideration by all the proposed bf numbers
            for kk in range(len(bf_numbers)):
                eq = eq_
                new_numbers[w] = parse_expr(RPN_to_eq(bf_numbers[kk]))

                jj = 0
                for parm in unsnapped_param_dict:
                    if parm!="p":
                        eq = eq.subs(parm, new_numbers[jj])
                        jj = jj + 1

                bf_on_numbers_expr = bf_on_numbers_expr + [eq]
        except:
            continue

    for i in range(len(bf_on_numbers_expr)):
        try:
            # Calculate the error of the new, snapped expression
            snapped_error = get_symbolic_expr_error(input_data,str(bf_on_numbers_expr[i]))
            # Calculate the complexity of the new, snapped expression
            expr = simplify(powsimp(bf_on_numbers_expr[i]))
            is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
            numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]

            snapped_complexity = 0
            for j in numbers_expr:
                snapped_complexity = snapped_complexity + get_number_DL_snapped(float(j))
            # Add the complexity due to symbols
            n_variables = len(expr.free_symbols)
            n_operations = len(count_ops(expr,visual=True).free_symbols)
            if n_operations!=0 or n_variables!=0:
                snapped_complexity = snapped_complexity + (n_variables+n_operations)*np.log2((n_variables+n_operations))

            PA.add(Point(x=snapped_complexity, y=snapped_error, data=str(expr)))
        except:
            continue

    return(PA)

