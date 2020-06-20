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
from S_snap import integerSnap
from S_snap import zeroSnap
from S_snap import rationalSnap
from S_get_symbolic_expr_error import get_symbolic_expr_error
from get_pareto import Point, ParetoSet

from sympy import preorder_traversal, count_ops
from sympy.abc import x,y
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, lambdify, N, simplify, powsimp, Rational, symbols, S, Float
import time
import re

from S_get_number_DL_snapped import get_number_DL_snapped

def intify(expr):
    floats = S(expr).atoms(Float)
    ints = [i for i in floats if int(i) == i]
    return expr.xreplace(dict(zip(ints, [int(i) for i in ints])))

# parameters: path to data, math (not RPN) expression 
def add_snap_expr_on_pareto_polyfit(pathdir, filename, math_expr, PA): 
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

#    # Get the numbers appearing in the expression
#    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
#    eq_numbers = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]
#
#    # Do zero snap one parameter at a time
#    zero_snapped_expr = []
#    for w in range(len(eq_numbers)):
#        try:
#            param_dict = {}
#            unsnapped_param_dict = {'p':1}
#            eq = unsnap_recur(expr,param_dict,unsnapped_param_dict)
#            new_numbers = zeroSnap(eq_numbers,w+1)
#            for kk in range(len(new_numbers)):
#                eq_numbers[new_numbers[kk][0]] = new_numbers[kk][1]
#            jj = 0
#            for parm in unsnapped_param_dict:
#                if parm!="p":
#                    eq = eq.subs(parm, eq_numbers[jj])
#                    jj = jj + 1
#            zero_snapped_expr = zero_snapped_expr + [eq]
#        except:
#            continue

    # Get the numbers appearing in the expression
    is_atomic_number = lambda expr:expr.is_Atom and expr.is_number
    eq_numbers = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]

    # Do integer snap one parameter at a time                                      
    integer_snapped_expr = []
    for w in range(len(eq_numbers)):
        try:
            param_dict = {}
            unsnapped_param_dict = {'p':1}
            eq = unsnap_recur(expr,param_dict,unsnapped_param_dict)
            del unsnapped_param_dict["p"]
            temp_unsnapped_param_dict = copy.deepcopy(unsnapped_param_dict)
            new_numbers = integerSnap(eq_numbers,w+1)
            new_numbers = {"p"+str(k): v for k, v in new_numbers.items()}
            temp_unsnapped_param_dict.update(new_numbers) 
            #for kk in range(len(new_numbers)):
            #    eq_numbers[new_numbers[kk][0]] = new_numbers[kk][1]
            new_eq = re.sub(r"(p\d*)",r"{\1}",str(eq))
            new_eq = new_eq.format_map(temp_unsnapped_param_dict)
            integer_snapped_expr = integer_snapped_expr + [parse_expr(new_eq)]
        except:
            continue

            # Get the numbers appearing in the expression                                                                                                            
    
    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    eq_numbers = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]

    # Do rational snap one parameter at a time
    rational_snapped_expr = []
    for w in range(len(eq_numbers)):
        try:
            param_dict = {}
            unsnapped_param_dict = {'p':1}
            eq = unsnap_recur(expr,param_dict,unsnapped_param_dict)
            del unsnapped_param_dict["p"]
            temp_unsnapped_param_dict = copy.deepcopy(unsnapped_param_dict)
            new_numbers = rationalSnap(eq_numbers,w+1)
            new_numbers = {"p"+str(k): v for k, v in new_numbers.items()}
            temp_unsnapped_param_dict.update(new_numbers)
            #for kk in range(len(new_numbers)):
            #    eq_numbers_snap[new_numbers[kk][0]] = new_numbers[kk][1][1:3]
            new_eq = re.sub(r"(p\d*)",r"{\1}",str(eq))
            new_eq = new_eq.format_map(temp_unsnapped_param_dict)
            rational_snapped_expr = rational_snapped_expr + [parse_expr(new_eq)]
        except:
            continue

    snapped_expr = np.append(integer_snapped_expr,rational_snapped_expr)
#    snapped_expr = np.append(snapped_expr,rational_snapped_expr)

    integer_snapped_expr = snapped_expr

    for i in range(len(snapped_expr)):
        try:
            # Calculate the error of the new, snapped expression
            snapped_error = get_symbolic_expr_error(input_data,str(snapped_expr[i]))
            # Calculate the complexity of the new, snapped expression
            expr = snapped_expr[i]
            for s in (expr.free_symbols):
                s = symbols(str(s), real = True)
            expr =  parse_expr(str(snapped_expr[i]),locals())
            expr = intify(expr)
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
        
        
        
            
