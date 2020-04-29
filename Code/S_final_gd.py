# Turns a mathematical expression (already RPN turned) to pytorch expression, trains the parameters, and returns the new error, complexity and the new symbolic expression

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.utils.data as utils
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import sympy

from sympy import *
from sympy.abc import x,y
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, lambdify, N

from S_get_number_DL import get_number_DL

# parameters: path to data, RPN expression (obtained from bf)
def final_gd(data_file, math_expr, lr = 1e-2, N_epochs = 5000):
    param_dict = {}
    unsnapped_param_dict = {'p':1}

    def unsnap_recur(expr, param_dict, unsnapped_param_dict):
        """Recursively transform each numerical value into a learnable parameter."""
        import sympy
        from sympy import Symbol
        if isinstance(expr, sympy.numbers.Float):
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

    # Load the actual data

    data = np.loadtxt(data_file)

    # Turn BF expression to pytorch expression
    eq = parse_expr(math_expr)
    # eq = parse_expr("cos(0.5*a)") # this is for test_sympy.txt
    # eq = parse_expr("cos(0.5*a)+sin(1.2*b)+6") # this is for test_sympy_2.txt
    eq = unsnap_recur(eq,param_dict,unsnapped_param_dict)
    

    N_vars = len(data[0])-1
    N_params = len(unsnapped_param_dict)

    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    variables = []
    params = []
    for i in range(N_vars):
        #variables = variables + ["x%s" %i]
        variables = variables + [possible_vars[i]]
    for i in range(N_params-1):
        params = params + ["p%s" %i]

    symbols = params + variables
    
    f = lambdify(symbols, N(eq), torch)

    # Set the trainable parameters in the expression

    trainable_parameters = []
    for i in unsnapped_param_dict:
        if i!="p":
            vars()[i] = torch.tensor(unsnapped_param_dict[i])
            vars()[i].requires_grad=True
            trainable_parameters = trainable_parameters + [vars()[i]]

    
    # Prepare the loaded data

    real_variables = []
    for i in range(len(data[0])-1):
        real_variables = real_variables + [torch.from_numpy(data[:,i]).float()]

    input = trainable_parameters + real_variables
    
    y = torch.from_numpy(data[:,-1]).float()
    

    for i in range(N_epochs):
        # this order is fixed i.e. first parameters
        yy = f(*input)
        loss = torch.mean((yy-y)**2)
        loss.backward()
        with torch.no_grad():
            for j in range(N_params-1):
                trainable_parameters[j] -= lr * trainable_parameters[j].grad
                trainable_parameters[j].grad.zero_()
                
    for i in range(N_epochs):
        # this order is fixed i.e. first parameters
        yy = f(*input)
        loss = torch.mean((yy-y)**2)
        loss.backward()
        with torch.no_grad():
            for j in range(N_params-1):
                trainable_parameters[j] -= lr/10 * trainable_parameters[j].grad
                trainable_parameters[j].grad.zero_()  
                
    # get the updated symbolic regression
    ii = -1
    complexity = 0
    for parm in unsnapped_param_dict:
        if ii == -1:
            ii = ii + 1
        else:
            eq = eq.subs(parm, trainable_parameters[ii])
            complexity = complexity + get_number_DL(trainable_parameters[ii].detach().numpy())
            n_variables = len(eq.free_symbols)
            n_operations = len(count_ops(eq,visual=True).free_symbols)
            if n_operations!=0 or n_variables!=0:
                complexity = complexity + (n_variables+n_operations)*np.log2((n_variables+n_operations))
            ii = ii+1
            
        
    error = torch.mean((f(*input)-y)**2).data.numpy()*1
    return error, complexity, eq


