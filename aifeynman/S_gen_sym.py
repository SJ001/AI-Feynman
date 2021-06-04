import numpy as np
from .RPN_to_eq import RPN_to_eq
from scipy.optimize import fsolve
from sympy import lambdify, N
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .get_pareto import Point, ParetoSet
from .S_get_expr_complexity import get_expr_complexity
from . import test_points
import os
import warnings
warnings.filterwarnings("ignore")
is_cuda = torch.cuda.is_available()


# fix this to work with the other variables constant
def check_gen_sym(pathdir,filename,model,gen_sym_idx,express,mu,sigma,nu=10):
    gen_sym_idx = np.append(gen_sym_idx,-1)
    data_all = np.loadtxt(pathdir+filename)
    # Choose only the data to be separated
    data = np.loadtxt(pathdir+filename)[:,gen_sym_idx]
    # Turn the equation from RPN to normal mathematical expression
    eq = RPN_to_eq(express)
    
    # Get the variables appearing in the equation
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    variables = []
    N_vars = len(data[0])-1
    for i in range(N_vars):
        variables = variables + [possible_vars[i]]
    symbols = variables
    f = lambdify(symbols, N(eq))

    fixed = data[:,0:-1]
    length_fixed = len(fixed)

    bm = np.ones(len(data[0])-1,dtype=bool)
    obj = test_points.init_general_test_point(eq, data[:,:-1], data[:,-1], bm)

    list_z = np.array([])
    z = 0
    i = 0
    while z<nu and i<len(data[0:1000]):
        # Generate functions based on the discovered possible equation and check if they are right
        dt = test_points.get_test_point(obj,data[i][:-1])
        diff = abs(f(*fixed[i])-f(*dt))
        with torch.no_grad():
            if diff<1e-4:
                if is_cuda:
                    dt_ = data_all[i]
                    ii = 0
                    for k in gen_sym_idx[:-1]:
                        dt_[k]=dt[ii]
                        ii = ii + 1
                    dt = torch.tensor(dt_).float().cuda().view(1,len(dt_))
                    dt = torch.cat((torch.tensor([np.zeros(len(dt[0]))]).float().cuda(),dt), 0)
                    error = torch.tensor(data[:,-1][i]).cuda()-model(dt[:,:-1])[1:]
                    error = error.cpu().detach().numpy()
                    list_z = np.append(list_z,np.log2(1+abs(error)*2**30))
                    z = np.sqrt(len(list_z))*(np.mean(list_z)-mu)/sigma
                else:
                    dt_ = data_all[i]
                    ii = 0
                    for k in gen_sym_idx[:-1]:
                        dt_[k]=dt[ii]
                        ii = ii + 1
                    dt = torch.tensor(dt_).float().view(1,len(dt_))
                    dt = torch.cat((torch.tensor([np.zeros(len(dt[0]))]).float(),dt), 0)
                    error =torch.tensor(data[:,-1][i])-model(dt[:,:-1])[1:]
                    error = error.detach().numpy()
                    list_z = np.append(list_z,np.log2(1+abs(error)*2**30))
                    z = np.sqrt(len(list_z))*(np.mean(list_z)-mu)/sigma
                    
                i = i + 1
            else:
                i = i + 1

    
    if i==len(data[0:1000]) and np.mean(list_z)<mu:
        return (1,express,np.mean(list_z),np.std(list_z))
    else:
        return (0,express,100,100)


def do_gen_sym(pathdir, filename, gen_sym_idx,express):
    gen_sym_idx = np.append(gen_sym_idx,-1)
    data_all = np.loadtxt(pathdir+filename)

    # Choose only the data to be separated
    data = np.loadtxt(pathdir+filename)[:,gen_sym_idx]
    # Turn the equation from RPN to normal mathematical expression
    eq = RPN_to_eq(express)
    # Get the variables appearing in the equation
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    variables = []

    N_vars = len(data[0])-1
    for i in range(N_vars):
        variables = variables + [possible_vars[i]]

    symbols = variables
    f = lambdify(symbols, N(eq))

    ii = 0
    for k in gen_sym_idx[1:-1]:
        data_all = np.delete(data_all,k-ii,1)
        ii = ii + 1

    new_data = f(*np.transpose(data[:,0:-1]))
    data_all[:,gen_sym_idx[0]]=new_data
    #save_data = np.column_stack((new_data,data_all))
    save_data = data_all

    try:
        os.mkdir("results/gen_sym")
    except:
        pass

    file_name = filename + "-gen_sym"
    np.savetxt("results/gen_sym/"+file_name,save_data)

    return ("results/gen_sym/", file_name)

def add_gen_sym_on_pareto(PA1,PA, gen_sym_idx, express):
    # Turn the equation from RPN to normal mathematical expression
    possible_vars = ["x%s" %i for i in np.arange(0,100,1)]
    gen_sym_idx = np.array(gen_sym_idx)
    math_eq = RPN_to_eq(express)

    PA1 = np.array(PA1.get_pareto_points()).astype('str')
    for i in range(len(PA1)):
        exp1 = PA1[i][2]
        temp_list = copy.deepcopy(gen_sym_idx)
        bf_eq = math_eq
        
        while(len(temp_list)>1):
            for j in range(len(possible_vars)-len(temp_list),temp_list[-1]-len(temp_list)+1,-1):
                exp1 = exp1.replace(possible_vars[j],possible_vars[j+1])
            temp_list = np.delete(temp_list,-1)
        
        # replace variables in bf_eq
        arr_idx = np.flip(np.arange(0,len(gen_sym_idx),1), axis=0)
        actual_idx = np.flip(gen_sym_idx, axis=0)
        for k in range(len(gen_sym_idx)):
            bf_eq = bf_eq.replace(possible_vars[arr_idx[k]],possible_vars[actual_idx[k]])

        exp1 = exp1.replace(possible_vars[temp_list[0]],"(" + bf_eq + ")")
        compl = get_expr_complexity(exp1)
        PA.add(Point(x=compl,y=float(PA1[i][1]),data=str(exp1)))

    return PA
