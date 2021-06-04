import numpy as np
from .RPN_to_eq import RPN_to_eq
from scipy.optimize import fsolve
from sympy import lambdify, N
import torch
import torch.nn as nn
import torch.nn.functional as F
from .get_pareto import Point, ParetoSet
from .S_get_expr_complexity import get_expr_complexity
from . import test_points
import os
import warnings
warnings.filterwarnings("ignore")
is_cuda = torch.cuda.is_available()


def check_compositionality(pathdir,filename,model,express,mu,sigma,nu=10):
    data = np.loadtxt(pathdir+filename)

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
    model.eval()
    while z<nu and i<len(data[0:1000]):
        # Generate functions based on the discovered possible equation and check if they are right
        dt = test_points.get_test_point(obj,data[i][:-1])
        diff = abs(f(*fixed[i])-f(*dt))
        with torch.no_grad():
            if diff<1e-4:
                if is_cuda:
                    dt = torch.tensor(dt).float().cuda().view(1,len(dt))
                    dt = torch.cat((torch.tensor([np.zeros(len(dt[0]))]).float().cuda(),dt), 0)
                    error = torch.tensor(data[:,-1][i]).cuda()-model(dt)[1:]
                    error = error.cpu().detach().numpy()
                    list_z = np.append(list_z,np.log2(1+abs(error)*2**30))
                    z = np.sqrt(len(list_z))*(np.mean(list_z)-mu)/sigma
                else:
                    dt = torch.tensor(dt).float().view(1,len(dt))
                    dt = torch.cat((torch.tensor([np.zeros(len(dt[0]))]).float(),dt), 0)
                    error =torch.tensor(data[:,-1][i])-model(dt)[1:]
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


def do_compositionality(pathdir,filename,express):
    data = np.loadtxt(pathdir+filename)
    eq = RPN_to_eq(express)
    # Get the variables appearing in the equation
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    variables = []

    N_vars = len(data[0])-1
    for i in range(N_vars):
        variables = variables + [possible_vars[i]]

    symbols = variables
    f = lambdify(symbols, N(eq))

    new_data = f(*np.transpose(data[:,0:-1]))
    save_data = np.column_stack((new_data,data[:,-1]))

    try:
        os.mkdir("results/compositionality")
    except:
        pass

    file_name = filename + "-comp"
    np.savetxt("results/compositionality/"+file_name,save_data)

    return ("results/compositionality/", file_name)


def add_comp_on_pareto(PA1,PA,express):
    eq = RPN_to_eq(express)
    PA1 = np.array(PA1.get_pareto_points()).astype('str')
    for i in range(len(PA1)):
        exp1 = PA1[i][2]
        exp1 = exp1.replace("x0",eq)
        compl = get_expr_complexity(exp1)
        PA.add(Point(x=compl,y=float(PA1[i][1]),data=str(exp1)))

    return PA

