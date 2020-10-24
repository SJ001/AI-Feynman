# add a function to compte complexity

from .get_pareto import Point, ParetoSet
from .RPN_to_pytorch import RPN_to_pytorch
from .RPN_to_eq import RPN_to_eq
import numpy as np
import matplotlib.pyplot as plt
from .S_brute_force import brute_force
from .S_get_number_DL_snapped import get_number_DL_snapped
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops
from .S_polyfit import polyfit
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .S_add_sym_on_pareto import add_sym_on_pareto
from .S_add_snap_expr_on_pareto import add_snap_expr_on_pareto
import os
from os import path


def run_bf_polyfit(pathdir,pathdir_transformed,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3, output_type=""):
    input_data = np.loadtxt(pathdir_transformed+filename)
#############################################################################################################################
    if np.isnan(input_data).any()==False:
        # run BF on the data (+)
        print("Checking for brute force + \n")
        brute_force(pathdir_transformed,filename,BF_try_time,BF_ops_file_type,"+")

        try:
            # load the BF output data
            bf_all_output = np.loadtxt("results.dat", dtype="str")
            express = bf_all_output[:,2]
            prefactors = bf_all_output[:,1]
            prefactors = [str(i) for i in prefactors]

            # Calculate the complexity of the bf expression the same way as for gradient descent case
            complexity = []
            errors = []
            eqns = []
            for i in range(len(prefactors)):
                try:
                    if output_type=="":
                        eqn = prefactors[i] + "+" + RPN_to_eq(express[i])
                    elif output_type=="acos":
                        eqn = "cos(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="asin":
                        eqn = "sin(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="atan":
                        eqn = "tan(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="cos":
                        eqn = "acos(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="exp":
                        eqn = "log(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="inverse":
                        eqn = "1/(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="log":
                        eqn = "exp(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="sin":
                        eqn = "asin(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="sqrt":
                        eqn = "(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")**2"
                    elif output_type=="squared":
                        eqn = "sqrt(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="tan":
                        eqn = "atan(" + prefactors[i] + "+" + RPN_to_eq(express[i]) + ")"

                    eqns = eqns + [eqn]
                    errors = errors + [get_symbolic_expr_error(input_data,eqn)]
                    expr = parse_expr(eqn)
                    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
                    numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]
                    compl = 0
                    for j in numbers_expr:
                        try:
                            compl = compl + get_number_DL_snapped(float(j))
                        except:
                            compl = compl + 1000000

                    # Add the complexity due to symbols
                    n_variables = len(expr.free_symbols)
                    n_operations = len(count_ops(expr,visual=True).free_symbols)
                    if n_operations!=0 or n_variables!=0:
                        compl = compl + (n_variables+n_operations)*np.log2((n_variables+n_operations))

                    complexity = complexity + [compl]
                except:
                    continue

            for i in range(len(complexity)):
                PA.add(Point(x=complexity[i], y=errors[i], data=eqns[i]))

            # run gradient descent of BF output parameters and add the results to the Pareto plot
            for i in range(len(express)):
                try:
                    bf_gd_update = RPN_to_pytorch(input_data,eqns[i])
                    PA.add(Point(x=bf_gd_update[1],y=bf_gd_update[0],data=bf_gd_update[2]))
                except:
                    continue
        except:
            pass

    #############################################################################################################################
        # run BF on the data (*)
        print("Checking for brute force * \n")
        brute_force(pathdir_transformed,filename,BF_try_time,BF_ops_file_type,"*")

        try:
            # load the BF output data
            bf_all_output = np.loadtxt("results.dat", dtype="str")
            express = bf_all_output[:,2]
            prefactors = bf_all_output[:,1]
            prefactors = [str(i) for i in prefactors]

            # Calculate the complexity of the bf expression the same way as for gradient descent case
            complexity = []
            errors = []
            eqns = []
            for i in range(len(prefactors)):
                try:
                    if output_type=="":
                        eqn = prefactors[i] + "*" + RPN_to_eq(express[i])
                    elif output_type=="acos":
                        eqn = "cos(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="asin":
                        eqn = "sin(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="atan":
                        eqn = "tan(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="cos":
                        eqn = "acos(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="exp":
                        eqn = "log(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="inverse":
                        eqn = "1/(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="log":
                        eqn = "exp(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="sin":
                        eqn = "asin(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="sqrt":
                        eqn = "(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")**2"
                    elif output_type=="squared":
                        eqn = "sqrt(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"
                    elif output_type=="tan":
                        eqn = "atan(" + prefactors[i] + "*" + RPN_to_eq(express[i]) + ")"

                    eqns = eqns + [eqn]
                    errors = errors + [get_symbolic_expr_error(input_data,eqn)]
                    expr = parse_expr(eqn)
                    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
                    numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]
                    compl = 0
                    for j in numbers_expr:
                        try:
                            compl = compl + get_number_DL_snapped(float(j))
                        except:
                            compl = compl + 1000000

                    # Add the complexity due to symbols
                    n_variables = len(expr.free_symbols)
                    n_operations = len(count_ops(expr,visual=True).free_symbols)
                    if n_operations!=0 or n_variables!=0:
                        compl = compl + (n_variables+n_operations)*np.log2((n_variables+n_operations))

                    complexity = complexity + [compl]
                except:
                    continue

            # add the BF output to the Pareto plot
            for i in range(len(complexity)):
                PA.add(Point(x=complexity[i], y=errors[i], data=eqns[i]))

            # run gradient descent of BF output parameters and add the results to the Pareto plot
            for i in range(len(express)):
                try:
                    bf_gd_update = RPN_to_pytorch(input_data,eqns[i])
                    PA.add(Point(x=bf_gd_update[1],y=bf_gd_update[0],data=bf_gd_update[2]))
                except:
                    continue
        except:
            pass

    #############################################################################################################################
        # run polyfit on the data
        print("Checking polyfit \n")
        try:
            polyfit_result = polyfit(polyfit_deg, pathdir_transformed+filename)
            eqn = str(polyfit_result[0])

            # Calculate the complexity of the polyfit expression the same way as for gradient descent case
            if output_type=="":
                eqn = eqn
            elif output_type=="acos":
                eqn = "cos(" + eqn + ")"
            elif output_type=="asin":
                eqn = "sin(" + eqn + ")"
            elif output_type=="atan":
                eqn = "tan(" + eqn + ")"
            elif output_type=="cos":
                eqn = "acos(" + eqn + ")"
            elif output_type=="exp":
                eqn = "log(" + eqn + ")"
            elif output_type=="inverse":
                eqn = "1/(" + eqn + ")"
            elif output_type=="log":
                eqn = "exp(" + eqn + ")"
            elif output_type=="sin":
                eqn = "asin(" + eqn + ")"
            elif output_type=="sqrt":
                eqn = "(" + eqn + ")**2"
            elif output_type=="squared":
                eqn = "sqrt(" + eqn + ")"
            elif output_type=="tan":
                eqn = "atan(" + eqn + ")"

            polyfit_err = get_symbolic_expr_error(input_data,eqn)
            expr = parse_expr(eqn)
            is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
            numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]
            complexity = 0
            for j in numbers_expr:
                complexity = complexity + get_number_DL_snapped(float(j))
            try:
                # Add the complexity due to symbols
                n_variables = len(polyfit_result[0].free_symbols)
                n_operations = len(count_ops(polyfit_result[0],visual=True).free_symbols)
                if n_operations!=0 or n_variables!=0:
                    complexity = complexity + (n_variables+n_operations)*np.log2((n_variables+n_operations))
            except:
                pass

            #run zero snap on polyfit output
            PA_poly = ParetoSet()
            PA_poly.add(Point(x=complexity, y=polyfit_err, data=str(eqn)))
            PA_poly = add_snap_expr_on_pareto(pathdir, filename, str(eqn), PA_poly)

            for l in range(len(PA_poly.get_pareto_points())):
                PA.add(Point(PA_poly.get_pareto_points()[l][0],PA_poly.get_pareto_points()[l][1],PA_poly.get_pareto_points()[l][2]))

        except:
            pass

        print("Pareto frontier in the current branch:")
        print("")
        print("Complexity #  MDL Loss #  Expression")
        for pareto_i in range(len(PA.get_pareto_points())):
            print(np.round(PA.get_pareto_points()[pareto_i][0],2),np.round(PA.get_pareto_points()[pareto_i][1],2),PA.get_pareto_points()[pareto_i][2])
        print("")

        return PA
    else:
        return PA
