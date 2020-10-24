import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from .get_pareto import Point, ParetoSet
from .RPN_to_pytorch import RPN_to_pytorch
from .RPN_to_eq import RPN_to_eq
from .S_NN_train import NN_train
from .S_NN_eval import NN_eval
from .S_symmetry import *
from .S_separability import *
from .S_change_output import *
from .S_brute_force import brute_force
from .S_combine_pareto import combine_pareto
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops
from .S_polyfit import polyfit
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .S_add_snap_expr_on_pareto import add_snap_expr_on_pareto
from .S_add_sym_on_pareto import add_sym_on_pareto
from .S_run_bf_polyfit import run_bf_polyfit
from .S_final_gd import final_gd
from .S_add_bf_on_numbers_on_pareto import add_bf_on_numbers_on_pareto
from .dimensionalAnalysis import dimensionalAnalysis
from .S_NN_get_gradients import evaluate_derivatives
from .S_brute_force_comp import brute_force_comp
from .S_brute_force_gen_sym import brute_force_gen_sym
from .S_compositionality import *
from .S_gen_sym import *
from .S_gradient_decomposition import identify_decompositions

PA = ParetoSet()
def run_AI_all(pathdir,filename,BF_try_time=60,BF_ops_file_type="14ops", polyfit_deg=4, NN_epochs=4000, PA=PA):
    try:
        os.mkdir("results/")
    except:
        pass

    # load the data for different checks
    data = np.loadtxt(pathdir+filename)

    # Run bf and polyfit
    PA = run_bf_polyfit(pathdir,pathdir,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_squared(pathdir,"results/mystery_world_squared/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)   

    # Run bf and polyfit on modified output

    PA = get_acos(pathdir,"results/mystery_world_acos/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_asin(pathdir,"results/mystery_world_asin/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_atan(pathdir,"results/mystery_world_atan/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_cos(pathdir,"results/mystery_world_cos/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_exp(pathdir,"results/mystery_world_exp/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_inverse(pathdir,"results/mystery_world_inverse/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_log(pathdir,"results/mystery_world_log/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_sin(pathdir,"results/mystery_world_sin/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_sqrt(pathdir,"results/mystery_world_sqrt/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_squared(pathdir,"results/mystery_world_squared/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)
    PA = get_tan(pathdir,"results/mystery_world_tan/",filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)

#############################################################################################################################
    # check if the NN is trained. If it is not, train it on the data.
    if len(data[0])<3:
        print("Just one variable!")
        pass
    elif path.exists("results/NN_trained_models/models/" + filename + ".h5"):# or len(data[0])<3:
        print("NN already trained \n")
        print("NN loss: ", NN_eval(pathdir,filename)[0], "\n")
        model_feynman = NN_eval(pathdir,filename)[1]
    elif path.exists("results/NN_trained_models/models/" + filename + "_pretrained.h5"):
        print("Found pretrained NN \n")
        model_feynman = NN_train(pathdir,filename,NN_epochs/2,lrs=1e-3,N_red_lr=3,pretrained_path="results/NN_trained_models/models/" + filename + "_pretrained.h5")
        print("NN loss after training: ", NN_eval(pathdir,filename), "\n")
    else:
        print("Training a NN on the data... \n")
        model_feynman = NN_train(pathdir,filename,NN_epochs)
        print("NN loss: ", NN_eval(pathdir,filename), "\n")

    
    # Check which symmetry/separability is the best
    # Symmetries
    print("Checking for symmetries...")
    symmetry_minus_result = check_translational_symmetry_minus(pathdir,filename)
    symmetry_divide_result = check_translational_symmetry_divide(pathdir,filename)
    symmetry_multiply_result = check_translational_symmetry_multiply(pathdir,filename)
    symmetry_plus_result = check_translational_symmetry_plus(pathdir,filename)
    print("")

    print("Checking for separabilities...")
    # Separabilities
    separability_plus_result = check_separability_plus(pathdir,filename)
    separability_multiply_result = check_separability_multiply(pathdir,filename)

    if symmetry_plus_result[0]==-1:
        idx_min = -1
    else:
        idx_min = np.argmin(np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0], separability_plus_result[0], separability_multiply_result[0]]))

    print("")
    # Check if compositionality is better than the best so far
    if idx_min==0:
        mu, sigma = symmetry_plus_result[3:]
    elif idx_min==1:
        mu, sigma = symmetry_minus_result[3:]
    elif idx_min==2:
        mu, sigma = symmetry_multiply_result[3:]
    elif idx_min==3:
        mu, sigma = symmetry_divide_result[3:]
    elif idx_min==4:
        mu, sigma = separability_plus_result[3:]
    elif idx_min==5:
        mu, sigma = separability_multiply_result[3:]

    print("Checking for compositionality...")
    # Save the gradients for compositionality
    try:
        succ_grad = evaluate_derivatives(pathdir,filename,model_feynman)
    except:
        succ_grad = 0

    idx_comp = 0
    if succ_grad == 1:
        #try:
        for qqqq in range(1):
            brute_force_comp("results/","gradients_comp_%s.txt" %filename,600,"14ops.txt")
            bf_all_output = np.loadtxt("results_comp.dat", dtype="str")
            for bf_i in range(len(bf_all_output)):
                idx_comp_temp = 0
                try:
                    express = bf_all_output[:,1][bf_i]
                    idx_comp_temp, eqq, new_mu, new_sigma = check_compositionality(pathdir,filename,model_feynman,express,mu,sigma,nu=10)
                    if idx_comp_temp==1:
                        idx_comp = 1
                        math_eq_comp = eqq
                        mu = new_mu
                        sigma = new_sigma
                except:
                    continue
        #except:
        #    idx_comp = 0
    else:
        idx_comp = 0
    print("")
    
    if idx_comp==1:
        idx_min = 6


    print("Checking for generalized symmetry...")
    # Check if generalized separabilty is better than the best so far
    idx_gen_sym = 0
    for kiiii in range(1):
        if len(data[0])>3:
            # find the best separability indices
            decomp_idx = identify_decompositions(pathdir,filename, model_feynman)
            brute_force_gen_sym("results/","gradients_gen_sym_%s" %filename,600,"14ops.txt")
            bf_all_output = np.loadtxt("results_gen_sym.dat", dtype="str")
            
            for bf_i in range(len(bf_all_output)):
                idx_gen_sym_temp = 0
                try:
                    express = bf_all_output[:,1][bf_i]
                    idx_gen_sym_temp, eqq, new_mu, new_sigma = check_gen_sym(pathdir,filename,model_feynman,decomp_idx,express,mu,sigma,nu=10)
                    if idx_gen_sym_temp==1:
                        idx_gen_sym = 1
                        math_eq_gen_sym = eqq
                        mu = new_mu
                        sigma = new_sigma
                except:
                    continue

    if idx_gen_sym==1:
        idx_min = 7
    print("")

    # Apply the best symmetry/separability and rerun the main function on this new file
    if idx_min == 0:
        print("Translational symmetry found for variables:", symmetry_plus_result[1],symmetry_plus_result[2])
        new_pathdir, new_filename = do_translational_symmetry_plus(pathdir,filename,symmetry_plus_result[1],symmetry_plus_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_plus_result[1],symmetry_plus_result[2],PA,"+")
        return PA

    elif idx_min == 1:
        print("Translational symmetry found for variables:", symmetry_minus_result[1],symmetry_minus_result[2])
        new_pathdir, new_filename = do_translational_symmetry_minus(pathdir,filename,symmetry_minus_result[1],symmetry_minus_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_minus_result[1],symmetry_minus_result[2],PA,"-")
        return PA

    elif idx_min == 2:
        print("Translational symmetry found for variables:", symmetry_multiply_result[1],symmetry_multiply_result[2])
        new_pathdir, new_filename = do_translational_symmetry_multiply(pathdir,filename,symmetry_multiply_result[1],symmetry_multiply_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_multiply_result[1],symmetry_multiply_result[2],PA,"*")
        return PA

    elif idx_min == 3:
        print("Translational symmetry found for variables:", symmetry_divide_result[1],symmetry_divide_result[2])
        new_pathdir, new_filename = do_translational_symmetry_divide(pathdir,filename,symmetry_divide_result[1],symmetry_divide_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_divide_result[1],symmetry_divide_result[2],PA,"/")
        return PA

    elif idx_min == 4:
        print("Additive separability found for variables:", separability_plus_result[1],separability_plus_result[2])
        new_pathdir1, new_filename1, new_pathdir2, new_filename2,  = do_separability_plus(pathdir,filename,separability_plus_result[1],separability_plus_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir1,new_filename1,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA2_ = ParetoSet()
        PA2 = run_AI_all(new_pathdir2,new_filename2,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA2_)
        combine_pareto_data = np.loadtxt(pathdir+filename)
        PA = combine_pareto(combine_pareto_data,PA1,PA2,separability_plus_result[1],separability_plus_result[2],PA,"+")
        return PA

    elif idx_min == 5:
        print("Multiplicative separability found for variables:", separability_multiply_result[1],separability_multiply_result[2])
        new_pathdir1, new_filename1, new_pathdir2, new_filename2,  = do_separability_multiply(pathdir,filename,separability_multiply_result[1],separability_multiply_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir1,new_filename1,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA2_ = ParetoSet()
        PA2 = run_AI_all(new_pathdir2,new_filename2,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA2_)
        combine_pareto_data = np.loadtxt(pathdir+filename)
        PA = combine_pareto(combine_pareto_data,PA1,PA2,separability_multiply_result[1],separability_multiply_result[2],PA,"*")
        return PA

    elif idx_min == 6:
        print("Compositionality found")
        new_pathdir, new_filename = do_compositionality(pathdir,filename,math_eq_comp)
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_comp_on_pareto(PA1,PA,math_eq_comp)
        return PA

    elif idx_min == 7:
        print("Generalized symmetry found")
        new_pathdir, new_filename = do_gen_sym(pathdir,filename,decomp_idx,math_eq_gen_sym)
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_gen_sym_on_pareto(PA1,PA, decomp_idx, math_eq_gen_sym)
        return PA
    else:
        return PA
# this runs snap on the output of aifeynman
def run_aifeynman(pathdir,filename,BF_try_time,BF_ops_file_type, polyfit_deg=4, NN_epochs=4000, vars_name=[],test_percentage=20):
    # If the variable names are passed, do the dimensional analysis first
    filename_orig = filename
    try:
        if vars_name!=[]:
            dimensionalAnalysis(pathdir,filename,vars_name)
            DR_file = filename + "_dim_red_variables.txt"
            filename = filename + "_dim_red"
        else:
            DR_file = ""
    except:
        DR_file = ""

    # Split the data into train and test set
    input_data = np.loadtxt(pathdir+filename)
    sep_idx = np.random.permutation(len(input_data))

    train_data = input_data[sep_idx[0:(100-test_percentage)*len(input_data)//100]]
    test_data = input_data[sep_idx[test_percentage*len(input_data)//100:len(input_data)]]

    np.savetxt(pathdir+filename+"_train",train_data)
    if test_data.size != 0:
        np.savetxt(pathdir+filename+"_test",test_data)

    PA = ParetoSet()
    # Run the code on the train data
    PA = run_AI_all(pathdir,filename+"_train",BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA=PA)
    PA_list = PA.get_pareto_points()

    '''
    # Run bf snap on the resulted equations
    for i in range(len(PA_list)):
        try:
            PA = add_bf_on_numbers_on_pareto(pathdir,filename,PA,PA_list[i][-1])
        except:
            continue
    PA_list = PA.get_pareto_points()
    '''

    np.savetxt("results/solution_before_snap_%s.txt" %filename,PA_list,fmt="%s")


    # Run zero, integer and rational snap on the resulted equations
    for j in range(len(PA_list)):
        PA = add_snap_expr_on_pareto(pathdir,filename,PA_list[j][-1],PA, "")

    PA_list = PA.get_pareto_points()
    np.savetxt("results/solution_first_snap_%s.txt" %filename,PA_list,fmt="%s")

    # Run gradient descent on the data one more time
    for i in range(len(PA_list)):
        try:
            gd_update = final_gd(pathdir,filename,PA_list[i][-1])
            PA.add(Point(x=gd_update[1],y=gd_update[0],data=gd_update[2]))
        except:
            continue

    PA_list = PA.get_pareto_points()
    for j in range(len(PA_list)):
        PA = add_snap_expr_on_pareto(pathdir,filename,PA_list[j][-1],PA, DR_file)

    list_dt = np.array(PA.get_pareto_points())
    data_file_len = len(np.loadtxt(pathdir+filename))
    log_err = []
    log_err_all = []
    for i in range(len(list_dt)):
        log_err = log_err + [np.log2(float(list_dt[i][1]))]
        log_err_all = log_err_all + [data_file_len*np.log2(float(list_dt[i][1]))]
    log_err = np.array(log_err)
    log_err_all = np.array(log_err_all)

    # Try the found expressions on the test data
    if DR_file=="" and test_data.size != 0:
        test_errors = []
        input_test_data = np.loadtxt(pathdir+filename+"_test")
        for i in range(len(list_dt)):
            test_errors = test_errors + [get_symbolic_expr_error(input_test_data,str(list_dt[i][-1]))]
        test_errors = np.array(test_errors)
        # Save all the data to file
        save_data = np.column_stack((test_errors,log_err,log_err_all,list_dt))
    else:
        save_data = np.column_stack((log_err,log_err_all,list_dt))
    np.savetxt("results/solution_%s" %filename_orig,save_data,fmt="%s")
    try:
        os.remove(pathdir+filename+"_test")
        os.remove(pathdir+filename+"_train")
    except:
        pass

