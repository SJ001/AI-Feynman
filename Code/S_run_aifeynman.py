import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from get_pareto import Point, ParetoSet
from RPN_to_pytorch import RPN_to_pytorch
from RPN_to_eq import RPN_to_eq
from S_NN_train import NN_train
from S_NN_eval import NN_eval
from S_symmetry import *
from S_separability import *
from S_change_output import *
from S_brute_force import brute_force
from S_combine_pareto import combine_pareto
from S_get_number_DL import get_number_DL
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops
from S_polyfit import polyfit
from S_get_symbolic_expr_error import get_symbolic_expr_error
from S_add_snap_expr_on_pareto import add_snap_expr_on_pareto
from S_add_sym_on_pareto import add_sym_on_pareto
from S_run_bf_polyfit import run_bf_polyfit


PA = ParetoSet()

def run_AI_all(pathdir,filename,BF_try_time=60,BF_ops_file_type="14ops", polyfit_deg=4, NN_epochs=4000, PA = PA):
    try:
        os.mkdir("results/")
    except:
        pass
    
    # load the data for different checks
    data = np.loadtxt(pathdir+filename)

    # Run bf and polyfit
    PA = run_bf_polyfit(pathdir,pathdir,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg)



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
    print("Checking for symmetry \n", filename)
    if path.exists("results/NN_trained_models/models/" + filename + ".h5") or len(data[0])<3:
        print("NN already trained \n")
        print("NN loss: ", NN_eval(pathdir,filename), "\n")
    else:
        print("Training a NN on the data... \n")
        NN_train(pathdir,filename,NN_epochs)
        print("NN loss: ", NN_eval(pathdir,filename), "\n")
        
    # Check which symmetry/separability is the best
    
    # Symmetries
    symmetry_minus_result = check_translational_symmetry_minus(pathdir,filename)
    symmetry_divide_result = check_translational_symmetry_divide(pathdir,filename)
    symmetry_multiply_result = check_translational_symmetry_multiply(pathdir,filename)
    symmetry_plus_result = check_translational_symmetry_plus(pathdir,filename)
    
    # Separabilities
    separability_plus_result = check_separability_plus(pathdir,filename)
    separability_multiply_result = check_separability_multiply(pathdir,filename)
    
    if symmetry_plus_result[0]==-1:
        idx_min = -1
    else:
        idx_min = np.argmin(np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0], separability_plus_result[0], separability_multiply_result[0]]))

    # Apply the best symmetry/separability and rerun the main function on this new file
    if idx_min == 0:
        new_pathdir, new_filename = do_translational_symmetry_plus(pathdir,filename,symmetry_plus_result[1],symmetry_plus_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_plus_result[1],symmetry_plus_result[2],PA,"+")
        return PA
    
    elif idx_min == 1:
        new_pathdir, new_filename = do_translational_symmetry_minus(pathdir,filename,symmetry_minus_result[1],symmetry_minus_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_minus_result[1],symmetry_minus_result[2],PA,"-")
        return PA
    
    elif idx_min == 2:
        new_pathdir, new_filename = do_translational_symmetry_multiply(pathdir,filename,symmetry_multiply_result[1],symmetry_multiply_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_multiply_result[1],symmetry_multiply_result[2],PA,"*")
        return PA
    
    elif idx_min == 3:
        new_pathdir, new_filename = do_translational_symmetry_divide(pathdir,filename,symmetry_divide_result[1],symmetry_divide_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir,new_filename,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA = add_sym_on_pareto(pathdir,filename,PA1,symmetry_divide_result[1],symmetry_divide_result[2],PA,"/")
        return PA
    
    elif idx_min == 4:
        new_pathdir1, new_filename1, new_pathdir2, new_filename2,  = do_separability_plus(pathdir,filename,separability_plus_result[1],separability_plus_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir1,new_filename1,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA2_ = ParetoSet()
        PA2 = run_AI_all(new_pathdir2,new_filename2,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA2_)
        PA = combine_pareto(pathdir,filename,PA1,PA2,separability_plus_result[1],separability_plus_result[2],PA,"+")
        return PA 
    
    elif idx_min == 5:
        new_pathdir1, new_filename1, new_pathdir2, new_filename2,  = do_separability_multiply(pathdir,filename,separability_multiply_result[1],separability_multiply_result[2])
        PA1_ = ParetoSet()
        PA1 = run_AI_all(new_pathdir1,new_filename1,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA1_)
        PA2_ = ParetoSet()
        PA2 = run_AI_all(new_pathdir2,new_filename2,BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA2_)
        PA = combine_pareto(pathdir,filename,PA1,PA2,separability_multiply_result[1],separability_multiply_result[2],PA,"*")
        return PA 
    else:
        return PA

# this runs snap on the output of aifeynman
def run_aifeynman(pathdir,filename,BF_try_time,BF_ops_file_type, polyfit_deg=4, NN_epochs=4000, DR_file=""):
    # Split the data into train and test set                                                                                                                                      
    input_data = np.loadtxt(pathdir+filename)
    sep_idx = np.random.permutation(len(input_data))
    train_data = input_data[sep_idx[0:8*len(input_data)//10]]
    test_data = input_data[sep_idx[8*len(input_data)//10:len(input_data)]]

    np.savetxt(pathdir+filename+"_train",train_data)
    np.savetxt(pathdir+filename+"_test",test_data)

    # Run the code on the train data 
    PA = run_AI_all(pathdir,filename+"_train",BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs)
    PA_list = PA.get_pareto_points()
    PA_snapped = ParetoSet()
    
    np.savetxt("results/solution_before_snap_%s.txt" %filename,PA_list,fmt="%s")
    for j in range(len(PA_list)):
        PA_snapped = add_snap_expr_on_pareto(pathdir,filename,PA_list[j][-1],PA_snapped, DR_file)

    list_dt = np.array(PA_snapped.get_pareto_points())
    
    data_file_len = len(np.loadtxt(pathdir+filename))
    log_err = []
    log_err_all = []
    for i in range(len(list_dt)):
        log_err = log_err + [np.log2(float(list_dt[i][1]))]
        log_err_all = log_err_all + [data_file_len*np.log2(float(list_dt[i][1]))]
    log_err = np.array(log_err)
    log_err_all = np.array(log_err_all)

    # Try the found expressions on the test data                                                                                                                                  
    if DR_file=="":
        test_errors = []
        for i in range(len(list_dt)):
            test_errors = test_errors + [get_symbolic_expr_error(pathdir,filename+"_test",str(list_dt[i][-1]))]
        test_errors = np.array(test_errors)
        # Save all the data to file                                                                                                                                               
        save_data = np.column_stack((test_errors,log_err,log_err_all,list_dt))
    else:
        save_data = np.column_stack((log_err,log_err_all,list_dt))

    np.savetxt("results/solution_%s.txt" %filename,save_data,fmt="%s")
