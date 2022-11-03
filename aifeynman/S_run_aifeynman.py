import tempfile

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import traceback
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from os import path
from .logging import std_out_err_redirect_tqdm
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
from .logging import log_exception
from .resources import _get_resource
from cython_wrapper import bf_grad_search


def run_AI_all(XY,BF_try_time=60, polyfit_deg=4, NN_epochs=4000, PA=ParetoSet(), pretrained_model=None, logger=logging.getLogger(__name__),
               bases=None, processes=1, disable_progressbar=False):
    if bases is None:
        bases = ["", "acos", "asin", "atan", "cos", "exp", "inverse", "log", "sin", "sqrt", "squared", "tan"]

    aritytemplates_path = _get_resource("arity2templates.txt")
    if True:
        print(f"Starting brute force with functions {bases} and duration {BF_try_time} seconds per subprocess.")

        # the following with statements are used to redirect logging, stdout and stderr through tqdm.write() such that the
        # logs behave well together with the tqdm progress bars
        # Ref: https://github.com/tqdm/tqdm#redirecting-writing
        with logging_redirect_tqdm():
            with std_out_err_redirect_tqdm() as orig_stdout:
                total_iters = 2*len(bases)
                with tqdm(total=total_iters, file=orig_stdout, position=1, bar_format='{desc} {n_fmt}/{total_fmt}|{bar}|', disable=disable_progressbar) as pbar:
                    '''
                    for base_function in func_iterator:
                        if base_function == "":
                            func_iterator.set_description("Current function: id")
                        else:
                            func_iterator.set_description(f"Current function: {base_function}")
                    '''
                    pbar.set_description("Running brute force module")
                    PA = run_bf_polyfit(XY, BF_try_time, aritytemplates_path, PA, polyfit_deg, bases, pbar, logger, processes)
        logger.info("Brute force done.")
    else:
        logger.warning("Skipping brute force part...")
    
    # TODO: change to 600 or used-provided
    bf_gradient_duration = 100
    band = 0.01
    sigma = 5.0
    mu = 0.0

    # check if the NN is trained. If it is not, train it on the data.
    if len(XY[0])<3:
        print("Only one variable in this data. No neural network will be trained.")
        return PA
    # elif path.exists("results/NN_trained_models/models/" + filename + ".h5"):# or len(data[0])<3:
    elif pretrained_model is not None:
        print("NN already trained")
        model_feynman = pretrained_model
    else:
        print("Training a NN on the data... \n")
        # TODO: Implement option for saving NN? (see above)
        model_feynman = NN_train(XY, NN_epochs)
        #print("NN loss: ", NN_eval(model_feynman, XY, logger=logger), "\n")

    with logging_redirect_tqdm():
        with std_out_err_redirect_tqdm() as orig_stdout:
            with tqdm(total=4, file=orig_stdout, position=1, bar_format='{desc} {n_fmt}/{total_fmt}|{bar}|', disable=disable_progressbar) as pbar:
                pbar.set_description("Checking for: Symmetries")

                # Check which symmetry/separability is the best
                # Symmetries
                #print("Checking for symmetries...")
                symmetry_minus_result = check_translational_symmetry_minus(model_feynman, XY, logger=logger)
                symmetry_divide_result = check_translational_symmetry_divide(model_feynman, XY, logger=logger)
                symmetry_multiply_result = check_translational_symmetry_multiply(model_feynman, XY, logger=logger)
                symmetry_plus_result = check_translational_symmetry_plus(model_feynman, XY, logger=logger)
                #print("")

                pbar.update(1)
                pbar.set_description("Checking for: Separabilities")
                #print("Checking for separabilities...")
                # Separabilities
                separability_plus_result = check_separability_plus(model_feynman, XY, logger=logger)
                separability_multiply_result = check_separability_multiply(model_feynman, XY, logger=logger)

                if symmetry_plus_result[0]==-1:
                    idx_min = -1
                else:
                    idx_min = np.argmin(np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0], separability_plus_result[0], separability_multiply_result[0]]))

                #print(f"idx_min = {idx_min}")
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

                pbar.update(1)
                pbar.set_description("Checking for: Compositionality")
                #print("Checking for compositionality...")
                # Save the gradients for compositionality

                succ_grad, derivates_data = evaluate_derivatives(XY, model_feynman, logger=logger)

                idx_comp = 0
                if succ_grad == 1:
                    with tempfile.NamedTemporaryFile() as bf_results:
                        bf_results_path = bf_results.name
                        logger.debug(f"Brute force results location: {bf_results_path}")

                        print(f"Running compositionability brute force method for {bf_gradient_duration} seconds.")
                        try:
                            # TODO: untested
                            bf_grad_search(derivates_data, derivates_data.shape[0], derivates_data.shape[1], bf_gradient_duration, band, sigma, bf_results_path, aritytemplates_path)
                            #brute_force_comp(derivates_data, bf_results_path, aritytemplates_path, bf_gradient_duration, sigma=10, band=0.01, logger=logger)
                            bf_all_output = np.loadtxt(bf_results_path, dtype="str")
                        except Exception as e:
                            log_exception(logger, e)
                            bf_all_output = []
                        except KeyboardInterrupt:
                            raise KeyboardInterrupt("Propagated from subprocess.")
                    #print(bf_all_output)

                    for candidate in bf_all_output:
                        try:
                            express = candidate[0]
                            num_vars = len(XY[0]) - 1
                            num_vars_expr = get_number_of_variables(express)
                            if num_vars != num_vars_expr:
                                logger.debug(f"Candidate expression only includes {num_vars_expr} out of {num_vars} variables. Skipping this candidate.")
                                continue
                            else:
                                logger.debug(f"Checking candidate expression {express}.")
                            idx_comp_temp, eqq, new_mu, new_sigma = check_compositionality(XY, model_feynman,express,mu,sigma,nu=10)
                            # if found decomposition:
                            if idx_comp_temp==1:
                                idx_comp = 1
                                math_eq_comp = eqq
                                mu = new_mu
                                sigma = new_sigma
                        except Exception as e:
                            log_exception(logger, e)
                            continue

                if idx_comp==1:
                    idx_min = 6

                pbar.update(1)
                pbar.set_description("Checking for: Generalized Symmetry")
                #print("Checking for generalized symmetry...")
                # Check if generalized separabilty is better than the best so far
                idx_gen_sym = 0

                if len(XY[0])>3:
                    try:
                        # find the best separability indices
                        decomp_idx, gradients_data = identify_decompositions(XY, model_feynman)
                        with tempfile.NamedTemporaryFile() as bf_results:
                            bf_results_path = bf_results.name
                            logger.info(f"Brute force gen sym results location: {bf_results_path}")
                            #brute_force_comp(derivates_data, bf_results_path, 600, "14ops.txt", sigma=10, band=0, logger=logger)
                            print(f"Running generalized symmetry brute force method for {bf_gradient_duration} seconds.")
                            try:
                                # TODO: untested
                                bf_grad_search(gradients_data, gradients_data.shape[0], gradients_data.shape[1], bf_gradient_duration, band, sigma, bf_results_path, aritytemplates_path)
                                #brute_force_gen_sym(gradients_data, bf_results_path, aritytemplates_path, cpp_module_path, bf_gradient_duration, "14ops.txt", sigma=10, band=0.01, logger=logger)
                                bf_all_output = np.loadtxt(bf_results_path, dtype="str")
                            except Exception as e:
                                log_exception(logger, e)
                                bf_all_output = []
                            except KeyboardInterrupt:
                                raise KeyboardInterrupt("Propagated from subprocess.")

                        for candidate in bf_all_output:
                            try:
                                express = candidate[0]
                                logger.debug(f"Expression is {express}")
                                idx_gen_sym_temp, eqq, new_mu, new_sigma = check_gen_sym(XY, model_feynman, decomp_idx, express, mu, sigma, nu=10)
                                if idx_gen_sym_temp==1:
                                    idx_gen_sym = 1
                                    math_eq_gen_sym = eqq
                                    mu = new_mu
                                    sigma = new_sigma
                            except Exception as e:
                                log_exception(logger, e)
                                continue
                    except Exception as e:
                        log_exception(logger, e)
                pbar.update(1)

                if idx_gen_sym==1:
                    idx_min = 7
    print("")

    # TODO: CHECK IF NETWORK WEIGHTS PASS WORKS FOR THE FOLLOWING METHODS
    # Apply the best symmetry/separability and rerun the main function on this new file
    if idx_min == 0:
        try:
            #print("Translational symmetry found for variables:", symmetry_plus_result[1],symmetry_plus_result[2])
            varname1 = f"x{symmetry_plus_result[1]}"
            varname2 = f"x{symmetry_plus_result[2]}"
            print(
                f"Translational symmetry found!\nApplying transformation {varname1}, {varname2} -> {varname1} + {varname2}.")

            XY_reduced, model_reduced = do_translational_symmetry_plus(XY, model_feynman, symmetry_plus_result[1],symmetry_plus_result[2])
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_reduced,BF_try_time, polyfit_deg, NN_epochs, PA1_, pretrained_model=model_reduced, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA1:\n{PA1.df()}")
            logger.debug(f"PA:\n{PA.df()}")
            PA = add_sym_on_pareto(PA1,symmetry_plus_result[1],symmetry_plus_result[2],PA,"+")
            logger.debug(f"Merging PA1 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 1:
        try:
            varname1 = f"x{symmetry_minus_result[1]}"
            varname2 = f"x{symmetry_minus_result[2]}"
            print(f"Translational symmetry found!\nApplying transformation {varname1}, {varname2} -> {varname1} - {varname2}.")
            XY_reduced, model_reduced = do_translational_symmetry_minus(XY, model_feynman, symmetry_minus_result[1], symmetry_minus_result[2])
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_reduced,BF_try_time, polyfit_deg, NN_epochs, PA1_, pretrained_model=model_reduced, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA1:\n{PA1.df()}")
            logger.debug(f"PA:\n{PA.df()}")
            PA = add_sym_on_pareto(PA1,symmetry_minus_result[1],symmetry_minus_result[2],PA,"-")
            logger.debug(f"Merging PA1 and PA yielded:\n{PA.df()}")

        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 2:
        try:
            #print("Translational symmetry found for variables:", symmetry_multiply_result[1],symmetry_multiply_result[2])
            varname1 = f"x{symmetry_multiply_result[1]}"
            varname2 = f"x{symmetry_multiply_result[2]}"
            print(
                f"Translational symmetry found!\nApplying transformation {varname1}, {varname2} -> {varname1} * {varname2}.")

            XY_reduced, model_reduced = do_translational_symmetry_multiply(XY, model_feynman, symmetry_multiply_result[1],symmetry_multiply_result[2])
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_reduced, BF_try_time, polyfit_deg, NN_epochs, PA1_, pretrained_model=model_reduced, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA1:\n{PA1.df()}")
            logger.debug(f"PA:\n{PA.df()}")
            PA = add_sym_on_pareto(PA1,symmetry_multiply_result[1],symmetry_multiply_result[2],PA,"*")
            logger.debug(f"Merging PA1 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 3:
        try:
            #print("Translational symmetry found for variables:", symmetry_divide_result[1],symmetry_divide_result[2])
            varname1 = f"x{symmetry_divide_result[1]}"
            varname2 = f"x{symmetry_divide_result[2]}"
            print(
                f"Translational symmetry found!\nApplying transformation {varname1}, {varname2} -> {varname1} / {varname2}.")

            XY_reduced, model_reduced = do_translational_symmetry_divide(XY, model_feynman, symmetry_divide_result[1],symmetry_divide_result[2])
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_reduced,BF_try_time, polyfit_deg, NN_epochs, PA1_, pretrained_model=model_reduced, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA1:\n{PA1.df()}")
            logger.debug(f"PA:\n{PA.df()}")
            PA = add_sym_on_pareto(PA1,symmetry_divide_result[1],symmetry_divide_result[2],PA,"/")
            logger.debug(f"Merging PA1 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 4:
        try:
            print("Additive separability found for variables:", separability_plus_result[1],separability_plus_result[2])
            XY_sep_1, XY_sep_2 = do_separability_plus(XY, model_feynman, separability_plus_result[1],separability_plus_result[2])
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_sep_1, BF_try_time, polyfit_deg, NN_epochs, PA1_, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            PA2_ = ParetoSet()
            PA2 = run_AI_all(XY_sep_2, BF_try_time, polyfit_deg, NN_epochs, PA2_, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA:\n{PA.df()}")
            logger.debug(f"PA1:\n{PA1.df()}")
            logger.debug(f"PA2:\n{PA2.df()}")
            #combine_pareto_data = np.loadtxt(pathdir+filename)
            PA = combine_pareto(XY, PA1, PA2, separability_plus_result[1],separability_plus_result[2],PA,"+", logger=logger)
            logger.debug(f"Merging PA1, PA2 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 5:
        try:
            print("Multiplicative separability found for variables:", separability_multiply_result[1],separability_multiply_result[2])
            XY_sep_1, XY_sep_2  = do_separability_multiply(XY, model_feynman, separability_multiply_result[1],separability_multiply_result[2])
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_sep_1,BF_try_time, polyfit_deg, NN_epochs, PA1_, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            PA2_ = ParetoSet()
            PA2 = run_AI_all(XY_sep_2,BF_try_time, polyfit_deg, NN_epochs, PA2_, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA:\n{PA.df()}")
            logger.debug(f"PA1:\n{PA1.df()}")
            logger.debug(f"PA2:\n{PA2.df()}")
            PA = combine_pareto(XY,PA1,PA2,separability_multiply_result[1],separability_multiply_result[2],PA,"*", logger=logger)
            logger.debug(f"Merging PA1, PA2 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 6:
        try:
            print("Compositionality found")
            XY_comp = do_compositionality(XY, math_eq_comp)
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_comp,BF_try_time, polyfit_deg, NN_epochs, PA1_, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA:\n{PA.df()}")
            logger.debug(f"PA1:\n{PA1.df()}")
            PA = add_comp_on_pareto(PA1, PA, math_eq_comp)
            logger.debug(f"Merging PA1 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA

    elif idx_min == 7:
        try:
            print("Generalized symmetry found")
            XY_gen_sym = do_gen_sym(XY, decomp_idx,math_eq_gen_sym)
            PA1_ = ParetoSet()
            PA1 = run_AI_all(XY_gen_sym,BF_try_time, polyfit_deg, NN_epochs, PA1_, logger=logger, bases=bases, processes=processes, disable_progressbar=disable_progressbar)
            logger.debug(f"PA:\n{PA.df()}")
            logger.debug(f"PA1:\n{PA1.df()}")
            PA = add_gen_sym_on_pareto(PA1,PA, decomp_idx, math_eq_gen_sym)
            logger.debug(f"Merging PA1 and PA yielded:\n{PA.df()}")
        except Exception as e:
            log_exception(logger, e)
        return PA
    else:
        return PA

'''
def run_aifeynman(pathdir,filename,BF_try_time,BF_ops_file_type, polyfit_deg=4, NN_epochs=4000, vars_name=[],test_percentage=20, debug=False, bases=None):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)  # change to logging.INFO to see reduced debug logging

    logger = logging.getLogger(__name__)

    if not bases:
        bases = ["", "acos", "asin", "atan", "cos", "exp", "inverse", "log", "sin", "sqrt", "squared", "tan"]

    # If the variable names are passed, do the dimensional analysis first
    filename_orig = filename
    try:
        if vars_name!=[]:
            print("Running dimensional analysis with passed vars_name.")
            dimensionalAnalysis(pathdir,filename,vars_name)
            DR_file = filename + "_dim_red_variables.txt"
            filename = filename + "_dim_red"
        else:
            print("No vars_name was given. Running without dimensional analysis.")
            DR_file = ""
    except Exception as e:
        log_exception(logger, e)
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
    PA = run_AI_all(pathdir,filename+"_train",BF_try_time,BF_ops_file_type, polyfit_deg, NN_epochs, PA=PA, logger=logger, bases=bases)
    PA_list = PA.get_pareto_points()

    '''
'''
    # Run bf snap on the resulted equations
    for i in range(len(PA_list)):
        try:
            PA = add_bf_on_numbers_on_pareto(pathdir,filename,PA,PA_list[i][-1])
        except:
            continue
    PA_list = PA.get_pareto_points()
    '''
'''
    np.savetxt("results/solution_before_snap_%s.txt" %filename,PA_list,fmt="%s", delimiter=',')


    # Run zero, integer and rational snap on the resulted equations
    for j in range(len(PA_list)):
        PA = add_snap_expr_on_pareto(pathdir,filename,PA_list[j][-1],PA, "", logger=logger)

    PA_list = PA.get_pareto_points()
    np.savetxt("results/solution_first_snap_%s.txt" %filename,PA_list,fmt="%s", delimiter=',')

    # Run gradient descent on the data one more time
    for i in range(len(PA_list)):
        try:
            dt = np.loadtxt(pathdir+filename)
            gd_update = final_gd(dt,PA_list[i][-1], logger=logger)
            PA.add(Point(x=gd_update[1],y=gd_update[0],data=gd_update[2]))
        except Exception as e:
            log_exception(logger, e)
            continue

    PA_list = PA.get_pareto_points()
    for j in range(len(PA_list)):
        PA = add_snap_expr_on_pareto(pathdir,filename,PA_list[j][-1],PA, DR_file, logger=logger)

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
            test_errors = test_errors + [get_symbolic_expr_error(input_test_data,str(list_dt[i][-1]), logger=logger)]
        test_errors = np.array(test_errors)
        # Save all the data to file
        save_data = np.column_stack((test_errors,log_err,log_err_all,list_dt))
    else:
        save_data = np.column_stack((log_err,log_err_all,list_dt))
    np.savetxt("results/solution_%s" %filename_orig,save_data,fmt="%s", delimiter=',')
    try:
        os.remove(pathdir+filename+"_test")
        os.remove(pathdir+filename+"_train")
    except:
        pass
    return PA
'''