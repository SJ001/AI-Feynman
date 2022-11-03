from multiprocessing.util import get_logger
from .get_pareto import Point, ParetoSet
from .RPN_to_pytorch import RPN_to_pytorch
from .RPN_to_eq import RPN_to_eq
import numpy as np
from .S_get_number_DL_snapped import get_number_DL_snapped
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops
from .S_polyfit import polyfit
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .S_add_snap_expr_on_pareto import add_snap_expr_on_pareto

from .logging import log_exception
import multiprocessing
import tempfile
import pandas as pd
import traceback
from cython_wrapper import bf_search
import logging

from contextlib import redirect_stdout
import os


def run_bf_polyfit(XY,BF_try_time, aritytemplates_path, PA, polyfit_deg, bases, pbar, logger, processes=2):
    global bf_workerprocess

    def bf_workerprocess(data, sep_type, output_type):
        with tempfile.NamedTemporaryFile() as bf_results:
            bf_results_path = bf_results.name
            logger = logging.getLogger('bf_results_path')
            print(os.getpid())
            logger.write = lambda msg: logger.debug(msg) if msg != '\n' else None
            logger.debug(f"Brute force results location: {bf_results_path}")
            # run BF on the data
            #print(f"Checking for brute force {sep_type} and output_type {output_type}")
            #brute_force(data, BF_try_time, aritytemplates_path, bf_results_path, sep_type,
            #            logger=logger)
            band = 0.01
            sigma = 5.0
            try:
                with redirect_stdout(logger):
                    bf_search(data, data.shape[0], data.shape[1], BF_try_time, sep_type, band, sigma,
                            bf_results_path, aritytemplates_path)
                bf_all_output = np.loadtxt(bf_results_path, dtype="str")
            except Exception as e:
                log_exception(logger, e)
                bf_all_output = []

        return bf_all_output


    # TODO: Add custom function definitions here.
    #  Also add custom inverse string representation in bruteforce result postprocessing
    basis_func_definition = {
        "asin": lambda d: np.arcsin(d),
        "acos": lambda d: np.arccos(d),
        "atan": lambda d: np.arctan(d),
        "sin": lambda d: np.sin(d),
        "cos": lambda d: np.cos(d),
        "tan": lambda d: np.tan(d),
        "exp": lambda d: np.exp(d),
        "log": lambda d: np.log(d),
        "inverse": lambda d: 1/d,
        "sqrt": lambda d: np.sqrt(d),
        "squared": lambda d: d**2,
        "": lambda d: d
    }
    multiprocessing_arguments = []

    for basis_func in bases:
        try:
            f = basis_func_definition[basis_func]
        except KeyError:
            logger.warning(f"No definition for basis function {basis_func} was given. Skipping.")
            logger.debug(traceback.format_exc())
            pbar.update(2)
            continue
        data_transformed = np.copy(XY)
        data_transformed[:, -1] = f(data_transformed[:, -1])
        if np.isnan(data_transformed).any():
            logger.info(
                f"Basis function {basis_func} was skipped because when applied to input data it produces NaNs.")
            pbar.update(2)
            continue
        # first try to fit polynomial to transformed data
        PA = run_polyfit(np.copy(data_transformed), basis_func, polyfit_deg, PA, logger)

        # then try brute force with separators + and *
        for sep_type in ["*", "+"]:
            if sep_type == "*":
                ndata_init = data_transformed.shape[0]
                ymax = np.max(np.abs(data_transformed[:, -1]))
                array_mask = np.abs(data_transformed[:, -1]) > 0.001 * ymax
                data_transformed = data_transformed[array_mask, :]
                ndata_post = data_transformed.shape[0]
                logger.info(
                    f"Removed {ndata_init - ndata_post} problematically small data points. {ndata_post} remains.")
                if ndata_post < 0.01 * ndata_init:
                    logger.info("Less than 1% of data points remains. Skipping round.")
                    pbar.update(1)
                    continue

            # if all tests so far have passed, then add config to multiprocessing queue
            arguments = (np.copy(data_transformed), sep_type, basis_func)
            multiprocessing_arguments.append(arguments)

            if len(multiprocessing_arguments) == processes:
                pbar.set_description("Running brute force module")
                print(f"Running brute force configurations over {len(multiprocessing_arguments)} cores:")
                for _, sep, outp in multiprocessing_arguments:
                    print(f"- Separator {sep} and transformation function {outp}")
                print(f"Brute force will now run for {BF_try_time} seconds.")
                with multiprocessing.Pool(processes) as pool:
                    # Run brute force processes in parallel
                    try:
                        bf_results = pool.starmap(bf_workerprocess, multiprocessing_arguments)
                    except KeyboardInterrupt:
                        pool.terminate()
                        raise KeyboardInterrupt("Propagated from subprocess.")
                # Iterate over results and process them
                for i, bf_result in enumerate(bf_results):
                    pbar.set_description("Processing brute force results")
                    curr_data, sep_type, output_type = multiprocessing_arguments[i]
                    PA = bf_result_processing(bf_result, sep_type, output_type, curr_data, PA, logger)
                    pbar.update(1)
                multiprocessing_arguments = []

                print("Pareto frontier in the current branch:")
                print(PA.df())
                #print("")
                #print("Complexity #  MDL Loss #  Expression")
                #for pareto_i in range(len(PA.get_pareto_points())):
                #    print(np.round(PA.get_pareto_points()[pareto_i][0], 2),
                #          np.round(PA.get_pareto_points()[pareto_i][1], 2), PA.get_pareto_points()[pareto_i][2])
                print("")

    # if there are still processes left to be run after the for loops end
    if len(multiprocessing_arguments) != 0:
        pbar.set_description("Running brute force module")
        print(f"Running brute force configurations over {len(multiprocessing_arguments)} cores:")
        for _, sep, outp in multiprocessing_arguments:
            print(f"- Separator {sep} and transformation function {outp}")
        print(f"Brute force will now run for {BF_try_time} seconds.")
        with multiprocessing.Pool(len(multiprocessing_arguments)) as p:
            # Run brute force processes in parallel
            bf_results = p.starmap(bf_workerprocess, multiprocessing_arguments)

        # Iterate over results and process them
        for i, bf_result in enumerate(bf_results):
            pbar.set_description("Processing brute force results")
            curr_data, sep_type, output_type = multiprocessing_arguments[i]
            PA = bf_result_processing(bf_result, sep_type, output_type, curr_data, PA, logger)
            pbar.update(1)

        print("Pareto frontier in the current branch:")
        print(PA.df())
        #print("")
        #print("Complexity #  MDL Loss #  Expression")
        #for pareto_i in range(len(PA.get_pareto_points())):
        #    print(np.round(PA.get_pareto_points()[pareto_i][0], 2),
        #          np.round(PA.get_pareto_points()[pareto_i][1], 2), PA.get_pareto_points()[pareto_i][2])
        print("")
    return PA


def run_polyfit(data, output_type, polyfit_deg, PA, logger):
    # run polyfit on the data
    print(f"Running polyfit on transformed data, {output_type}")
    try:
        polyfit_result = polyfit(data, polyfit_deg, logger=logger)
        eqn = str(polyfit_result[0])
        logger.debug(f"Polyfit result is: {eqn}")
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

        polyfit_err = get_symbolic_expr_error(data,eqn, logger=logger)
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
        except Exception as e:
            log_exception(logger, e)
            pass

        #run zero snap on polyfit output
        PA_poly = ParetoSet()
        PA_poly.add(Point(x=complexity, y=polyfit_err, data=str(eqn)))
        PA_poly = add_snap_expr_on_pareto(data, str(eqn), PA_poly, logger=logger)

        for l in range(len(PA_poly.get_pareto_points())):
            PA.add(Point(PA_poly.get_pareto_points()[l][0],PA_poly.get_pareto_points()[l][1],PA_poly.get_pareto_points()[l][2]))

    except Exception as e:
        log_exception(logger, e)

    return PA


def bf_result_processing(bf_result, sep_type, output_type, data, PA, logger):
    if bf_result.size == 0:
        logger.info(f"Brute force module returned no result for operator {sep_type} and output type {output_type}.")
        return PA

    if len(bf_result.shape) == 1:
        bf_result = np.expand_dims(bf_result, axis=0)

    df = pd.DataFrame(bf_result,
                      columns=['Expr', 'Error [bits]', 'Prefactor', 'N_evals', 'N_formulas', 'Average evals'])
    print(f"Results of brute force with operator {sep_type} and output type {output_type}:")
    print(df)
    express = bf_result[:, 0]
    prefactors = bf_result[:, 2]
    try:
        # Calculate the complexity of the bf expression the same way as for gradient descent case
        complexity = []
        errors = []
        eqns = []
        # TODO: make this work with custom basis funcs
        for i in range(len(prefactors)):
            logger.debug(f"Processing expression {express[i]} with prefactor {prefactors[i]}.")
            try:
                if output_type == "":
                    eqn = prefactors[i] + sep_type + RPN_to_eq(express[i])
                elif output_type == "acos":
                    eqn = "cos(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "asin":
                    eqn = "sin(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "atan":
                    eqn = "tan(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "cos":
                    eqn = "acos(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "exp":
                    eqn = "log(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "inverse":
                    eqn = "1/(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "log":
                    eqn = "exp(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "sin":
                    eqn = "asin(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "sqrt":
                    eqn = "(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")**2"
                elif output_type == "squared":
                    eqn = "sqrt(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                elif output_type == "tan":
                    eqn = "atan(" + prefactors[i] + sep_type + RPN_to_eq(express[i]) + ")"
                else:
                    logger.error(f"output type {output_type} could not be found. If a custom function was given, please check that a corresponding definition was also given.")
                    continue


                eqns = eqns + [eqn]
                error = get_symbolic_expr_error(data, eqn, logger=logger)
                errors = errors + [error]
                expr = parse_expr(eqn)
                is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
                numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if
                                is_atomic_number(subexpression)]
                compl = 0
                for j in numbers_expr:
                    compl = compl + get_number_DL_snapped(float(j))

                # Add the complexity due to symbols
                n_variables = len(expr.free_symbols)
                n_operations = len(count_ops(expr, visual=True).free_symbols)
                if n_operations != 0 or n_variables != 0:
                    compl = compl + (n_variables + n_operations) * np.log2((n_variables + n_operations))
                #print(f"eq: {eqn} and error: {error} and compl: {compl}")
                complexity = complexity + [compl]
            except Exception as e:
                log_exception(logger, e)
                continue

        for i in range(len(complexity)):
            PA.add(Point(x=complexity[i], y=errors[i], data=eqns[i]))

        # run gradient descent of BF output parameters and add the results to the Pareto plot
        for i in range(len(express)):
            try:
                bf_gd_update = RPN_to_pytorch(data, eqns[i], logger=logger)
                logger.debug(f"gd. Adding ({bf_gd_update[1]}, {bf_gd_update[0]}, {bf_gd_update[2]}) to PA.")
                PA.add(Point(x=bf_gd_update[1], y=bf_gd_update[0], data=bf_gd_update[2]))
                logger.debug(f"PA is now:")
                logger.debug(PA.df())
            except Exception as e:
                log_exception(logger, e)
                return PA

    except Exception as e:
        log_exception(logger, e)
        return PA
    return PA