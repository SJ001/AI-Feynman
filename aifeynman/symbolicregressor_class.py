import logging
import traceback
import os
import numpy as np
import pandas
import pandas as pd
from sympy import symbols

from .get_pareto import ParetoSet, Point
from .logging import log_exception
from .dimensionalAnalysis import dimensional_analysis
from .S_final_gd import final_gd
from .S_run_aifeynman import run_AI_all
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .S_add_snap_expr_on_pareto import add_snap_expr_on_pareto


class SymbolicRegressor:
    def __init__(self, BF_try_time, BF_ops_file_type, polyfit_deg=4, NN_epochs=4000, vars_name=[], test_percentage=20, debug=False, bases=None, custom_funcs={}):
        """
        Initializes SymbolicRegressor class

        Parameters:
            bases : basis functions
            BF_try_time : Duration of brute force method before timeout
            BF_ops_file_type : file containing operations for brute force method
            polyfit_deg : degree of polynomial to fit
            NN_epochs : number of epochs to train neural network
            vars_name : list. name of variables, should equal variable names found in units-file
            test_percentage : percentage of data to use for test instead of train of neural network
            debug : if True, logs additional information for debug purposes.
            custom_funcs : dict where key is name of basis function and value is function which transforms an np.ndarray
        """

        if not isinstance(BF_try_time, int):
            raise TypeError(f"BF_try_time={BF_try_time} should be integer.")
        if not isinstance(BF_ops_file_type, str):
            raise TypeError(f"BF_ops_file_type={BF_ops_file_type} is not a valid file path.")
        if not isinstance(polyfit_deg, int):
            raise TypeError(f"polyfit_deg={polyfit_deg} should be integer.")
        if not isinstance(NN_epochs, int):
            raise TypeError(f"NN_epochs={NN_epochs} is not an integer.")
        if not isinstance(vars_name, list):
            raise TypeError(f"vars_name={vars_name} should be a list.")
        if not isinstance(test_percentage, int):
            raise TypeError(f"test_percentage={test_percentage} should be an integer.")
        if not isinstance(debug, bool):
            raise TypeError(f"debug={debug} should be a bool.")
        if not isinstance(custom_funcs, dict):
            raise TypeError(f"custom_funcs={custom_funcs} should be a dictionary.")

        if not bases:
            self.bases = ["", "acos", "asin", "atan", "cos", "exp", "inverse", "log", "sin", "sqrt", "squared", "tan"]
        else:
            self.bases = bases

        self.PA = ParetoSet()
        self.BF_try_time = BF_try_time
        self.BF_ops_file_type = BF_ops_file_type
        self.polyfit_deg = polyfit_deg
        self.NN_epochs = NN_epochs
        self.vars_name = vars_name
        self.test_percentage = test_percentage
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)

        self.logger = logging.getLogger(__name__)
        self.basis_definitions = {
            "asin": lambda d: np.arcsin(d),
            "acos": lambda d: np.arccos(d),
            "atan": lambda d: np.arctan(d),
            "sin": lambda d: np.sin(d),
            "cos": lambda d: np.cos(d),
            "tan": lambda d: np.tan(d),
            "exp": lambda d: np.exp(d),
            "log": lambda d: np.log(d),
            "inverse": lambda d: 1 / d,
            "sqrt": lambda d: np.sqrt(d),
            "squared": lambda d: d ** 2,
            "": lambda d: d
        }
        self.temp_files = []

        # Append custom functions to the dictionary
        for func, definition in custom_funcs.items():
            self.basis_definitions[func] = definition

    def fit(self, X, Y, units_data=None):
        """
        Runs symbolic regression on the data X with target Y.

        Parameters:
            X : numpy.ndarray
            Y : numpy.ndarray
            units_data : pandas.DataFrame or str
        """

        logger = self.logger

        # If the variable names are passed, do the dimensional analysis first

        filename = "filename_placeholder.txt"
        pathdir = "./temp_pathdir/"

        filename_orig = filename

        if not os.path.isdir(pathdir):
            os.mkdir(pathdir)

        try:
            if self.vars_name:
                print("Running dimensional analysis with passed vars_name.")
                if not units_data:
                    units_pd = pd.read_excel("units.xlsx")
                elif isinstance(units_data, str):
                    units_pd = pd.read_excel(units_data)
                elif isinstance(units_data, pd.DataFrame):
                    units_pd = units_data
                else:
                    raise TypeError("Argument units_data should be str or pandas.DataFrame.")
                X, Y = self.dimensionalAnalysis(X, Y, self.vars_name, units_pd)
                DR_file = filename + "_dim_red_variables.txt"
                filename = filename + "_dim_red"
            else:
                print("No vars_name was given. Running without dimensional analysis.")
                DR_file = ""
        except Exception as e:
            print("Dimensional analysis could not be completed. See debug log for more information.")
            log_exception(logger, e)
            DR_file = ""

        # Split the data into train and test set
        input_data = np.column_stack((X, Y))
        np.savetxt(pathdir+filename, input_data)

        sep_idx = np.random.permutation(len(input_data))

        train_data = input_data[sep_idx[0:(100 - self.test_percentage) * len(input_data) // 100]]
        test_data = input_data[sep_idx[self.test_percentage * len(input_data) // 100:len(input_data)]]

        np.savetxt(pathdir + filename + "_train", train_data)
        if test_data.size != 0:
            np.savetxt(pathdir + filename + "_test", test_data)

        #self.PA = ParetoSet()
        # Run the code on the train data
        self.PA = run_AI_all(pathdir, filename + "_train", self.BF_try_time, self.BF_ops_file_type, self.polyfit_deg, self.NN_epochs, PA=self.PA,
                        logger=self.logger, bases=self.bases)
        PA_list = self.PA.get_pareto_points()

        '''
        # Run bf snap on the resulted equations
        for i in range(len(PA_list)):
            try:
                self.PA = add_bf_on_numbers_on_pareto(pathdir,filename,self.PA,PA_list[i][-1])
            except:
                continue
        PA_list = self.PA.get_pareto_points()
        '''

        np.savetxt("results/solution_before_snap_%s.txt" % filename, PA_list, fmt="%s", delimiter=',')

        # Run zero, integer and rational snap on the resulted equations
        for j in range(len(PA_list)):
            self.PA = add_snap_expr_on_pareto(pathdir, filename, PA_list[j][-1], self.PA, "", logger=logger)

        PA_list = self.PA.get_pareto_points()
        np.savetxt("results/solution_first_snap_%s.txt" % filename, PA_list, fmt="%s", delimiter=',')

        # Run gradient descent on the data one more time
        for i in range(len(PA_list)):
            try:
                dt = np.loadtxt(pathdir + filename)
                gd_update = final_gd(dt, PA_list[i][-1], logger=logger)
                self.PA.add(Point(x=gd_update[1], y=gd_update[0], data=gd_update[2]))
            except Exception as e:
                log_exception(logger, e)
                continue

        PA_list = self.PA.get_pareto_points()
        for j in range(len(PA_list)):
            self.PA = add_snap_expr_on_pareto(pathdir, filename, PA_list[j][-1], self.PA, DR_file, logger=logger)

        list_dt = np.array(PA.get_pareto_points())
        data_file_len = len(np.loadtxt(pathdir + filename))
        log_err = []
        log_err_all = []
        for i in range(len(list_dt)):
            log_err = log_err + [np.log2(float(list_dt[i][1]))]
            log_err_all = log_err_all + [data_file_len * np.log2(float(list_dt[i][1]))]
        log_err = np.array(log_err)
        log_err_all = np.array(log_err_all)

        # Try the found expressions on the test data
        if DR_file == "" and test_data.size != 0:
            test_errors = []
            input_test_data = np.loadtxt(pathdir + filename + "_test")
            for i in range(len(list_dt)):
                test_errors = test_errors + [
                    get_symbolic_expr_error(input_test_data, str(list_dt[i][-1]), logger=logger)]
            test_errors = np.array(test_errors)
            # Save all the data to file
            save_data = np.column_stack((test_errors, log_err, log_err_all, list_dt))
        else:
            save_data = np.column_stack((log_err, log_err_all, list_dt))
        np.savetxt("results/solution_%s" % filename_orig, save_data, fmt="%s", delimiter=',')
        try:
            os.remove(pathdir + filename + "_test")
            os.remove(pathdir + filename + "_train")
        except:
            pass
        return self.PA

    def fit_to_file(self, pathdir, filename, delimiter=' '):
        """
        Runs symbolic regression on file specified by arguments. Wrapper for original run_aifeynman function.

        Parameters:
            pathdir : string
            filename : string
            delimiter : delimiter used in data files
        """
        '''
        run_aifeynman(
            pathdir,
            filename,
            self.BF_try_time,
            self.BF_ops_file_type,
            polyfit_deg=self.polyfit_deg,
            NN_epochs=self.NN_epochs,
            vars_name=self.vars_name,
            test_percentage=self.test_percentage,
            debug=self.debug
            )
        '''
        """
        # If the variable names are passed, do the dimensional analysis first
        filename_orig = filename
        try:
            if self.vars_name != []:
                print("Running dimensional analysis with passed vars_name.")
                dimensionalAnalysis(pathdir, filename, vars_name)
                DR_file = filename + "_dim_red_variables.txt"
                filename = filename + "_dim_red"
            else:
                print("No vars_name was given. Running without dimensional analysis.")
                DR_file = ""
        except Exception as e:
            log_exception(self.logger, e)
            DR_file = ""

        # Split the data into train and test set
        input_data = np.loadtxt(pathdir + filename)
        sep_idx = np.random.permutation(len(input_data))

        train_data = input_data[sep_idx[0:(100 - test_percentage) * len(input_data) // 100]]
        test_data = input_data[sep_idx[test_percentage * len(input_data) // 100:len(input_data)]]

        np.savetxt(pathdir + filename + "_train", train_data)
        if test_data.size != 0:
            np.savetxt(pathdir + filename + "_test", test_data)

        PA = ParetoSet()
        # Run the code on the train data
        self.PA = run_AI_all(pathdir, filename + "_train", BF_try_time, BF_ops_file_type, polyfit_deg, self.NN_epochs, PA=self.PA,
                        logger=logger)
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

        np.savetxt("results/solution_before_snap_%s.txt" % filename, PA_list, fmt="%s", delimiter=',')

        # Run zero, integer and rational snap on the resulted equations
        for j in range(len(PA_list)):
            PA = add_snap_expr_on_pareto(pathdir, filename, PA_list[j][-1], PA, "", logger=logger)

        PA_list = PA.get_pareto_points()
        np.savetxt("results/solution_first_snap_%s.txt" % filename, PA_list, fmt="%s", delimiter=',')

        # Run gradient descent on the data one more time
        for i in range(len(PA_list)):
            try:
                dt = np.loadtxt(pathdir + filename)
                gd_update = final_gd(dt, PA_list[i][-1], logger=logger)
                PA.add(Point(x=gd_update[1], y=gd_update[0], data=gd_update[2]))
            except Exception as e:
                log_exception(logger, e)
                continue

        PA_list = PA.get_pareto_points()
        for j in range(len(PA_list)):
            PA = add_snap_expr_on_pareto(pathdir, filename, PA_list[j][-1], PA, DR_file, logger=logger)

        list_dt = np.array(PA.get_pareto_points())
        data_file_len = len(np.loadtxt(pathdir + filename))
        log_err = []
        log_err_all = []
        for i in range(len(list_dt)):
            log_err = log_err + [np.log2(float(list_dt[i][1]))]
            log_err_all = log_err_all + [data_file_len * np.log2(float(list_dt[i][1]))]
        log_err = np.array(log_err)
        log_err_all = np.array(log_err_all)

        # Try the found expressions on the test data
        if DR_file == "" and test_data.size != 0:
            test_errors = []
            input_test_data = np.loadtxt(pathdir + filename + "_test")
            for i in range(len(list_dt)):
                test_errors = test_errors + [
                    get_symbolic_expr_error(input_test_data, str(list_dt[i][-1]), logger=logger)]
            test_errors = np.array(test_errors)
            # Save all the data to file
            save_data = np.column_stack((test_errors, log_err, log_err_all, list_dt))
        else:
            save_data = np.column_stack((log_err, log_err_all, list_dt))
        np.savetxt("results/solution_%s" % filename_orig, save_data, fmt="%s", delimiter=',')
        try:
            os.remove(pathdir + filename + "_test")
            os.remove(pathdir + filename + "_train")
        except:
            pass
        return PA
        """
        pass

    def get_frontier(self):
        return self.PA

    def cleanup(self, success=True):
        for fl in ["args.dat", "brute_solutions.dat", "brute_constant.dat", "brute_formulas.dat", "qaz.dat"]:
            if fl not in self.temp_files:
                self.temp_files.append(fl)
        for file_path in self.temp_files.copy():
            try:
                os.remove(file_path)
                self.temp_files.remove(file_path)
            except OSError:
                self.logger.info(f"Could not remove file {file_path} during cleanup.")
                self.logger.debug(traceback.format_exc())

    def dimensionalAnalysis(self, X, Y, eq_symbols, units_pd):
        """
        Runs dimensional reduction on the data X with target Y, using units table units_pd and variable names eq_symbols.

        Parameters:
            X : numpy.ndarray
            Y : numpy.ndarray
            eq_symbols: list of strings
            units_pd: pandas.dataFrame

        Returns:
            dimless_data : numpy.ndarray
        """

        # Fix: Return file_sym_list in some way
        units = {}
        for i in range(len(units_pd["Variable"])):
            val = [units_pd["m"][i], units_pd["s"][i], units_pd["kg"][i], units_pd["T"][i], units_pd["V"][i], units_pd["cd"][i]]
            val = np.array(val)
            units[units_pd["Variable"][i]] = val

        dependent_var = eq_symbols[-1]

        file_sym_list = []

        varibs = X.T
        deps = Y

        # get the data in symbolic form and associate the corresponding values to it
        input = []
        for i in range(len(eq_symbols) - 1):
            input = input + [eq_symbols[i]]
            vars()[eq_symbols[i]] = varibs[i]
        output = dependent_var

        # Check if all the independent variables are dimensionless
        ok = 0
        for j in range(len(input)):
            if (units[input[j]].any()):
                ok = 1

        if ok == 0:
            dimless_data = X
            if dimless_data.ndim == 1:
                dimless_data = np.reshape(dimless_data, (1, len(dimless_data)))
                dimless_data = dimless_data.T
            for j in range(len(input)):
                file_sym_list.append(str(input[j]))
            return dimless_data
        else:
            # get the symbolic form of the solved part
            solved_powers = dimensional_analysis(input, output, units)[0]
            input_sym = symbols(input)
            sol = 1
            for i in range(len(input_sym)):
                sol = sol * input_sym[i] ** np.round(solved_powers[i], 2)
            file_sym_list.append(str(sol))

            # get the symbolic form of the unsolved part
            unsolved_powers = dimensional_analysis(input, output, units)[1]

            unsolved = []
            for i in range(len(unsolved_powers)):
                uns = 1
                for j in range(len(unsolved_powers[i])):
                    uns = uns * input_sym[j] ** unsolved_powers[i][j]
                file_sym_list.append(str(uns))
                unsolved = unsolved + [uns]

            # get the discovered part of the function
            func = 1
            for j in range(len(input)):
                func = func * vars()[input[j]] ** dimensional_analysis(input, output, units)[0][j]
            func = np.array(func)

            # get the new variables needed
            new_vars = []
            for i in range(len(dimensional_analysis(input, output, units)[1])):
                nv = 1
                for j in range(len(input)):
                    nv = nv * vars()[input[j]] ** dimensional_analysis(input, output, units)[1][i][j]
                new_vars = new_vars + [nv]

            new_vars = np.array(new_vars)
            new_dependent = deps / func

            if new_vars.size == 0:
                print("All variables are reduced in dimensional analysis. Either the target is a monomial of the "
                      "input variables or incorrect units were given. Dimensional analysis result is therefore "
                      "discarded.")
                return X, Y

            #all_variables = np.vstack((new_vars, new_dependent)).T
            return new_vars.T, new_dependent



