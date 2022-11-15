import logging
import numpy as np
import pandas as pd
from sympy import symbols

from .get_pareto import ParetoSet, Point
from .logging import log_exception
from .dimensionalAnalysis import dimensional_analysis
from .S_final_gd import final_gd
from .S_run_aifeynman import run_AI_all
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .S_add_snap_expr_on_pareto import add_snap_expr_on_pareto
from .S_get_number_DL_snapped import get_number_DL_snapped


class SymbolicRegressor:
    def __init__(self, BF_try_time, polyfit_deg=4, NN_epochs=4000, test_percentage=20, debug=False, bases=None, custom_funcs={}, num_processes=2, disable_progressbar=False):
        """
        Initializes SymbolicRegressor class

        Parameters:
            bases : basis functions
            BF_try_time : Duration of brute force method before timeout
            polyfit_deg : degree of polynomial to fit
            NN_epochs : number of epochs to train neural network
            test_percentage : percentage of data to use for test instead of train of neural network
            debug : if True, logs additional information for debug purposes.
            custom_funcs : dict where key is name of basis function and value is function which transforms an np.ndarray
            num_processes : the number of processes to run in parallel. for ideal performance this number should not be larger than the number of virtual cores available on your system
        """

        self.result_before_snap = None
        self.result_first_snap = None
        self.fit_result = None
        if not isinstance(BF_try_time, int):
            raise TypeError(f"BF_try_time={BF_try_time} should be integer.")
        # TODO: implement BF_ops_file_type passing to bf module
        #if not isinstance(BF_ops_file_type, str):
        #    raise TypeError(f"BF_ops_file_type={BF_ops_file_type} is not a valid file path.")
        if not isinstance(polyfit_deg, int):
            raise TypeError(f"polyfit_deg={polyfit_deg} should be integer.")
        if not isinstance(NN_epochs, int):
            raise TypeError(f"NN_epochs={NN_epochs} is not an integer.")
        if not isinstance(test_percentage, int):
            raise TypeError(f"test_percentage={test_percentage} should be an integer.")
        if not isinstance(debug, bool):
            raise TypeError(f"debug={debug} should be a bool.")
        if not isinstance(custom_funcs, dict):
            raise TypeError(f"custom_funcs={custom_funcs} should be a dictionary.")
        if not isinstance(num_processes, int):
            raise TypeError(f"num_processes={num_processes} should be an integer.")
        if not isinstance(disable_progressbar, bool):
            raise TypeError(f"disable_progressbar={num_processes} should be a bool.")

        if not bases:
            self.bases = ["", "acos", "asin", "atan", "cos", "exp", "inverse", "log", "sin", "sqrt", "squared", "tan"]
        else:
            for base in bases:
                if base not in ["", "acos", "asin", "atan", "cos", "exp", "inverse", "log", "sin", "sqrt", "squared", "tan"]:
                    raise ValueError(f'"{base}" is invalid basis function.')
            self.bases = bases

        self.PA = ParetoSet()
        self.BF_try_time = BF_try_time
        #self.BF_ops_file_type = BF_ops_file_type
        self.polyfit_deg = polyfit_deg
        self.NN_epochs = NN_epochs
        self.test_percentage = test_percentage
        self.debug = debug
        self.num_processes = num_processes

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

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
        self.overall_factor = ""
        self.disable_progressbar = disable_progressbar

        # Append custom functions to the dictionary
        for func, definition in custom_funcs.items():
            self.basis_definitions[func] = definition

    def fit(self, X, Y, units_data=None, vars_name=None):
        """
        Runs symbolic regression on the data X with target Y.

        Parameters:
            X : numpy.ndarray
            Y : numpy.ndarray
            units_data : pandas.DataFrame or str
            vars_name : list. name of variables, should equal variable names found in units_data
        """
        if vars_name is None:
            vars_name = []

        if not isinstance(vars_name, list):
            raise TypeError(f"vars_name={vars_name} should be a list.")
        logger = self.logger
        self.vars_name = vars_name
        self.overall_factor = ""

        try:
            if len(self.vars_name) != 0:
                logger.info("Running dimensional analysis with passed vars_name.")
                if isinstance(units_data, str):
                    units_pd = pd.read_excel(units_data)
                elif isinstance(units_data, pd.DataFrame):
                    units_pd = units_data
                elif units_data is None:
                    units_pd = pd.read_excel("units.xlsx")
                else:
                    raise TypeError("Argument units_data should be str or pandas.DataFrame.")
                X, Y, self.vars_name, self.overall_factor = self.dimensionalAnalysis(X, Y, self.vars_name, units_pd)
                logger.info(f"Dimensional reduction succeeded. AI Feynman will be run on {len(self.vars_name)} variables: {self.vars_name} with overall factor {self.overall_factor}.")
            else:
                logger.info("No vars_name was given. Running without dimensional analysis.")
        except Exception as e:
            logger.warning("Dimensional analysis could not be completed. See debug log for more information.")
            log_exception(logger, e)
        if X.size == 0:
            self.logger.debug("All variables reduced in dimensional reduction. Finding the correct coefficient by the mean of the remaining column.")
            c = sum(Y)[0] / Y.size
            error = get_symbolic_expr_error(Y, str(c), self.logger)
            compl = get_number_DL_snapped(float(c))
            self.PA.add(Point(x=compl, y=error, data=str(c)))
            XY = Y
            XY_train = Y
            XY_test = Y
        else:
            # Split the data into train and test set
            XY = np.column_stack((X, Y))[:,:]

            sep_idx = np.random.permutation(len(XY))

            XY_train = XY[sep_idx[0:(100 - self.test_percentage) * len(XY) // 100]]
            XY_test = XY[sep_idx[self.test_percentage * len(XY) // 100:len(XY)]]

            # Run the code on the train data
            self.PA = run_AI_all(XY_train, self.BF_try_time, self.polyfit_deg, self.NN_epochs, PA=self.PA,
                            logger=self.logger, bases=self.bases, processes=self.num_processes, disable_progressbar=self.disable_progressbar)

        PA_list = self.PA.get_pareto_points()
        print("Post-processing the results...")
        self.result_before_snap = PA_list

        # TODO: implement and uncomment?
        ''' 
        # Run bf snap on the resulted equations
        for i in range(len(PA_list)):
            try:
                self.PA = add_bf_on_numbers_on_pareto(pathdir,filename,self.PA,PA_list[i][-1])
            except:
                continue
        PA_list = self.PA.get_pareto_points()
        '''

        # Run zero, integer and rational snap on the resulted equations
        for j in range(len(PA_list)):
            self.PA = add_snap_expr_on_pareto(XY, PA_list[j][-1], self.PA, "", logger=logger)

        PA_list = self.PA.get_pareto_points()
        self.result_first_snap = PA_list
        # Run gradient descent on the data one more time
        for i in range(len(PA_list)):
            try:
                dt = XY
                gd_update = final_gd(dt, PA_list[i][-1], logger=logger)
                self.PA.add(Point(x=gd_update[1], y=gd_update[0], data=gd_update[2]))
                logger.debug(f"Adding ({gd_update[1]}, {gd_update[0]}, {gd_update[2]}) to PA, which is now:\n{self.PA.df()}")
            except Exception as e:
                log_exception(logger, e)
                continue

        PA_list = self.PA.get_pareto_points()
        for j in range(len(PA_list)):
            self.PA = add_snap_expr_on_pareto(XY, PA_list[j][-1], self.PA, self.overall_factor, self.vars_name, logger=logger)

        list_dt = self.PA.get_pareto_points()
        data_file_len = len(XY)
        log_err = []
        log_err_all = []
        for i in range(len(list_dt)):
            log_err = log_err + [np.log2(float(list_dt[i][1]))]
            log_err_all = log_err_all + [data_file_len * np.log2(float(list_dt[i][1]))]

        # Try the found expressions on the test data
        if self.overall_factor == "" and XY_test.size != 0:
            test_errors = []
            for i in range(len(list_dt)):
                test_errors = test_errors + [
                    get_symbolic_expr_error(XY_test, str(list_dt[i][-1]), logger=logger)]
            save_data = [[test_errors[i], log_err[i], log_err_all[i]] + list_dt[i] for i in range(len(list_dt))]

        else:
            save_data = [[log_err[i], log_err_all[i]] + list_dt[i] for i in range(len(list_dt))]

        for i in range(len(save_data)):
            try:
                expr = save_data[i][-1]
                if self.vars_name != []:
                    for j, var in enumerate(self.vars_name):
                        def_name = "x" + str(j)
                        expr = expr.replace(def_name, "(" + str(var) + ")")
                if self.overall_factor != '':
                    expr = self.overall_factor + f"*({expr})"
                save_data[i][-1] = expr
            except Exception as e:
                log_exception(logger, e)
        self.fit_result = save_data
        logger.info(f"Results:")
        for candidate in self.fit_result:
            logger.info(candidate)
        return self.PA, save_data

    def fit_to_file(self, data_path, units_data_path=None, vars_name=None):
        data = np.loadtxt(data_path)
        X = data[:, :-1]
        Y = data[:, -1:]

        self.logger.debug(f"Loaded dataset with {X.shape[1]} independent variables over {X.shape[0]} datapoints.")

        if units_data_path is not None:
            units_data = pd.read_csv(units_data_path)
            PA, save_data = self.fit(X, Y, units_data, vars_name)
        else:
            PA, save_data = self.fit(X, Y)
        return PA, save_data

    def get_frontier(self):
        return self.PA

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

        units = {}
        for i in range(len(units_pd["Variable"])):
            try:
                val = [units_pd["m"][i], units_pd["s"][i], units_pd["kg"][i], units_pd["T"][i], units_pd["V"][i], units_pd["cd"][i]]
            except KeyError:
                val = [units_pd["m"][i], units_pd["s"][i], units_pd["kg"][i], units_pd["T"][i], units_pd["V"][i]]
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
            return dimless_data, Y, file_sym_list, ""
        else:
            diman_result = dimensional_analysis(input, output, units)
            # get the symbolic form of the solved part
            solved_powers = diman_result[0]
            input_sym = symbols(input)
            sol = 1
            for i in range(len(input_sym)):
                sol = sol * input_sym[i] ** np.round(solved_powers[i], 2)
            #file_sym_list.append(str(sol))
            overall_factor = str(sol)

            # get the symbolic form of the unsolved part
            unsolved_powers = diman_result[1]

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
            func = np.array(func, dtype=np.float64)

            # get the new variables needed
            new_vars = []
            for i in range(len(dimensional_analysis(input, output, units)[1])):
                nv = 1
                for j in range(len(input)):
                    nv = nv * vars()[input[j]] ** dimensional_analysis(input, output, units)[1][i][j]
                new_vars = new_vars + [nv]

            new_vars = np.array(new_vars, dtype=np.float64)

            new_dependent = deps.reshape((deps.shape[0], 1)) / func.reshape((func.shape[0], 1))

            #all_variables = np.vstack((new_vars, new_dependent)).T
            return new_vars.T, new_dependent, file_sym_list, overall_factor



