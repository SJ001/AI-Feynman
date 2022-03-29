import numpy as np
import os
import traceback
from .S_run_bf_polyfit import run_bf_polyfit
from .logging import log_exception


def get_transform(pathdir, filename, BF_try_time, BF_ops_file_type, PA, basis_func, polyfit_deg=3, logger=None):
    pathdir_write_to = pathdir + "results/mystery_world_{}/".format(basis_func)
    try:
        os.mkdir(pathdir_write_to)
    except OSError:
        pass
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

    data = np.loadtxt(pathdir + filename)
    try:
        f = basis_func_definition[basis_func]
    except KeyError:
        logger.warning(f"No definition for basis function {basis_func} was given. Skipping.")
        logger.debug(traceback.format_exc())
        return PA
    try:
        data[:,-1] = f(data[:,-1])
        np.savetxt(pathdir_write_to + filename, data)
        PA = run_bf_polyfit(pathdir, pathdir_write_to, filename, BF_try_time, BF_ops_file_type, PA, polyfit_deg, basis_func, logger)
    except Exception as e:
        log_exception(logger, e)
        return PA
    return PA
