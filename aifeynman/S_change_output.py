import numpy as np
import os
import traceback
from .S_run_bf_polyfit import run_bf_polyfit
from .logging import log_exception


def get_transform(XY, BF_try_time, aritytemplates_path, PA, basis_func, polyfit_deg=3, logger=None, processes=2):

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

    data = np.copy(XY)
    try:
        f = basis_func_definition[basis_func]
    except KeyError:
        logger.warning(f"No definition for basis function {basis_func} was given. Skipping.")
        logger.debug(traceback.format_exc())
        return PA
    try:
        # Transform the target vector using basis_func, then try to fit to the transformed data
        data[:, -1] = f(data[:, -1])
        PA = run_bf_polyfit(data, BF_try_time, aritytemplates_path, PA, polyfit_deg, basis_func, logger, processes)
    except Exception as e:
        log_exception(logger, e)
        return PA
    return PA
