import ctypes
from numpy import ctypeslib as npct
from .logging import log_exception
from cython_wrapper import bf_search



#def brute_force(data, BF_try_time, aritytemplates_path, BF_ops_file_type, results_path, sep_type="*", sigma=10, band=0.01, logger=None):
def brute_force(data, BF_try_time, aritytemplates_path, results_path, sep_type="*", sigma=10, band=0.01, logger=None):
    try:
        bf_search(data, data.shape[0], data.shape[1], BF_try_time, sep_type, float(band), float(sigma), results_path, aritytemplates_path)

    except Exception as e:
        log_exception(logger, e)
