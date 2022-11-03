from .logging import log_exception
import ctypes
from numpy import ctypeslib as npct
from cython_wrapper import bf_grad_search
import numpy as np


def brute_force_comp(data, results_path, aritytemplates_path, BF_try_time, sigma=10, band=0, logger=None):
    try:
        '''
        try:
            # using .load_library() without the file extension should work cross-platform?
            # https://numpy.org/devdocs/reference/routines.ctypeslib.html
            # TODO: SWAP HARDCODED MODULE LINK
            
            cpp_bf_module = npct.load_library("main", cpp_module_path)
        except OSError:
            raise OSError(
                f"Brute force c++ module could not be loaded from directory {cpp_module_path}. Make sure that the module was compiled properly.")
            
        # configure c types
        cpp_bf_module.bf_gradient.argtypes = [npct.ndpointer(dtype=ctypes.c_float, ndim=2),
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_float,
                                              ctypes.c_float,
                                              ctypes.c_wchar_p,
                                              ctypes.c_wchar_p
                                              ]

        data_ctypes = data.astype(ctypes.c_float)
        sigma_ctypes = ctypes.c_float(sigma)
        band_ctypes = ctypes.c_float(band)
        try:
            cpp_bf_module.bf_gradient(
                data_ctypes,
                data.shape[0],
                data.shape[1],
                BF_try_time,
                band_ctypes,
                sigma_ctypes,
                results_path,
                aritytemplates_path
            )
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Propagated from c++ subprocess.")
        '''
        bf_grad_search(data, data.shape[0], data.shape[1], BF_try_time, float(band), float(sigma), results_path, aritytemplates_path)
    except Exception as e:
        log_exception(logger, e)
