from .logging import log_exception
from numpy import ctypeslib as npct
import ctypes


def brute_force_gen_sym(data, results_path, aritytemplates_path, cpp_module_path, BF_try_time, BF_ops_file_type, sigma=10, band=0, logger=None):
    try:
        try:
            # using .load_library() without the file extension should work cross-platform?
            # https://numpy.org/devdocs/reference/routines.ctypeslib.html
            # TODO: CHANGE TO CYTHON WRAPPER IMPORT
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

    except Exception as e:
        log_exception(logger, e)
