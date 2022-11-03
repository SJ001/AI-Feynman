# distutils: language=c++
# cython: language_level=3

from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "bruteforce.hpp":
    cdef int bruteforce(double *data, int n_data, int n_input, int duration, char operation_type, float bitmargin, float nu, string save_path, string arity2templates_file)

cdef extern from "bruteforce.hpp":
    cdef int bf_gradient(double *data, int n_data, int n_input, int duration, float bitmargin, float nu, string save_path, string arity2templates_file)


def bf_search(data, n_data, n_input, duration, operation_type, bitmargin, nu, save_path, arity2templates_file):
    cdef np.ndarray[double,mode="c",ndim=2] cpp_data = np.ascontiguousarray(data, dtype=np.float64)
    cdef string cpp_save_path = save_path.encode('UTF-8')
    cdef string cpp_arity2templates_file = arity2templates_file.encode('UTF-8')
    cdef char cpp_operation
    if operation_type == "+":
        cpp_operation = b'+'
    else:
        cpp_operation = b'*'
    cdef float cpp_bitmargin = bitmargin
    cdef float cpp_nu = nu
    cdef int cpp_n_data = n_data
    cdef int cpp_n_input = n_input
    cdef int cpp_duration = duration
    i = bruteforce(&cpp_data[0,0], cpp_n_data, cpp_n_input, cpp_duration, cpp_operation, cpp_bitmargin, cpp_nu, cpp_save_path, cpp_arity2templates_file)
    return i


def bf_grad_search(data, n_data, n_input, duration, bitmargin, nu, save_path, arity2templates_file):
    cdef np.ndarray[double,mode="c",ndim=2] cpp_data = np.ascontiguousarray(data, dtype=np.float64)
    cdef string cpp_save_path = save_path.encode('UTF-8')
    cdef string cpp_arity2templates_file = arity2templates_file.encode('UTF-8')
    cdef float cpp_bitmargin = bitmargin
    cdef float cpp_nu = nu
    cdef int cpp_n_data = n_data
    cdef int cpp_n_input = n_input
    cdef int cpp_duration = duration
    i = bf_gradient(&cpp_data[0,0], cpp_n_data, cpp_n_input, cpp_duration, cpp_bitmargin, cpp_nu, cpp_save_path, cpp_arity2templates_file)
    return i