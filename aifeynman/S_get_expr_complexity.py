import numpy as np
from .S_snap import bestApproximation
from .S_get_number_DL_snapped import get_number_DL_snapped
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops


def get_expr_complexity(expr):
    expr = parse_expr(expr)
    compl = 0

    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]

    for j in numbers_expr:
        try:
            compl = compl + get_number_DL_snapped(float(j))
        except:
            compl = compl + 1000000

    n_variables = len(expr.free_symbols)
    n_operations = len(count_ops(expr,visual=True).free_symbols)

    if n_operations!=0 or n_variables!=0:
        compl = compl + (n_variables+n_operations)*np.log2((n_variables+n_operations))

    return compl
