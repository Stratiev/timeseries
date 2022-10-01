import numpy as np
from time import perf_counter


def time_func(func, kwargs):
    """
    Times a function call to func with key word arguments kwargs.
    """
    t0 = perf_counter()
    output = func(**kwargs)
    t1 = perf_counter()
    return t1-t0, output


def simple_corr_matrix(p, n):
    """
    Returns an nxn correlation matrix, s.t. all diagonal entries are 1 and all off-diagonal entries are p.
    """
    if p > 1 or p < -(1/(n-1)):
        raise ValueError(f"Invalid value of p: {p}. The allowed range is {-(1/(n-1))} <= p <= 1.")
    corr_matrix = np.ones((n, n))*p  + np.diag([(1-p)] * n)
    return corr_matrix
