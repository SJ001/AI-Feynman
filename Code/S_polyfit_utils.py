import numpy as np
from numpy import linalg, zeros, ones, hstack, asarray
import itertools
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
import os
from sympy import symbols, Add, Mul, S


def basis_vector(n, i):
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

def as_tall(x):
    return x.reshape(x.shape + (1,))


def multipolyfit(xs, y, deg):
    
    y = asarray(y).squeeze()
    rows = y.shape[0]
    xs = asarray(xs)
    try:
        num_covariates = xs.shape[1]
    except:
        num_covariates = 1
        xs = np.reshape(xs,(len(xs),1))

    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))
    
    generators = [basis_vector(num_covariates+1, i) for i in range(num_covariates+1)]
        
    # All combinations of degrees
    powers = map(sum, itertools.combinations_with_replacement(generators, deg))

    # Raise data to specified degree pattern, stack in order
    A = hstack(asarray([as_tall((xs**p).prod(1)) for p in powers]))
    params = lsqr(A, y)[0] # get the best params of the fit
    rms = lsqr(A, y)[4] # get the rms params of the fit
    
    return (params, rms)


def getBest(xs,y,max_deg):
    results = []
    for i in range(0,max_deg+1):
        results = results + [multipolyfit(xs,y,i)]
    results = np.array(results)
    # get the parameters and error of the fit with the lowest rms error
    params = results[np.argmin(results[:,1:])][0]
    error = results[np.argmin(results[:,1:])][1]
    deg = np.argmin(results[:,1:])
    return (params, error, deg)

