# Calculates the complexity of a number to be used for the Pareto frontier

import numpy as np 

def get_number_DL(n):
    epsilon = 1e-10
    # check if integer
    if np.isnan(n):
        return 1000000
    elif np.abs(n - int(n)) < epsilon:
        return np.log2(1+abs(n))
    elif np.abs(n - np.pi) < epsilon:
        return np.log2(1+3)
    # check if real
    else:
        PrecisionFloorLoss = 1e-14
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2

