# Calculates the complexity of a number to be used for the Pareto frontier after snapping

import numpy as np
from .S_snap import bestApproximation

def get_number_DL_snapped(n):
    epsilon = 1e-10
    n = float(n)
    if np.isnan(n):
        return 1000000
    elif np.abs(n - int(n)) < epsilon:
        return np.log2(1 + abs(int(n)))
    elif np.abs(n - bestApproximation(n,10000)[0]) < epsilon:
        _, numerator, denominator, _ = bestApproximation(n, 10000)
        return np.log2((1 + abs(numerator)) * abs(denominator))
    elif np.abs(n - np.pi) < epsilon:
        return np.log2(1+3)
    else:
        PrecisionFloorLoss = 1e-14
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2



