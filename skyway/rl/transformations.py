import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cubic_edge_sensitivity(x):
    return .5 * (2*x-1) ** 3 + .5