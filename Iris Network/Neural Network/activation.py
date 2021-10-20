import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    # ReLU:
    # return np.maximum(x, 0)
    # tanh(x):
    # return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
