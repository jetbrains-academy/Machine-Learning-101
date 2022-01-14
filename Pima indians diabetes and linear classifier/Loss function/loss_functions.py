import numpy as np


def log_loss(M):
    return np.log2(1 + np.exp(-M)), -1 / (np.log(2)*(1 + np.exp(M)))


def sigmoid_loss(M):
    return 2 / (1 + np.exp(M)), -2 * np.exp(M) / (np.exp(M) + 1) ** 2
