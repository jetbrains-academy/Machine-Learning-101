import numpy as np


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def taxicab_dist(x, y):
    return np.abs(x - y).sum()
