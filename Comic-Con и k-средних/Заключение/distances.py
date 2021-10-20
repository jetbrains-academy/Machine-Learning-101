import numpy as np


def euclidean_distance(A, B):
    return np.sqrt(np.sum(np.square(A - B), axis=1))
