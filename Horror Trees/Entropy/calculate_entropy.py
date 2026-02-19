import numpy as np


def entropy(y):
    # Use numpy.unique to obtain sorted unique elements and their counts
    _, counts = np.unique(y, return_counts=True)
    # Calculate the proportion of each class in the whole dataset
    # the return value should be an array of proportions
    p = counts / len(y)
    # Calculate and return entropy using the formula from the task
    # logarithm can be calculated using the numpy.log2 function
    return -(p * np.log2(p)).sum()
