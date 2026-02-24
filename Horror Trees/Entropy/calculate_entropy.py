import numpy as np


def entropy(y):
    # Use numpy.unique to obtain sorted unique elements and their respective counts
    _, counts = np.unique(y, return_counts=True)
    # Calculate the proportion of each class in the whole dataset,
    # returning an array of these probabilities
    p = counts / len(y)
    # Compute the entropy based on the provided formula,
    # utilizing the numpy.log2 function for logarithmic calculations
    return -(p * np.log2(p)).sum()
