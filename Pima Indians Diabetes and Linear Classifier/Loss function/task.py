import numpy as np


# The function loads data from a text file with a relative path
# 'fname'
# and returns it as a pair of arrays: features
# and diabetes presence.
def read_data(fname):
    # The genfromtxt method loads data from a text file and splits columns
    # based on the provided delimiter.
    data = np.genfromtxt(fname, delimiter=',')
    # The data is split into X (all columns but the last) and
    # y (the last column).
    X, y = data[:, :-1], data[:, -1]
    # The features are rescaled:
    # X is standardized by centering features around the mean
    # with a unit standard deviation. This means that the mean
    # and standard deviation of the standard scores are 0 and 1, respectively.
    # This procedure is recommended for data that follows a normal distribution.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # A column of -1s is prepended to the left of the X array.
    # It acts as a pseudo-feature that simplifies our vector
    # calculations later on.
    X = np.concatenate((-np.ones(len(X)).reshape(-1, 1), X), axis=1)
    # y is standardized: centered around 0 with a standard deviation of 1.
    y =  -(y * 2 - 1)
    return X, y

if __name__ == '__main__':
    X, y = read_data("pima-indians-diabetes.csv")
