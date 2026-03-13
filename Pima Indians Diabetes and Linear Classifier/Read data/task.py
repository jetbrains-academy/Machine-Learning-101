import numpy as np


# The function loads data from a text file with a relative path
# 'fname'
# and returns it as a pair of arrays: features
# and diabetes presence.
def read_data(fname):
    # Load data from a CSV file using numpy.genfromtxt.
    data = np.genfromtxt(fname, delimiter=',')
    # The data is split into X (all columns but the last) and
    # y (the last column).
    X, y = data[:, :-1], data[:, -1]
    # Normalize features: for each column subtract its mean
    # and divide by its standard deviation.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Add a column of -1s to the left of X.
    # It acts as a pseudo-feature that simplifies our vector
    # calculations later on.
    X = np.concatenate((-np.ones(len(X)).reshape(-1, 1), X), axis=1)
    # Convert labels from {0,1} to {1,-1}.
    y =  -(y * 2 - 1)
    return X, y


if __name__ == '__main__':
    X, y = read_data("pima-indians-diabetes.csv")
    print(f'Features in X array:', X)
    print(f'Diabetes: ', y)
