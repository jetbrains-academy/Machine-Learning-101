import matplotlib.pyplot
import numpy as np


def read_data(fname):
    data = np.genfromtxt(fname, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # rescale features
    X = np.concatenate((-np.ones(len(X)).reshape(-1, 1), X), axis=1)
    y = -(y * 2 - 1)  # {0, 1} -> {1, -1}
    return X, y


if __name__ == '__main__':
    X, y = read_data("pima-indians-diabetes.csv")

