import numpy as np
from distances import euclidean_dist
from metric_classification import knn


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    # A slice with first train_len elements of X will be
    # the training sample for objects.
    X_train = X[indices[:train_len]]
    # A slice with the rest of X will be
    # the testing sample for objects.
    X_test = X[indices[train_len:]]
    # A slice with first train_len of y will be
    # the training sample for classes.
    y_train = y[indices[:train_len]]
    # A slice with the rest of y will be
    # the testing sample for classes.
    y_test = y[indices[train_len:]]
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')
    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    y_predicted = knn(X_train, y_train, X_test, 5, euclidean_dist)
