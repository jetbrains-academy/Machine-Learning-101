import numpy as np
from metric_classification import knn
from crossvalidation import loocv
from metrics import precision_recall, print_precision_recall
from distances import euclidean_dist, taxicab_dist


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')

    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    y_euclidean_predicted = knn(X_train, y_train, X_test, 5, euclidean_dist)
    print_precision_recall(precision_recall(y_euclidean_predicted, y_test))

    euclidean_opt = loocv(X_train, y_train, euclidean_dist)
    taxicab_opt = loocv(X_train, y_train, taxicab_dist)

    print("optimal euclidian k = " + str(euclidean_opt))
    print("optimal taxicab k = " + str(taxicab_opt))

    y_euclidean_predicted = knn(X_train, y_train, X_test, euclidean_opt, euclidean_dist)
    print_precision_recall(precision_recall(y_euclidean_predicted, y_test))

    y_taxicab_predicted = knn(X_train, y_train, X_test, taxicab_opt, euclidean_dist)
    print_precision_recall(precision_recall(y_taxicab_predicted, y_test))
