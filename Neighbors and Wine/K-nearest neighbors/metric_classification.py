import numpy as np


def knn(X_train, y_train, X_test, k, dist):
    def classify_single(x):
        dists = [dist(x, i) for i in X_train]
        indices = np.argpartition(dists, k)[:k]
        return np.argmax(np.bincount(y_train[indices]))

    return [classify_single(x) for x in X_test]
