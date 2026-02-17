import numpy as np


def knn(X_train, y_train, X_test, k, dist):
    # The function will return the class for x based on its neighbors from the X_train
    # sample.
    def classify_single(x):
        # Here we create an array of distances from x to each of the X_train objects.
        dists = [dist(x, i) for i in X_train]
        # This array will contain the indices of k nearest to the x objects.
        indices = np.argpartition(dists, k)[:k]
        # The function returns the most frequent class among those in y_train represented
        # by the indices.
        return np.argmax(np.bincount(y_train[indices]))

    return [classify_single(x) for x in X_test]
