import numpy as np


def knn(X_train, y_train, X_test, k, dist):
    # The function predicts the class for x based on its neighbors within the X_train
    # sample.
    def classify_single(x):
        # Here, we create an array of distances between x and every object in X_train.
        dists = [dist(x, i) for i in X_train]
        # This array will contain the indices of the k nearest objects to x.
        indices = np.argpartition(dists, k)[:k]
        # The function returns the most frequent class among the labels in y_train identified
        # by the indices.
        return np.argmax(np.bincount(y_train[indices]))

    return [classify_single(x) for x in X_test]
