import numpy as np
from metric_classification import knn


def loocv(X_train, y_train, dist):
    def loo(k):
        # This is a counter for incorrectly predicted classes.
        c = 0
        for i in range(len(X_train)):
            # Here, we define the training sample
            # by excluding the object at index i.
            x_train_cur = np.vstack([X_train[:i], X_train[i + 1:]])
            # Here, we define the training sample of classes
            # by excluding the target at index i.
            y_train_cur = np.concatenate((y_train[:i], y_train[i + 1:]))
            # Here, we add a condition to check if the algorithm trained on the samples
            # misclassifies the test case.
            # Note that the test case corresponds to index i in both samples.
            if knn(x_train_cur, y_train_cur, X_train[i:i + 1], k, dist)[0] != y_train[i]:
                c += 1
        return c

    # Here, we construct a list of error counts for each
    # k value. We find the lowest error and return its k. We add 1 to the index,
    # since indexing starts at 0.
    loos = list(map(loo, range(1, len(X_train) - 1)))
    return np.argmin(loos) + 1
