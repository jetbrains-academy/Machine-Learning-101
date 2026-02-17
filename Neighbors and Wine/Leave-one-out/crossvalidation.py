import numpy as np
from metric_classification import knn


def loocv(X_train, y_train, dist):
    def loo(k):
        # This is a counter keeping track of the wrongly determined classes.
        c = 0
        for i in range(len(X_train)):
            # Here we should choose the current training sample of objects
            # including all but the one with index i.
            x_train_cur = np.vstack([X_train[:i], X_train[i + 1:]])
            # Here we should choose the current training sample of classes
            # including all but the one with index i.
            y_train_cur = np.concatenate((y_train[:i], y_train[i + 1:]))
            # Here we should place a condition when the algorithm trained on the samples
            # would be wrong in determining the class of the testing case.
            # Note that the testing case is the one with index i in both samples.
            if knn(x_train_cur, y_train_cur, X_train[i:i + 1], k, dist)[0] != y_train[i]:
                c += 1
        return c

    # Here we construct a list of all error counts mapped to the corresponding
    # k values. We select the lowest one and return its corresponding k. It is shifted
    # by 1, as the array indices start with 0.
    loos = list(map(loo, range(1, len(X_train) - 1)))
    return np.argmin(loos) + 1
