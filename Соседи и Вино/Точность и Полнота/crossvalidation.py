import numpy as np
from metric_classification import knn


def loocv(X_train, y_train, dist):
    def loo(k):
        c = 0
        for i in range(len(X_train)):
            x_train_cur = np.vstack([X_train[:i], X_train[i + 1:]])
            y_train_cur = np.concatenate((y_train[:i], y_train[i + 1:]))
            if knn(x_train_cur, y_train_cur, X_train[i:i + 1], k, dist)[0] != y_train[i]:
                c += 1
        return c

    loos = list(map(loo, range(1, len(X_train) - 1)))
    return np.argmin(loos) + 1