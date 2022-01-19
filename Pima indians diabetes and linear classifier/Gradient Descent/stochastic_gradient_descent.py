import numpy as np
from loss_functions import sigmoid_loss


class StochasticGradientDescent:
    def __init__(self, *, alpha, loss=sigmoid_loss, k=1, n_iter=100):
        # k is the size of the samples the train sample is split into
        self.k = k
        # n_iter is the number of iteration we suppose would be enough
        self.n_iter = n_iter
        self.alpha = alpha
        self.weights = []
        self.loss = loss

    def fit(self, X, y):
        # The initiation of random weights is similar to the original
        # gradient descent
        n = X.shape[1]
        rng = np.random.default_rng()
        self.weights = rng.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
        errors = []

        q, _ = self.calc_grad(self.weights, X, y)
        # eta value should be in [0,1] range, this one is chosen arbitrary
        eta = 1 / len(y)
        for i in range(self.n_iter):
            batch_index = np.random.choice(len(y), self.k)
            loss, grad = self.calc_grad(self.weights, X[batch_index], y[batch_index])
            q = q * (1 - eta) + loss * eta
            errors.append(q)

            self.weights -= self.alpha * grad

        return errors

    def calc_grad(self, w, X, y):
        M = X.dot(w) * y
        loss, loss_deric = self.loss(M)
        grad = X.T * loss_deric * y
        return loss.mean(), grad.mean(axis=1)

    def predict(self, X):
        # The result format is similar to the original gradient descent
        # prediction function
        return np.sign(X.dot(self.weights))
