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
        # the algorithm will iterate exactly n_iter times
        for i in range(self.n_iter):
            # we need to generate a batch of random k indices from the sample
            # np.random.choice is a perfect fit for this
            batch_index = # generate the batch
            loss, grad = # calculate the gradient using the current weights, X and y batches
            # here we evaluate the quality functional
            q = q * (1 - eta) + loss * eta
            errors.append(q)

            # Updating the weights
            self.weights -= self.alpha * grad

        return errors

    def calc_grad(self, w, X, y):
        # margin indicates how "deep" is an object inside its class. The less the margin
        # the closer it is to the class boundary on the hyperplane
        margin = X.dot(w) * y
        loss, loss_derivative = self.loss(margin)
        # The gradient could be evaluated as a multiplication product of
        # X transposed, loss derivative and y
        grad = # initialize it here
        # the function returns the mean values of loss and grad - for evaluating error and
        # updating the weights accordingly
        return loss.mean(), grad.mean(axis=1)

    def predict(self, X):
        # The result format is similar to the original gradient descent
        # prediction function
        return np.sign(X.dot(self.weights))
