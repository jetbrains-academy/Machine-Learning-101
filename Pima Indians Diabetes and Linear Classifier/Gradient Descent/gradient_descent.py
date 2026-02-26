import numpy as np
from loss_functions import sigmoid_loss


class GradientDescent:
    # Weights are randomly initialized at the start of the fitting process.
    # The default loss function is sigmoid, and the default threshold is set to 0.01.
    def __init__(self, *, alpha, threshold=1e-2, loss=sigmoid_loss):
        self.weights = []
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss

    # This method fits the weights using the training sample X.
    def fit(self, X, y):
        # Set n to the number of objects in X.
        n = X.shape[1]
        # Here we use a sample drawn from a uniform distribution:
        # the low boundary of the interval is set to -1 / (2 * n),
        # the high boundary of the interval is set to 1 / (2 * n).
        # The sample is of size n.
        rng = np.random.default_rng()
        self.weights = rng.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
        # At the moment, there are no errors.
        errors = []

        # The algorithm stops once the threshold is reached.
        while True:
            M = X.dot(self.weights) * y
            loss, derivative =  self.loss(M)

            # The gradient is the sum along axis 0 of the following array:
            # the product of transposed X and y (X.T * y)
            # multiplied by the transposed derivative (derivative.T)
            # with the final result transposed again (.T).
            grad_q =  np.sum((derivative.T * (X.T * y)).T, axis=0)
            new_weights = self.weights - self.alpha * grad_q

            # Record the total loss.
            errors.append(np.sum(loss))
            # If the new_weights are close enough to the old ones, the fitting process is complete.
            if np.linalg.norm(self.weights - new_weights) < self.threshold:
                break
            # Update the stored weights before the next iteration.
            self.weights =  new_weights
        # The method returns the loss history recorded during the fitting process.
        return errors

    def predict(self, X):
        # Since there are only two classes, we can return the np.sign of the
        # dot product between X and the weights.
        return np.sign(X.dot(self.weights))
