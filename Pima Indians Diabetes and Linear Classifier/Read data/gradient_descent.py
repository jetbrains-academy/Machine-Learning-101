import numpy as np
from loss_functions import sigmoid_loss


class GradientDescent:
    # The weighs will be randomly initiated as the first step of the fitting process.
    # The default loss function is sigmoid, and the default threshold is set to 0.01
    def __init__(self, *, alpha, threshold=1e-2, loss=sigmoid_loss):
        self.weights = []
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss

    # This method is fitting the weights using the training X sample
    def fit(self, X, y):
        # Set n to the number of objects in the
        n = X.shape[1]
        # Here we use a sample from a uniform distribution:
        # low boundary of the interval set to  -1 / (2 * n)
        # high boundary of the interval is set to 1 / (2 * n)
        # The size of the sample is n
        rng = np.random.default_rng()
        self.weights = rng.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
        # At the moment there is no errors
        errors = []

        # The algorithm will be interrupted after reaching the threshold
        while True:
            M = # Set M to a dot product of X and weights multiplied by y
            loss, derivative = # calculate them using loss function of the instance

            # The gradient is a sum along the 0 axis of the following array:
            # transposed X multiplied by y (X.T * y)
            # then multiplied by transposed derivative (derivative.T)
            # with the product of this multiplication transposed again (.T)
            grad_q = # write the expression
            new_weights = # update the weights

            # Updating the errors with the loss
            errors.append(np.sum(loss))
            # If the new_weights are close enough to the old ones, the fitting is complete
            if np.linalg.norm(self.weights - new_weights) < self.threshold:
                break
            # Update the stored weights before the next iteration
            self.weights = # set it to the new ones
        # The method returns the losses stored along the fitting process
        return errors

    def predict(self, X):
        # As we have only two normalized classes we can return the np.sign of the
        # X dot product weights
        return # return the predicted classes
