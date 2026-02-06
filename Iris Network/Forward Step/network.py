import numpy as np
from activation import sigmoid
from derivative import sigmoid_derivative

class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.w2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def feedforward(self, X):
        """
        self.layer1 = TODO
        return Result
        """
        pass

    def backward(self, X, y, output, learning_rate=0.01):
        """ TODO:
        delta_l2 = Calculate the error for the output layer
        delta_l1 = Calculate the error for the hidden layer
        self.w2 += Update the weight coefficients of the output layer
        self.w1 += Update the weight coefficients of the hidden layer
        """
        pass

    def train(self, X, y, n_iter=20000):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass
