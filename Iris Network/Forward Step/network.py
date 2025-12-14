import numpy as np
from activation import sigmoid
from derivative import sigmoid_derivative

class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.w2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def feedforward(self, X):
        self.layer1 = sigmoid(np.dot(X, self.w1))
        return sigmoid(np.dot(self.layer1, self.w2))
