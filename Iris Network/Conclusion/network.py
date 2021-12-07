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

    def backward(self, X, y, output, learning_rate=0.01):
        l2_delta = (y - output) * sigmoid_derivative(output)
        l1_delta = np.dot(l2_delta, self.w2.T) * sigmoid_derivative(self.layer1)
        self.w2 += (np.dot(self.layer1.T, l2_delta) * learning_rate)
        self.w1 += (np.dot(X.T, l1_delta) * learning_rate)

    def train(self, X, y, n_iter=20000):
        for itr in range(n_iter):
            l2 = self.feedforward(X)
            self.backward(X, y, l2)

    def predict(self, X):
        return self.feedforward(X)
