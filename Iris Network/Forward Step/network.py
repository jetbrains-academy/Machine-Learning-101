import numpy as np
from activation import sigmoid
from derivative import sigmoid_derivative


# Here we implemented the class NN to store our neural network. It contains two weight
# arrays - w1 and w2, which will be updated as the network learns. You will implement the
# class methods required to train the network and perform object classification.
class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.w2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def feedforward(self, X):
        # First, apply the activation function to the product
        # of the input data and the first array of weights
        self.layer1 = sigmoid(np.dot(X, self.w1))
        # Next, apply the activation function to the product
        # of the second array of weights and the result of calculation on
        # layer1, and return the result
        return sigmoid(np.dot(self.layer1, self.w2))
        pass

    def backward(self, X, y, output, learning_rate=0.01):
        """ TODO:
        delta_l2 = Calculate output layer error as the difference between the real class
        labels and the network output, and multiply it element-wise
        by the sigmoid derivative of the output

        delta_l1 = Calculate hidden layer error as the product of the output layer
        error and the second array of weights, multiplied element-wise by
        the sigmoid derivative of the hidden layer output (layer1)

        self.w2 += Adjust w2 weight coefficients by adding the matrix product of
        the hidden layer output (layer1) and the output layer error,
        multiplied element-wise by the learning rate

        self.w1 += Adjust w1 weight coefficients by adding the matrix product of
        the input data (X) and the hidden layer error,
        multiplied element-wise by the learning rate
        """
        pass

    def train(self, X, y, n_iter=20000):
        pass
        # Call the feedforward and backward methods of the data n_iter times
        # to train the network. This method does not return a value
        # for itr in range(n_iter):
            # l2 = # TODO
            # TODO

    def predict(self, X):
        pass
        # This method should feed all the objects from a dataset to the trained network
        # return # TODO
