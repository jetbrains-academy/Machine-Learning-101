import numpy as np
from activation import sigmoid
from derivative import sigmoid_derivative


# Here we implemented the class NN to store our neural network. It contains two weight
# arrays - w1 and w2, which will be updated as the network learns. You will implement the
# methods of this class that are needed to train the network and to use it for object classification.
class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.w2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def feedforward(self, X):
        # first use the activation function on the product of multiplication
        # of input data and the first array of weights
        self.layer1 = # TODO
        # next, use the activation function on the product of multiplication
        # of the second array of weights and the result of calculation on
        # layer1, and return the result
        return # TODO

    # This method is to be implemented in the task "Backpropagation".
    def backward(self, X, y, output, learning_rate=0.01):
        pass
        # Calculate output layer error as difference of real class labels and output of
        # the network, and multiply it elementwise by the derivative of the sigmoid
        # function with respect to the output
        # l2_delta = # TODO
        # Calculate hidden layer error as product of output layer error and the second
        # array of weights, multiplied elementwise by the derivative of the sigmoid
        # function with respect to the hidden layer output (layer1)
        # l1_delta = # TODO
        # Adjust w2 weight coefficients by adding the matrix product of the hidden layer output (layer1)
        # and the output layer error, multiplied elementwise by the learning rate
        # self.w2 += # TODO
        # Adjust w1 weight coefficients by adding the matrix product of the input data (X)
        # and the hidden layer error, multiplied elementwise by the learning rate
        # self.w1 += # TODO

    # The following two methods are to be implemented in the task "Train and Predict".
    def train(self, X, y, n_iter=20000):
        pass
        # Call the feedforward and backward methods of the data n_iter times
        # to train the network. This method should not return anything
        # for itr in range(n_iter):
            # l2 = # TODO
            # TODO

    def predict(self, X):
        pass
        # This method should feed all the objects from a dataset to the trained network
        # return # TODO
