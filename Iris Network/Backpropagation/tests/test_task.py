import unittest

import numpy as np
from network import NN


class TestCase(unittest.TestCase):
    def test_weights_updated(self):
        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

        y_train = np.array([[0, 1, 1, 0]]).T
        nn = NN(3, 3, 1)
        nn_w1_initial = np.copy(nn.w1)
        nn_w2_initial = np.copy(nn.w2)
        y_predicted = nn.feedforward(X_train)
        nn.backward(X_train, y_train, y_predicted)
        self.assertFalse(np.array_equal(nn_w1_initial, nn.w1))
        self.assertFalse(np.array_equal(nn_w2_initial, nn.w2))
