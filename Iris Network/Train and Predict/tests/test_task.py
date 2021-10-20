import unittest

import numpy as np
from numpy.ma.testutils import assert_array_equal

from network import NN


class TestCase(unittest.TestCase):
    def test_predict(self):
        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

        y_train = np.array([[0, 1, 1, 0]]).T
        nn = NN(3, 3, 1)
        nn.train(X_train, y_train)
        nn_y = nn.predict(X_train)
        assert_array_equal((nn_y > 0.5).astype(int), y_train)
