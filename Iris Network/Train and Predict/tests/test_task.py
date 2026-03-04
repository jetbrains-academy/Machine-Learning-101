import unittest

import numpy as np
from safe_assert import safe_assert_array_equal

from network import NN


class TestCase(unittest.TestCase):
    def test_predict(self):
        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

        y_train = np.array([[0, 1, 1, 0]]).T
        nn = NN(3, 3, 1)

        for i in range(11):
            if i == 10:
                self.fail("train method failed: the network could not learn the training data after 10 attempts")
            try:
                nn.train(X_train, y_train)
                nn_y = nn.predict(X_train)
                safe_assert_array_equal((nn_y > 0.5).astype(int), y_train, "Incorrect predictions after training")
                break
            except:
                continue
