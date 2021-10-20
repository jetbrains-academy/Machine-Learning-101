import unittest
import numpy as np

from network import NN


class TestCase(unittest.TestCase):
    def test_forward_shape(self):
        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

        y_train = np.array([[0, 1, 1, 0]]).T
        nn = NN(3, 3, 1)
        y_predicted = nn.feedforward(X_train)
        self.assertTupleEqual((4, 1), y_predicted.shape, msg="Something wrong with the output!")
        #self.assertEqual(4, y_predicted.shape[0])
        #self.assertEqual(1, y_predicted.shape[1])
    def test_layer_shape(self):
        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

        y_train = np.array([[0, 1, 1, 0]]).T
        nn = NN(3, 3, 1)
        nn.feedforward(X_train)
        self.assertTupleEqual((4, 3), nn.layer1.shape, msg="Something wrong with layer 1!")
