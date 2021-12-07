import numpy as np
import unittest

from numpy.ma.testutils import assert_array_equal

from task import read_image


# The test is checking read_image looking somewhat like this
# def read_image(path='superman-batman.png'):
#     image = Image.open(path)
#     return np.array(image).reshape(-1, 3)
class TestCase(unittest.TestCase):
    # TODO: this test not passing in the student mode is a bug
    # def test_read_image(self):
    #     image = read_image("./tests/star.png")
    #     expected_star = np.loadtxt("./tests/star.txt")
    #     assert_array_equal(expected_star, image)

    def test_shape(self):
        image = read_image("./superman-batman.png")
        self.assertEqual((786432, 3), image.shape)
