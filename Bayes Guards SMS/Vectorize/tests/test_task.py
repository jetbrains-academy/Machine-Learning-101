import unittest
import numpy as np
from numpy.ma.testutils import assert_array_equal

from vectorize import vectorize


class TestCase(unittest.TestCase):
    def test_dictionary(self):
        X = np.array(['Who let the dogs out?', 'Who, who, who, who?'])
        dictionary, result = vectorize(X)
        keys = {'dogs', 'let', 'out', 'the', 'who'}
        values = {0, 1, 2, 3, 4}
        self.assertEqual(keys, set(dictionary.keys()), msg="Something wrong with the dictionary!")
        self.assertEqual(values, set(dictionary.values()), msg="Something wrong with the dictionary!")

    def test_vectorization(self):
        X = np.array(["Who let the dogs out?", " Who, who, who, who?"])
        dictionary, result = vectorize(X)
        assert_array_equal(np.array([1, 1, 1, 1, 1]), result[0])
        self.assertTrue(4 in result[1], msg="Something wrong with the output matrix! "
                                            "If a word is repeated its occurrences need to be counted!")
