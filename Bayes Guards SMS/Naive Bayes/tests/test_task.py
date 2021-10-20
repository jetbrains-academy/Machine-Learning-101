import unittest

import numpy as np

from bayes import NaiveBayes


class TestCase(unittest.TestCase):
    def test_classes_words_count(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertTrue(9 in nb.classes_words_count, msg="Improper values in classes_words_count")
        self.assertTrue(11 in nb.classes_words_count, msg="Improper values in classes_words_count")

    def test_classes_prior(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertTrue(0.4 in nb.classes_prior, msg="Improper values in classes_prior")
        self.assertTrue(0.6 in nb.classes_prior, msg="Improper values in classes_prior")

    def test_zero_likelihood(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertTrue(0 in nb.likelihood, msg="Improper values in likelihood")
