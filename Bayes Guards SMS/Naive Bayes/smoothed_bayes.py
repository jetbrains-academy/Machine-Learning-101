import numpy as np
from vectorize import *

class SmoothedNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.unique_classes = np.unique(y)

        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        self.classes_prior = # TODO
        self.classes_words_count = # TODO
        self.likelihood = # TODO

        # TODO

    def predict(self, X):
        # TODO
        pass

    def score(self, X, y):
        # TODO
        pass