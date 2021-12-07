import numpy as np
from vectorize import *

class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.unique_classes = np.unique(y)

        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        self.classes_prior = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.classes_words_count = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.likelihood = np.full((len(self.unique_classes), self.dict_size + 1), self.alpha, dtype=np.float64)

        for i, clazz in enumerate(self.unique_classes):
            y_i_mask = y == clazz
            y_i_sum = np.sum(y_i_mask)
            self.classes_prior[i] = y_i_sum / len(y)
            self.classes_words_count[i] = np.sum(X[y_i_mask])
            self.likelihood[i, :-1] += np.sum(X[y_i_mask], 0)

            denominator = self.classes_words_count[i] + self.alpha * self.dict_size
            self.likelihood[i] = self.likelihood[i] / denominator

    def predict(self, X):
        result = []
        X = split_by_words(X)
        for message in X:
            unique = np.unique(message)
            index_array = np.zeros(unique.shape, dtype=np.int64)

            for i, word in enumerate(unique):
                word_index = self.dictionary[word] if word in self.dictionary else self.dict_size
                index_array[i] = word_index

            log_likelihood = np.log(self.likelihood[:, index_array])
            posterior = np.log(self.classes_prior) + np.sum(log_likelihood, axis=1)
            predicted = self.unique_classes[np.argmax(posterior)]
            result.append(predicted)
        return result

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)