import numpy as np
import codecs
from bayes import NaiveBayes
from vectorize import *

def test_train_split(X, y, ratio=0.8):
    mask = np.random.uniform(size=len(y)) < ratio
    return X[mask], y[mask], X[~mask], y[~mask]


def read_data(path):
    file = codecs.open(path, encoding='latin1')
    text = np.loadtxt(file, dtype=np.bytes_, delimiter='\t', unpack=True)
    return np.char.decode(text)


if __name__ == '__main__':
    y, X = read_data('spam.txt')
    X_train, y_train, X_test, y_test = test_train_split(X, y)

    index_dict, vectorization = vectorize(X_train)
    print('Last 10 items of your index dictionary: ', dict(list(index_dict.items())[-10:]))
    print('Vectorization array dimensions: ', vectorization.shape)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    print('Total number of words in each class: ', nb.classes_words_count)
    print('Class prior probabilities: ', nb.classes_prior)
    print('Relative word frequencies for each class: ', nb.likelihood)
