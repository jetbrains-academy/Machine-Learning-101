import numpy as np
import string

def split_by_words(X):
    return np.char.split(np.char.translate(np.char.lower(X), str.maketrans('', '', string.punctuation)))

def vectorize(X):
    X_len = len(X)
    X_split = split_by_words(X)

    uniques = np.unique(np.hstack(X_split))
    index_dict = {}
    for index, word in enumerate(uniques):
        index_dict[word] = index

    vectorization = np.zeros((X_len, len(index_dict)), dtype=np.int64)
    for index, message in enumerate(X_split):
        unique, count = np.unique(message, return_counts=True)
        for i, word in enumerate(unique):
            word_index = index_dict[word]
            vectorization[index, word_index] = count[i]

    return index_dict, vectorization