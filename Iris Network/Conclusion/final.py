import numpy as np
import pandas as pd
from network import NN
from evaluate import accuracy


def read_data(fpath):
    iris = pd.read_csv(fpath)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2
    iris = iris[iris['species'] != 2]
    return iris[['petal_length', 'petal_width']].values, iris[['species']].values.astype('uint8')


def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


if __name__ == '__main__':
    X, y = read_data('iris.csv')
    trainX, trainY, testX, testY = train_test_split(X, y, 0.7)
    nn = NN(len(X[0]), 5, 1)
    nn.train(trainX, trainY)
    print(accuracy(nn, testX, testY))
