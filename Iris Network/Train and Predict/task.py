import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network import NN
from evaluate import accuracy


def read_data(fpath):
    iris = pd.read_csv(fpath)
    iris['species'] = iris['species'].map({
        'virginica': 0,
        'versicolor': 1,
        'setosa': 2
    }).astype('uint8')
    iris = iris[iris['species'] != 2]

    return (
        iris[['petal_length', 'petal_width']].values,
        iris[['species']].values
    )


def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], s=40, cmap=plt.cm.Spectral)
    plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()


def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]



if __name__ == '__main__':
    X, y = read_data('iris.csv')
    # comment the following line if you don't need the plot anymore
    plot_data(X, y)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
    nn = NN(len(X[0]), 5, 1)
    output = nn.feedforward(X_train)
    print(output)
    print(f'w1 before backward propagation: \n{nn.w1} \nw2 before backward propagation:\n{nn.w2}')
    nn.backward(X_train, y_train, output)
    print(f'w1 after backward propagation: \n{nn.w1} \nw2 after backward propagation:\n{nn.w2}')
    nn.train(X_train, y_train)
    print(f'w1 after training: \n{nn.w1} \nw2 after training:\n{nn.w2}')

