import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network import NN
from evaluate import accuracy


# This function reads table data from the csv file and retains only two species for our analysis
# as well as only two kinds of measurements - petal length and width.
def read_data(fpath):
    iris = pd.read_csv(fpath)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2
    iris = iris[iris['species'] != 2]
    return iris[['petal_length', 'petal_width']].values, iris[['species']].values.astype('uint8')


# This function plots the input data using matplotlib.pyplot so that you can visualize the distribution.
def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], s=40, cmap=plt.cm.Spectral)
    plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()


# This function splits the dataset into train set and test set using a provided ratio for the split
def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


if __name__ == '__main__':
    X, y = read_data('iris.csv')
    # Comment the following line after the 'Forward Step' task.
    plot_data(X, y)
    nn = NN(len(X[0]), 5, 1)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
    output = nn.feedforward(X_train)
    print(output)
