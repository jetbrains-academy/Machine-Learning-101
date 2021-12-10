from node import Node                                                       # 1

import pandas as pd                                                         # 2
from calculate_entropy import entropy

import numpy as np                                                          # 3
from divide import Predicate

from tree import DecisionTree                                               # 5


def read_data(path):                                                        # 2
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop(labels='type', axis=1)
    return X.to_numpy(), y, X.columns.values


if __name__ == '__main__':
    node = Node(1, 2, [1, 2], [3, 4])                                       # 1
    print(f'{node}\n')

    X, y, columns = read_data("halloween.csv")                              # 2
    print(f'dataset entropy: {entropy(y)}\n')

    predicate = Predicate(3, 'clear')                                       # 3
    X = np.array([[1, 1, 1, 'clear'],
                  [2, 2, 2, 'clear'],
                  [3, 3, 3, 'green'],
                  [1, 2, 3, 'black']])
    y = np.array([1, 1, 2, 2])

    X1, y1, X2, y2 = predicate.divide(X, y)
    print(f'Division result: '
          f'\nFirst group labels: {y1} '
          f'\nFirst group objects: {X1} '
          f'\nSecond group labels: {y2} '
          f'\nSecond group objects: {X2}\n')

    print(f'Information Gain: {predicate.information_gain(X, y)}\n')        # 4

    tree = DecisionTree().build(X, y)                                       # 5
    print(f'{tree}\n')

    print(f'Class prediction for object X[0]: {tree.predict(X[0])}\n')      # 6

