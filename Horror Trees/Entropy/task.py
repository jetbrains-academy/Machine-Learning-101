from calculate_entropy import entropy
import pandas as pd
from node import Node

def read_data(path):
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop('type', 1)
    return X.to_numpy(), y, X.columns.values


if __name__ == '__main__':
    node = Node(1, 2, [1, 2], [3, 4])
    print(f'{node}\n')

    X, y, columns = read_data("halloween.csv")
    print(f'dataset entropy: {entropy(y)}\n')
