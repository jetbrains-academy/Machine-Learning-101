from node import Node
import pandas as pd


# this function reads data from file, we will need it later
def read_data(path):
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop(labels='type', axis=1)
    return X.to_numpy(), y, X.columns.values


if __name__ == '__main__':
    node = Node(1, 2, [1, 2], [3, 4])
    print(f'{node}\n')




