from calculate_entropy import entropy

class Predicate:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def divide(self, X, y):
        if isinstance(self.value, int) or isinstance(self.value, float):
            mask = X[:, self.column] >= self.value
        else:
            mask = X[:, self.column] == self.value

        return X[mask], y[mask], X[~mask], y[~mask]

    def information_gain(self, X, y):
        X1, y1, X2, y2 = self.divide(X, y)
        p = float(len(X1)) / len(X)
        gain = entropy(y) - p * entropy(y1) - (1 - p) * entropy(y2)
        return gain