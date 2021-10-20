import numpy as np
from PIL import Image, ImageDraw
import pandas as pd


class LabelEncoder:
    def encode(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def decode(self, y):
        return self.classes_[y]


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -(p * np.log2(p)).sum()


class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


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
        return entropy(y) - p * entropy(y1) - (1 - p) * entropy(y2)


class DecisionTree:
    def build(self, X, y):
        self.root = self.build_subtree(X, y)
        return self

    def build_subtree(self, X, y):
        predicate = self.get_best_predicate(X, y)

        if predicate:
            X1, y1, X2, y2 = predicate.divide(X, y)
            true_branch = self.build_subtree(X1, y1)
            false_branch = self.build_subtree(X2, y2)
            return Node(column=predicate.column, value=predicate.value,
                        true_branch=true_branch, false_branch=false_branch)
        else:
            unique_y = np.unique(y, return_counts=True)
            return unique_y[np.argmax(unique_y[1])][0]

    def get_best_predicate(self, X, y):
        best_predicate = None
        best_gain = 0.0
        column_count = len(X[0])

        for column in range(0, column_count):
            column_values = np.unique(X[:, column])

            for value in column_values:
                predicate = Predicate(column, value)
                gain = predicate.information_gain(X, y)
                if gain > best_gain:
                    best_predicate = predicate
                    best_gain = gain

        return best_predicate

    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        if not isinstance(sub_tree, Node):
            return sub_tree
        else:
            v = x[sub_tree.column]
            if isinstance(v, int) or isinstance(v, float):
                if v >= sub_tree.value:
                    branch = sub_tree.true_branch
                else:
                    branch = sub_tree.false_branch
            else:
                if v == sub_tree.value:
                    branch = sub_tree.true_branch
                else:
                    branch = sub_tree.false_branch
            return self.classify_subtree(x, branch)


def getwidth(tree):
    if not isinstance(tree, Node):
        return 1
    return getwidth(tree.true_branch) + getwidth(tree.false_branch)


def getdepth(tree):
    if not isinstance(tree, Node):
        return 0
    return max(getdepth(tree.true_branch), getdepth(tree.false_branch)) + 1


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if isinstance(tree, Node):
        # Get the width of each branch
        shift = 100
        w1 = getwidth(tree.false_branch) * shift
        w2 = getwidth(tree.true_branch) * shift

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), columns[tree.column] + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + shift), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + shift), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.false_branch, left + w1 / 2, y + shift)
        drawnode(draw, tree.true_branch, right - w2 / 2, y + shift)
    else:
        txt = label_encoder.decode(tree)
        draw.text((x - 20, y), txt, (0, 0, 0))


def read_data(path):
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop('type', 1)
    return X.as_matrix(), y, X.columns.values


if __name__ == '__main__':
    path = "halloween.csv"
    X, y, columns = read_data(path)
    label_encoder = LabelEncoder()
    y = label_encoder.encode(y)

    tree = DecisionTree()
    tree = tree.build(X, y)

    print(label_encoder.decode(tree.predict(X[0])))
    drawtree(tree.root)
