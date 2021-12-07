import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from task import Node
from tree import DecisionTree


def read_data(path):
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop('type', 1)
    return X.to_numpy(), y, X.columns.values


class LabelEncoder:
    def encode(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def decode(self, y):
        return self.classes_[y]


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


if __name__ == '__main__':
    path = "halloween.csv"
    X, y, columns = read_data(path)
    label_encoder = LabelEncoder()
    y = label_encoder.encode(y)

    tree = DecisionTree()
    tree = tree.build(X, y)

    print(label_encoder.decode(tree.predict(X[0])))
    drawtree(tree.root)
