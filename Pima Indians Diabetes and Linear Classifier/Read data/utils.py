import numpy as np
import matplotlib.pyplot as plt
from loss_functions import log_loss, sigmoid_loss
from precision_recall import print_precision_recall, precision_recall
from gradient_descent import GradientDescent


# This function will split the data into a train and control samples
# the output is X_train, y_train, X_test, y_test
def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    rng = np.random.default_rng()
    rng.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


def plot_classification(X, y):
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8)
    for loss in [sigmoid_loss, log_loss]:
        plt.clf()
        # You can try adjusting alpha value, but a less one could lead to the
        # exploding gradients. It will be indicated by a RuntimeWarning about encountering
        # an overflow in the loss function's exp
        for alpha, color in zip([1e-6, 1e-5, 1e-4], ["red", "blue", "green"]):
            gd = GradientDescent(alpha=alpha, loss=loss, threshold=1e-5)
            plt.plot(gd.fit(X_train, y_train), label=str(alpha), color=color, alpha=0.7, linewidth=1)
            print("GradientDescent({}, alpha={})".format(loss.__name__, alpha))
            print_precision_recall(precision_recall(gd.predict(X_test), y_test))
            print(gd.weights.tolist())

        plt.title("GradientDescent({})".format(loss.__name__))
        plt.legend()
        plt.savefig("gradient-{}.png".format(loss.__name__))
