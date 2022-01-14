import numpy as np
import matplotlib.pyplot as plt
from loss_functions import log_loss, sigmoid_loss
from precision_recall import print_precision_recall, precision_recall
from gradient_descent import GradientDescent
from stochastic_gradient_descent import StochasticGradientDescent


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

    n_iter = 5000

    for loss in [sigmoid_loss, log_loss]:
        for k in [1, 10, 50]:
            plt.clf()
            for alpha, color in zip([1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                                    ["red", "blue", "green", "magenta", "yellow", "cyan"]):
                gd = StochasticGradientDescent(alpha=alpha, k=k, n_iter=n_iter)
                plt.plot(gd.fit(X_train, y_train), label=str(alpha), color=color, alpha=0.7,
                         linewidth=1)
                print("SGD({}, k={}, alpha={})".format(loss.__name__, k, alpha))
                print_precision_recall(precision_recall(gd.predict(X_test), y_test))
                print(gd.weights.tolist())
            plt.ylim((plt.ylim()[0], min(1.5, plt.ylim()[1])))
            plt.title("SGD({}, k={})".format(loss.__name__, k))
            plt.legend()
            plt.savefig("sdg-{}-{}.png".format(loss.__name__, k))
