The problem with the previously discussed algorithm is that in order to find a new approximation of the weight vector, we need to find the gradient for each element in the sample, which may significantly slow down the algorithm. The stochastic gradient descent (**SGD**) calculates the gradient for each update using only one (randomly chosen) training object $x_i$.
The idea is that the gradient calculated this way is a [stochastic](https://en.wikipedia.org/wiki/Stochastic_process) approximation of the gradient calculated from the whole training sample.
Thus, calculating each update becomes much easier, while the algorithm will mainly proceed in the same direction if we look at sum total of multiple updates.

**Stochastic mini-batch gradient descent** calculates the gradient for each small group of objects from the training sample data.

First, the training sample is divided into small batches (for example, with `k` elements in each). Updating takes place for each batch. Depending on the task, `k` takes values from 30 to 500.
### Task

Implement the method of stochastic gradient descent for training the linear classifier.

Just like in `GradientDescent`, the `fit` method has to
return the values of the cost function $Q$ at each iteration.
To estimate $Q$ at each iteration, use the following formula:
$$Q = (1 − \eta)Q + \eta L_i$$

where $L_i$ is the value of the loss function at the given iteration and $\eta \in [0, 1]$, used for calculating the estimate of $Q$, may be any value, for example,
`1 / len(X)`. As the value of $Q$ is not stable, it should not be used
to detect convergence. As we saw in the previous task, an inappropriate selection of parameters won't yield the desired result.

Instead, we suggest that you use an "optimistic strategy": do exactly `n_iter`
iterations and hope that it will be enough for the stochastic gradient descent
to converge.

The `k` parameter defines the size of the random subset of
`X`, which is used to calculate the gradient.

In the `stohastic_gradient-descent.py` file, you can find a template for implementation. Note that the function for calculating the gradient is provided separately (`calc_grad(self, w, X, y)`), the $\eta$ parameter is added, and – most importantly – in the `fit` function, the `while` cycle is substituted by `for i in range(self.n_iter)`. This solves the problem of `Q` instability; however, it may result in a worse approximation of the result. Such a trade-off between the working time, reliability, and stability often occurs in machine learning algorithm settings.

For output visualization, we need to modify the `plot_classification(X, y)` function:
```python
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8)
    n_iter = 5000
    for loss in [sigmoid_loss, log_loss]:
        for k in [1, 10, 50]:
            plt.clf()
            for alpha, color in zip([1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                                    ["red", "blue", "green", "magenta", "yellow", "cyan"]):
                gd = StochasticGradientDescent(alpha=alpha, k=k, n_iter=n_iter)
```
We've added the number of iterations `n_iter = 5000` and one more internal loop to see the results of work with `k = 1`, `k = 10`, and `k = 50`. To save the image for each chosen batch size, we also need to edit the lines defining the image names:
```python
plt.title("SGD({}, k={})".format(loss.__name__, k))
plt.legend()
plt.savefig("sdg-{}-{}.png".format(loss.__name__, k))
```

Besides, you will need to modify several lines in other files:

*utils.py*:
```python
from stochastic_gradient_descent import StochasticGradientDescent
```
*task.py*:
```python
from utils import plot_classification
```

*main in task.py*:
```python
plot_classification(X, y)
```

Then, run `task.py`. You will see the visualizations of the algorithm work for different `k` with different `loss` functions in *Course View* on the left.

Try changing the `n_iter` and `k` parameters and see how it affects the error rate and the working time of the algorithm.

The result of the algorithm's work will be a preset system with a weight vector, which allows us, with a certain degree of probability, to make a prediction based on the anamnesis data and decide whether a person may have type 2 diabetes. The accuracy of the system depends on the parameters set for training the algorithm.