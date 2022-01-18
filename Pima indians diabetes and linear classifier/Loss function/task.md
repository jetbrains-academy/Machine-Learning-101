When classifying an object, the linear classifier makes a decision based on the [linear](https://en.wikipedia.org/wiki/Linear_function_(calculus)) prediction function, which combines a set of weight coefficients and a feature vector.

If the input feature vector is $\vec {x}$, the output is

$$y=f(\vec {w} \cdot \vec {x}) = f ( \sum_{j} w_{j} x_{j})$$

where $\vec {w}$ is a weight vector and $f$ is the above-mentioned linear function, which converts the product of the two vectors into the desired output (in our case, the answer whether a person has type 2 diabetes).

First, our task is to calibrate the weight vector $\vec {w}$ based on the training sample. Thus, the algorithm will "remember" the contribution of each feature (for example, the age or the body weight index) to the possibility of finding diabetes.

Oftentimes, $f$ is a **threshold function**, which maps $ \vec {w} \cdot \vec {x}$ above a certain threshold to the first class and other values — to the second one.

For our classification task, we can picture the linear classifier as the division of high-dimensionality input data by a hyperplane: all objects on one side of the hyperplane are classified as the presence of diabetes, while all others – as its absence (see the picture in the previous task showing examples with data space dimensionality n = 2 and n = 3).

By **loss** we mean the prediction error. The goal is to minimize the error and receive a most accurate result.

There are [several ways](https://en.wikipedia.org/wiki/Loss_functions_for_classification) of calculating the loss. We will use the `log_loss` and `sigmoid_loss` functions.

The logarithmic loss function:

$$L(M) = \log_2(1 + e^{-M})$$

The sigmoid loss function (aka the sigmoid):

$$L(M) = 2(1 + e^{M})^{-1}$$

### Task

Implement the logarithmic (`log_loss`) and sigmoid (`sigmoid_loss`) loss functions. The loss function needs to accept a vector of *margins* (margin is a value that characterizes how close the classified object is to the boundary between two classes, for more details see the next task) and return a pair of vectors – a vector of loss function values and a vector of its derivatives. For example, if we use
the exponential loss function:

$$L(M) = {M}^{n}$$

    def power_loss(M, n=5):
        return M ** n, n * (M ** (n - 1))

Loss functions are defined in `loss_functions.py`.

<div class="hint">
The derivative of the logarithmic loss function:
$$L(M) = \dfrac{-1}{\log_2(1 + e^{-M})}$$
</div>

<div class="hint">
The derivative of the sigmoid loss function:
$$L(M) = \dfrac{-2 * e^{M}}{{(1 + e^{M})}^{2}}$$
</div>

The work of the loss function will be visualized in the next task after we build the vector of weights $\vec{w}$.