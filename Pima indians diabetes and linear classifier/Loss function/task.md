When classifying an object, the linear classifier makes a decision based on the [linear](https://ru.wikipedia.org/wiki/%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F_%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F) prediction function, which combines a set of weight coefficients and a feature vector.

If the input feature vector is $\vec {x}$, the output is

$$y=f(\vec {w} \cdot \vec {x}) = f ( \sum_{j} w_{j} x_{j})$$

where $\vec {w}$ is a weight vector and $f$ is the above-mentioned linear function, which converts the product of the two vectors into the desired output (in our case, the answer whether a person has type 2 diabetes).

First, our task is to set the weight vector $\vec {w}$ based on the training sample. Thus, the algorithm will "remember" the contribution of each feature (for example, the age or the body weight index) to the possibility of finding diabetes.

Oftentimes, f is a [threshold function](https://neerc.ifmo.ru/wiki/index.php?title=%D0%9F%D0%BE%D1%80%D0%BE%D0%B3%D0%BE%D0%B2%D0%B0%D1%8F_%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F), which maps $ \vec {w} \cdot \vec {x}$ above a certain threshold to the first class and other values — to the second one.

For our classification task, we can present the linear classifier as the division of high-dimensionality input data by a hyperplane: all objects on one side of the hyperplane are classified as the presence of diabetes, while all others – as its absence (see the picture in the previous task showing examples with data space dimensionality n = 2 and n = 3).

By **loss** we mean the error in the prediction. The goal is to minimize the error and receive a most accurate result.

There are [several ways](https://en.wikipedia.org/wiki/Loss_functions_for_classification) of calculating the loss. We will use the `log_loss` and `sigmoid_loss` functions.

The logarithmic loss function:

$$L(M) = \log_2(1 + e^{-M})$$

The sigmoid loss function (aka the sigmoid):

$$L(M) = 2(1 + e^{M})^{-1}$$

### Task

Implement the logarithmic (`log_loss`) and sigmoid (`sigmoid_loss`) loss functions. The loss function needs to take the vector of *margins* (the value that defines how close the classified object is to the boundary between two classes, for more details see the next task) and return a pair of vectors – the vector of the loss function values and the vector of its derivatives. For example, if we use
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

We will get the visualization of the loss functions' work in the next task, after building the weight vector $\vec{w}$.