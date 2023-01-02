An **error** is a value reflecting the discrepancy between the expected and received answers; it has to decrease with each iteration.
If it is not happening, the algorithm works improperly. We can calculate the error in several ways, for example, using such methods as
[Sum of Squared Errors](https://en.wikipedia.org/wiki/Residual_sum_of_squares) 
(SSE), [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE), or [Root MSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation). 

As we use a random set of weights and biases at initialization and the result of the network's work will be all but random, we need to change the weights so that the network outputs the correct result (i.e.,
the output data of the classification should match the class labels known to us).
In our case, the correct result will mean the prediction of the iris type according to petal length and width that will match
the value indicated in the column "species" for that object. The necessary weight change is achieved by backward propagation.

<a href="https://en.wikipedia.org/wiki/Backpropagation">The backpropagation algorithm</a> is 
a popular feedforward neural network training algorithm. It belongs to the supervised learning type, so we need to indicate target values in the training sample
(thus, we know the characteristics of the types of irises).

Backpropagation uses the loss function, which shows how far the network is from the correct answer.

<h2>Losses</h2>

To calculate the loss function, we will use the sum of squared estimate of errors (SSE):

$$Loss(y, \hat{y}) = \sum\limits_{i=1}^{n} (y_i - \hat{y}_i) ^ 2$$

where $\hat{y}$ is the predicted output data;

$y$ is the real output data.

The Sum of Squared Errors (SSE) is the sum of differences between each predicted value and the real value.
The difference is squared so that we can use its absolute value. This value is the measure of our neural network's inaccuracy, and it should be minimized.

The task of training is to find such a combination of weights and biases that will best minimize the loss function. To find out how much and in what direction
weights and biases should be changed, we need to know the dependence of the derivative of the loss function on them.

<details>
It's necessary to note that the task of finding the global minimum of this function is usually very complex and
most probably, we will get one of the local minimums, which may be better or worse; however, which minimum exactly we will find
depends on the initial random combination of weights and biases.
</details>

In the <a href="https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20first,the%20direction%20of%20steepest%20descent.">Gradient Descent</a> 
chapter of the previous lesson, we said that the derivative (or <a href="https://en.wikipedia.org/wiki/Gradient">gradient</a>) shows the slope 
of the function graph. If we know the derivative of the loss function, we know the direction in which its value declines and thus, can update the weights
and biases increasing or decreasing them according to this value.

Yet, it's impossible to just calculate the dependence of the derivative of the loss function on weights and biases because the function equation contains neither weights nor biases.
Such calculations would require defining some connecting rule.
$$\frac {\partial Loss(y, \hat{y})}{\partial W} = \frac { \partial Loss(y, \hat{y} ) } {\partial \hat{y}} \frac { \partial \hat{y} } {\partial z} \frac { \partial z } {\partial W} $$
$$= 2 (y - \hat{y} ) * z (1- z) * x$$
where $z = Wx + b$

Here is the calculation of weights gain, i.e., the values that will be added to the weights of neurons of the output ($\delta_{o}$) and hidden layers ($\delta_{h}$):


$$\delta_o=(OUT_{real} - OUT_{actual}) * f_a'(OUT_{real})$$   
$$\delta_h=f_a'(OUT_h) * (w_i * \delta_i)$$

Weight update occurs according to the following formula:

$$weight = weight + learning\\_rate * error * input$$

where $weight$ is the weight; $learning\\_rate$ is the learning rate, that is a network settings <a href="https://en.wikipedia.org/wiki/Learning_rate">parameter</a> we need to indicate; $error$ is the error calculated for the neuron
in the previous step; and $input$ is the value of the input data which produced the error.

<details>
Learning rate in machine learning and in statistics is a settings parameter of the optimization algorithm 
that determines the step size at each iteration while approaching the minimum of the loss function.
</details>

<h2>Task</h2>
In the `network.py` file, implement the method `backward` of the `NN` class, which performs the following operations:

<ul>
<li>Calculate the error for the output layer (<code>delta_l2</code>) as the difference between the network results (<code>output</code>) and the real class labels (<code>y</code>) multiplied elementwise by the derivative of the activation function for output ($\delta_{o}$ formula).</li>
<li>Calculate the error for the hidden layer (<code>delta_l1</code>) as the product of input layer error matrices and the weights <code>w2</code> multiplied elementwise by the derivative of the activation function WRT the output data of the hidden layer (<code>layer1</code>) ($\delta_{h}$ formula).</li>
<li>Adjust the weight coefficients of the output layer (<code>w2</code>) by calculating the vector product of the hidden layer (<code>layer1</code>) and the output layer error (<code>delta_l2</code>) multiplied elementwise by the learning rate (formula 3).</li>
<li>Adjust the weight coefficients of the hidden layer (<code>w1</code>) by calculating the vector product of the input layer (<code>X</code>) and the hidden layer error (<code>delta_l1</code>), multiplied elementwise by the learning rate (formula 3).</li>
</ul>

Before you start, delete the `pass` operator and uncomment all lines that are not task commentaries.
The derivative of the activation function is implemented in the `derivative.py` module.

<div class="hint">When multiplying matrices, you will need to transpose some of them!</div>


To see the results of your code's work, you can add the following lines to the `main` block in `task.py` and run it:

```python
print(f'w1 before backward propagation: \n{nn.w1} \nw2 before backward propagation:\n{nn.w2}')
nn.backward(X_train, y_train, output)
print(f'w1 after backward propagation: \n{nn.w1} \nw2 after backward propagation:\n{nn.w2}')
```
This code will allow you to see the weigh changes after backpropagation.