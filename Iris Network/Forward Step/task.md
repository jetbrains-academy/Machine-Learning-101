<h2>Training a neural network</h2>

<p>Let's remember the output data $\hat{y}$ formula for a simple 2-layer neural network:</p>

$$\hat{y} = \sigma(W_2 \sigma(W_1x + b_1) + b_2$$

<p>Weights $W$ and biases $b$ are the only parameters affecting the output data $\hat{y}$. Properly selected values
of weights and biases determine the correctness of the result. The process of adjusting weights and biases is, in fact, the process of training a neural network</p>

<p>Each iteration of the training process includes the following steps:</p>

<ul>
<li>Calculating the predicted output data – feedforward</li>
<li>Updating weights and biases – backpropagation</li>
</ul>

<h3>Feedforward</h3>

Feedforward is a regular calculation from the input layer to the output layer; for a simple 2-layer neural network, the output data will be as follows:
$$\hat{y} = \sigma(W_2 \sigma(W_1x + b_1) + b_2$$

<h2>Data</h2>
Fisher's irises (Anderson's irises, or the iris data) is the most common dataset used for testing machine learning algorithms. The data contain 4 characteristics
for various types of irises: <i>setosa</i>, <i>versicolor</i>, and <i>virginica</i>. Each type is represented by 50 flowers.

<ul>
<li>sepal length</li>
<li>sepal width</li>
<li>petal length</li>
<li>petal width</li>
</ul>


<p>We will build a neural network-based classification model relying on these data. For convenience, we will use only the petal length and width
of <i>versicolor</i> and <i>virginica</i> irises.</p>

<h2>Task</h2>

<p>In the <code>network.py</code> file, class <code>NN</code> is implemented – a neural network with <code>input_size</code> of input neurons, <code>hidden_size</code> of hidden neurons, and 
<code>output_size</code> of output neurons. Attributes <code>w1</code> and <code>w2</code> are the connection weights between the input and hidden neurons and between the hidden and output neurons
respectively. <code>input_size</code> will depend on the input data.</p>

<p>Implement the <code>feedforward</code> method. It has to <a href="https://en.wikipedia.org/wiki/Matrix_multiplication">multiply</a> 
the weights matrix <code>w1</code> by the matrix of the input data and then apply the activation function to the matrix product. 
Then, the method has to multiply the data matrix received in the previous step by the weights matrix <code>w2</code>,
apply the activation function to the product, and return the result. Other methods of the class will be implemented in further steps.</p>

<p>For more simplicity, we assume the biases to be equal to 0.</p>

<div class="hint">
To multiply the matrices, you can use the <a href="https://numpy.org/doc/stable/reference/generated/numpy.dot.html">numpy.dot</a> function.</div>

You can run `task.py` in the tasks to check how your code works. In this task, you can also get a graph
illustrating the distribution of the chosen characteristics in the data. You don't need to modify `task.py` in this task.

