**Training a neural network** means finding the right combination of weights and biases for solving the required problem. 
The process of setting up a neural network involves successive implementation of the `forward` and `backward` steps for fine-tuning classifier weights.

<h3>Task</h3>
In the `network.py` file, implement the `train` method of the `NN` class. Besides data, it takes the `n_iter` parameter, which sets
the necessary number of iterations. The method should call two other (previously implemented) methods in the right order. It does not return anything.

<div class="hint">
On each iteration, generate predictions via <code>feedforward</code> and update the model's parameters using <code>backward</code> propagation.

```python
        for itr in range(n_iter):
            l2 = self.feedforward(X)
            self.backward(X, y, l2)
```
</div>

Augment the implementation by the `predict` method, which passes all objects from the `X` matrix through the trained neural network.

<div class="hint"> 
The <code>predict</code> method is a required part of the neural network's interface.
We will implement it here, even though it simply acts as a wrapper for the <code>feedforward</code> method. 

```python
return self.feedforward(X)
```
While this case is straightforward, other scenarios may require a more complex implementation.
</div>

Before you begin, delete the `pass` statement and uncomment all lines that are not task-related comments.

To see the results of your code in this step, add the following lines to the `main` block in `task.py`:

```python
nn.train(X_train, y_train)
print(f'w1 after training: \n{nn.w1} \nw2 after training:\n{nn.w2}')
```
This code will allow you to see the weigh changes after the training.

 

