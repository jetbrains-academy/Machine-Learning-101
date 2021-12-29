**Training a neural network** means finding the right combination of weights and biases for solving the required problem. 
The process of setting up a neural network involves successive implementation of the `forward` and `backward` steps for fine-tuning classifier weights.

<h2>Task</h2>
In the `network.py` file, implement the `train` method of the `NN` class. Besides data, it takes the `n_iter` parameter, which sets
the necessary number of iterations. The method should call two other (previously implemented) methods in the right order. It does not return anything.

Augment the implementation by the `predict` method, which passes all objects from the `X` matrix through the trained neural network.

Before you start, delete the `pass` operator and uncomment all lines that are not task commentaries.

<div class="hint"> The <code>predict</code> method is a part of the interface of a program the neural network is expected to include, so we will implement it 
despite the fact that it just calls the <code>feedforward</code> method. It's a lucky coincidence – in other cases, there might be
something else.</div>

To see the results of your code in this step, add the following lines to the `main` block in `task.py`:

```python
nn.train(X_train, y_train)
print(f'w1 after training: \n{nn.w1} \nw2 after training:\n{nn.w2}')
```
This code will allow you to see the weigh changes after the training.

 

