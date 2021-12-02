In our case, the class variable has only two possible values:
Spam or Ham. Of course, there are cases when the classification is multi-dimensional. That's why we need to
find the class variable with maximum precision. Using
the below formula of a classification algorithm, we can get such a class according
to the available predictors.

$$y=\arg\max\limits_{y \in Y}  \prod  P(y) \times  P(x_j |y)$$

$y$ are Spam or Ham classes;

$x_j$ is the j-th word in a sentence.


In case the message is quite lengthy, we'll need to multiply a lot of very small numbers,
which lead to a situation when the result of a floating point operation is so close to zero
that the order of the number goes beyond the word length and the computer won't be able to
present it in memory. Such a situation is referred to as
[arithmetic underflow](https://ru.wikipedia.org/wiki/%D0%98%D1%81%D1%87%D0%B5%D0%B7%D0%BD%D0%BE%D0%B2%D0%B5%D0%BD%D0%B8%D0%B5_%D0%BF%D0%BE%D1%80%D1%8F%D0%B4%D0%BA%D0%B0), and the standard method of avoiding it to
apply a logarithm to the expression under $argmax$. Thus, our algorithm formula
becomes as follows:

$$ \arg\max\limits_{y \in Y} [ \log(P(y)) + \sum\limits_{j=1}^{|V|} log(p(x_j |y))]$$

## Task
In the `bayes.py` file, realize the `predict` method of the `NaiveBayes` class, which 
takes an array of objects `X` and returns a list of corresponding class labels. Before doing it,
delete the `pass`operator and uncomment all lines except comments.

<div class="hint">In order to uncomment the required lines, you can select the whole block with comments and press Ctrl + / 
(Windows, Linux) or ⌘ + / (MacOs). </div>

- First, you need to turn each message within the array into a vector of separate words with the
  help of the `split_by_words()` function.
- In each message, find a set of unique words and create a vector with zeros of
  the same size.
- For each unique word from the list, find a correspondence in the dictionary; if you find it,
  write its index to the vector created in the previous step, if not – write an index
  equal to the dictionary length. All such words have the same
  probability.
- Calculate `log_likelihood` by applying the `np.log()` function to the array slice `likelihood`
  received with the help of `index_array`; thus, the array will contain the probabilities of only
  those words that occur in our sentence. 
- Use the above formula to calculate the most probable class for this message.
- Return the list of most probable classes of all messages in the input array.


Then, realize the`score` method, which passes the testing sample through the algorithm, compares the received
class labels with the real ons and returns the proportion of correctly classified objects.

<div class="hint">
Posterior probabilities for each class are calculated as the sum of the prior probability logarithm and the summarized logarithms of probabilities for the words from 
<code>log_likelihood</code> 
positioned along one axis, i.e., separately for each class.
</div>

<div class="hint">
After finding the posterior probabilities for classes, you need to determine which 
one is the largest among them and choose a class corresponding to it from <code>unique_classes</code>. Here, 
you may use the <a href="https://numpy.org/doc/stable/reference/generated/numpy.argmax.html">numpy.argmax</a> function.
</div>

To see the results of your code, you can add the following
lines to the `if __name__ == '__main__':` block in `task.py` and then launch it:

```python
print(nb.predict(["This is not a spam"]))
print("Score:")
print(nb.score(X_test, y_test))
print(nb.score(X_train, y_train))
```
