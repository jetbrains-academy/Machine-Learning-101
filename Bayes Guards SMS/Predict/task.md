In our case, the class variable has only two possible values:
`Spam` or `Ham`. Of course, there are cases when the classification is multi-dimensional. That's why we need to
find the class variable with the maximum probability. Using
the below formula of a classification algorithm, we can get such a class according
to the available predictors.

$$y=\arg\max\limits_{y \in Y}  \prod  P(y) \times  P(x_j |y)$$

$y$ are `Spam` or `Ham` classes;

$x_j$ is the j-th word in a sentence.


In case the message is quite lengthy, we'll need to multiply a lot of very small numbers,
which may lead to a situation when the result of a floating point operation is so close to zero
(has such a precise absolute value) that it will not be representable as a normal floating-point number.
Such a situation is referred to as
[arithmetic underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow), and the standard method of avoiding it to
apply a logarithm to the expression under $argmax$. Thus, our algorithm formula
becomes as follows:

$$ \arg\max\limits_{y \in Y} [ \log(P(y)) + \sum\limits_{j=1}^{|V|} log(p(x_j |y))]$$

## Task
In the `smoothed_bayes.py` file, implement the `predict` method of the `SmoothedNaiveBayes` class, which 
takes an array of objects `X` and returns a list of corresponding class labels. Before doing it,
delete the `pass` operator.

<div class="hint">In order to uncomment the required lines, you can select the whole block with comments and press Ctrl + / 
(Windows, Linux) or ⌘ + / (MacOs). </div>

- First, you need to turn each message within the array into a vector of separate words with the
  help of the `split_by_words()` function.
- In each message, find a set of unique words and create a vector of zeros of
  the same size.

<div class="hint">
Create a zero-initialized array to store the index of each unique word.

```python
index_array = np.zeros(unique.shape, dtype=np.int64)
```
</div>

- For each unique word from the list, find a correspondence in the dictionary; if you find it,
  write its index to the vector created in the previous step, if not – write an index
  equal to the dictionary length. All such words have the same
  probability.

<div class="hint">

For each word, assign its dictionary index to `word_index`; if the word is not found, set `word_index` to `dict_size`.

```python
    word_index = self.dictionary[word] if word in self.dictionary else self.dict_size
```

**Why do this?**

Because `self.likelihood` stores word probabilities by index rather than the string itself.

</div>

- Calculate `log_likelihood` by applying the `np.log()` function to the slice of the `likelihood` array
  obtained with the help of `index_array`; thus, the array will contain the probabilities of only
  those words that occur in our sentence. 

<div class="hint">

`self.likelihood[c, w]` represents the probability of word `w` for class `c`.
Select the probabilities for the message words and calculate their logs:

```python
log_likelihood = np.log(self.likelihood[:, index_array])
```
</div>

- Compute the posterior score for each class.

<div class="hint" title="Posterior meaning">
The posterior score indicates the probability of each class given the message.
Compute it by summing the log-probabilities of the words (across <code>axis=1</code>) and adding the log prior.
</div>

<div class="hint" title="Posterior formula">

```python
posterior = np.log(self.classes_prior) + np.sum(log_likelihood, axis=1)
```
</div>

- Identify the class with the highest score for each message.

<div class="hint" title="Find the best class">
After finding the posterior probabilities for classes, you need to determine which 
one is the largest among them and choose a class corresponding to it from <code>unique_classes</code>. Here, 
you may use the <a href="https://numpy.org/doc/stable/reference/generated/numpy.argmax.html">numpy.argmax</a> function.
</div>

<div class="hint" title="Prediction formula">

```python
predicted = self.unique_classes[np.argmax(posterior)]
```
</div>

Then, implement the `score` method, which passes the test samples through the algorithm, compares the predicted
class labels with the true labels, and returns the proportion of correctly classified objects.

<div class="hint">
Use <code>predict</code> to generate labels, compare them with the true labels <code>y</code>, and return the fraction of matches:

<pre><code>return np.sum(self.predict(X) == y) / len(y)</code></pre>
</div>

To see the results of your code, you can add the following
lines to the `main` block in `task.py` and then run it:

```python
print(nb.predict(["This is not a spam"]))
print("Score:")
print(nb.score(X_test, y_test))
print(nb.score(X_train, y_train))
```
