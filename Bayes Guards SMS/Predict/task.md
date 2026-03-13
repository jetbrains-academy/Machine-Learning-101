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
- For each unique word from the list, find a correspondence in the dictionary; if you find it,
  write its index to the vector created in the previous step, if not – write an index
  equal to the dictionary length. All such words have the same
  probability.
- Calculate `log_likelihood` by applying the `np.log()` function to the slice of the `likelihood` array
  obtained with the help of `index_array`; thus, the array will contain the probabilities of only
  those words that occur in our sentence. 
- Use the above formula to calculate the most probable class for this message.
- Return the list of most probable classes of all messages in the input array.


Then, implement the `score` method, which passes the testing sample through the algorithm, compares the received
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

<div class="hint">
For every message, you need to convert its words into dictionary indices.
Create an array that will store the index of each unique word.

```python
index_array = np.zeros(unique.shape, dtype=np.int64)
```

</div>

<div class="hint">

Each word must be mapped to its index in `self.dictionary.`

If the word exists in the dictionary, use its stored index.
If it does not exist, use the special index for unknown words: `self.dict_size` (length of the dictionary).

```python
for i, word in enumerate(unique):
    word_index = self.dictionary[word] if word in self.dictionary else self.dict_size
```
**Why do this?**

Because `self.likelihood` stores word probabilities by index, not by the word itself.

</div>

<div class="hint">

`self.likelihood` stores the probabilities of words for each class.

Its shape is:
`number_of_classes * (dictionary_size + 1)`


- each **row** corresponds to a class  
- each **column** corresponds to a word index in the dictionary

For example, `self.likelihood[c, w]` represents the probability of word `w`
appearing in class `c`.

To compute the likelihood of the current message, select only the columns
that correspond to the words in the message and apply the logarithm:

```python
log_likelihood = np.log(self.likelihood[:, index_array])
```
</div>

<div class="hint">

At this point, `log_likelihood` already contains the log-probabilities of the words from the current message for **each class**.

Now you need to get one final score for every class.

To do this you need to combine two things:
<ul>
<li>how typical the words of the message are for that class</li>
<li>how common that class is in the training data</li>
</ul>

The first part is obtained from <code>log_likelihood</code>.  
The second part is the logarithm of the prior probability of the class stored in <code>classes_prior</code>.

Therefore, add the log prior probability to the summed log-likelihood values:

```python
posterior = np.log(self.classes_prior) + np.sum(log_likelihood, axis=1)
```
</div>

<div class="hint">
At this point, <code>posterior</code> contains one score for each class.  
These scores represent how likely the message belongs to each class.

Your task is to select the class with the highest score.

First, find the index of the largest value in the <code>posterior</code> array.  
You can use <code>numpy.argmax</code> for this — it returns the position of the maximum element.

Then use this index to retrieve the corresponding class label from <code>self.unique_classes</code>.

<pre><code>predicted = self.unique_classes[np.argmax(posterior)]</code></pre>

This selects the class with the highest posterior score.
</div>

<div class="hint">
The classifier already knows how to predict class labels using the <code>predict</code> method.  
However, we also need a simple way to check how well the model performs.

The <code>score</code> method does this by comparing the predicted labels with the true labels and computing the fraction of matches.

You can obtain the predicted labels by calling <code>predict</code>.

When two arrays are compared with <code>==</code>, NumPy produces a boolean array where each position shows whether the prediction is correct.
Summing this array gives the number of correct predictions.  
To obtain the required value, divide this number by the total number of objects.

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
