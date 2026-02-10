Now we need to modify the probability formula so that we could use it with word occurrence.
To do that, we can utilize certain basic probability features. Let's remember the formula
of calculating the probability of event $A$ occurring in case event $B$ has already occurred:


$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

In our case, we need to calculate the probability of a message being spam in case of the occurrence of a certain set of words in it, i.e.,
$P(Spam|sentence)$. Thus, with Bayes' theorem, we get the following:

$$P(Spam|sentence) = \frac{P(sentence|Spam) \times P(Spam)}{P(sentence)}$$

In our classifier, we are only trying to find the most probable class, so we can ignore
the denominator, which will be the same for both classes, and compare only the numerators:


$$P(sentence|spam) \times P(spam)$$

and

$$P(sentence|ham) \times P(ham)$$

The **naive Bayes classifier** is an algorithm for solving tasks of two-class or multi-class classification.
It is called "naive" because probability calculations for each class are simplified 
for convenience. It is assumed that the presence of a certain characteristic is not connected with the
presence of others. It is a very strong assumption, which is hardly applicable to
real data: it is unlikely that the characteristics are not interconnected. Still, this approach
works surprisingly well even with the data that do not meet this
condition.

In our case, the above assumption means that the probability of some word occurring in a message
does not depend on the occurrence of other words in that message.

$$P(\text{Who let the dogs out}) = P(\text{Who}) \times P(\text{let}) \times P(\text{the}) \times P(\text{dogs}) \times P(\text{out})$$


### Task

In the `bayes.py` file, implement the method `fit` of the `NaiveBayes` class,
which calculates and saves as class attributes the parameters of the sample we will need
at the classification stage:
- `classes_prior`&nbsp;— it is the assessment of the prior probability of classes presented as a NumPy vector of length 2 
(the number of classes). It is calculated as the proportion of each class in the whole sample:
  $$P (\text{spam}) = \frac{N_{spam}}{N_{documents}}$$

- `classes_words_count`&nbsp;— it is the total number of words for messages of each class
  presented as a vector of length 2. To calculate it, use the matrix `X` (received through `vectorize(X)`) 
  and the mask `y_i_mask`. You will need the [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) function.
- `likelihood`&nbsp;— these are relative occurrences of words for each class presented as a NumPy array
  of the shape `(2, M)`, where `M` is the size of the dictionary. To calculate the occurrence of each
  word in each class, we first need to calculate the occurrence of each word in all messages of the class
  with the `X` matrix (received through `vectorize(X)`) and the 
  `y_i_mask` mask, and then divide each element by the total number of words in the messages of the class 
  (`classes_words_count`).
  
<div class="hint">

The `numpy.sum` function allows summarizing both all the array elements and
the elements along the chosen axis. By default, `axis=None`, that is all the elements of an array are summarized.</div>

To see how your code works, you can run `task.py`.
In this task, add the following lines to the `main` block before running
your code:
```python
nb = NaiveBayes()
nb.fit(X_train, y_train)
print('Total number of words in each class: ', nb.classes_words_count)
print('Class prior probabilities: ', nb.classes_prior)
print('Relative word frequencies for each class: ', nb.likelihood)
```
Besides, you need to import the module with the required class to `task.py`:
```python
from bayes import NaiveBayes
```
