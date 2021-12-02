In this lesson, we will develop a naive [Bayes classifier](http://www.machinelearning.ru/wiki/index.php?title=%D0%9D%D0%B0%D0%B8%D0%B2%D0%BD%D1%8B%D0%B9_%D0%B1%D0%B0%D0%B9%D0%B5%D1%81%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9_%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%82%D0%BE%D1%80) to detect
spam in SMS messages. Despite the huge success of machine learning over the recent years, the naive Bayes algorithm
remains not only one of the simplest but also one of the fastest, most precise and reliable algorithms.
It is successfully used for numerous purposes (recommendation systems, real-time data classification, etc.),
but it is exceptionally good in tasks involving natural language processing.

<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
</style>
![bayes](Thomas_Bayes.png)

The naive Bayes classifier is based on [Bayes' theorem](https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D0%BE%D1%80%D0%B5%D0%BC%D0%B0_%D0%91%D0%B0%D0%B9%D0%B5%D1%81%D0%B0). Bayes' theorem
is a mathematical formula used for conditional probability calculation.

**Conditional (posterior) probability** is the probability of an event provided that
another event has already occurred (by conjecture, assumption, or an assertion – 
whether confirmed or unconfirmed).

Here is the formula for calculating conditional probability:

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

It shows how often an event $A$ occurs in case of an event $B$. We need to know the following:
- How often the event $B$ occurs in case of the occurrence of the event $A$, which the formula indicates as $P(B|A)$.
- The unconditional (prior) probability of the event $A$, which the formula indicates as $P(A)$.
- The unconditional probability of the event $B$, which the formula indicates as $P(B)$.

One might say that Bayes' theorem is a way to calculate probability on the basis of our knowledge of other probabilities.

<div class="hint"><b>The paradox of Bayes' theorem</b>. Let's assume there is a disease with
the occurrence among the population equalling 0,001 and a diagnostic method (test) that reveals a sick person with the probability of 0,9
but also has the 0,01 probability of a false positive result – an erroneous diagnosis of the disease in a healthy person.
We need to calculate the probability of a person being healthy when
the test identified them as sick. According to Bayes' theorem, this probability is 91.7%, that is 
the majority of people diagnosed as "sick" are in fact healthy. The reason is that, according to the task condition, 
the probability of a false positive result, even if small, is by an order of magnitude larger than the proportion of
sick people in the sample group. You can find the calculations and a detailed explanation <a href="https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D0%BE%D1%80%D0%B5%D0%BC%D0%B0_%D0%91%D0%B0%D0%B9%D0%B5%D1%81%D0%B0#%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_4_%E2%80%94_%D0%BF%D0%B0%D1%80%D0%B0%D0%B4%D0%BE%D0%BA%D1%81_%D1%82%D0%B5%D0%BE%D1%80%D0%B5%D0%BC%D1%8B_%D0%91%D0%B0%D0%B9%D0%B5%D1%81%D0%B0">here</a>. </div>

The **naive classifier** is a probability model. It calculates the probability for each text class
and returns the more probable one.

Our task will be to calculate the probability of a given sentence being "Spam" or "Ham" and then 
choose the more probable variant.

$P(Spam|sentence)$ is the probability that a sentence is “Spam”, taking into account
the set of words in a specific sentence.


### Task

In the "Spam.txt" file, you will find a dataset containing marked sentences. The first word in each line 
is an identifier of the “Spam” or “Ham” class; then, after a tabulation, follows the body of the message.

Before building a classifier, we need to present the data in a format convenient for classifying. To do that,
we will use a standard text-processing model named [Bag of words](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%88%D0%BE%D0%BA_%D1%81%D0%BB%D0%BE%D0%B2). It allows calculating 
the number of occurrences for each word in a document, regardless of word order and syntactic structures.

In the `vectorize.py` file, we have realized a function `split_by_words()`, which takes a vector of text lines and returns a vector of word lists for each line.
It utilizes the following functions:

- [numpy.char.lower](https://numpy.org/doc/stable/reference/generated/numpy.char.lower.html) – it returns a list of elements transformed into lowercase.
- [numpy.char.translate](https://numpy.org/doc/stable/reference/generated/numpy.char.translate.html#numpy-char-translate) — it allows transforming a text line by applying a certain transformation to each character.
- [numpy.char.split](https://numpy.org/devdocs/reference/generated/numpy.char.split.html#numpy-char-split) — it returns a list of words for each array element (lines).
- [str.maketrans](https://docs.python.org/3/library/stdtypes.html#str.maketrans) — returns a character translation table. It uses three
  arguments: x, y, and z, where ‘x’ and ‘y’ are text lines of the same length and the characters in ‘x’ are
  substituted by those from ‘y’. The ‘z’ argument is a line (in our case, string.punctuation), all characters of which
  are substituted by `None`. It helps to get rid of punctuation.

In the same file, realize the`vectorize` function. It should do the following:
1) Find the number of messages in the input.
2) Get a vector of separate word lists from them using the `split_by_words()` function.
3) Get a one-dimensional array of unique words from it.
4) Build a dictionary, where each unique word will be assigned an index.
5) Create a matrix of the order (N, M), where M is the dictionary size. The j-th element of each matrix line
   is a number x, which indicates that the j-th word occurred x times in the i-th message.
6) Return the dictionary and the matrix.

<div class="hint">
In this task, you might want to use the following functions:
<a href="https://numpy.org/doc/stable/reference/generated/numpy.unique.html?highlight=unique#numpy.unique">numpy.unique</a>&nbsp;— we've come across this function on several occasions in the previous lessons.
<a href="https://numpy.org/doc/stable/reference/generated/numpy.hstack.html">numpy.hstack</a>&nbsp;— it concatenates arrays along the second axis, except for one-dimensional arrays, which
are concatenated along the first axis.
</div>

To see how your code works, you can launch `task.py`. You don't need to modify this file in the current task.

> <i>This course is currently in the Alpha version. You can help us improve it by answering questions after each task in the following
> <a href="https://docs.google.com/forms/d/e/1FAIpQLSd3V5XUAMjCyU4uOuri9WKBEXpVsRfzCfMfVtnS8AzjqdXqFw/viewform?usp=sf_link">form</a>.
> Thanks :) </i>