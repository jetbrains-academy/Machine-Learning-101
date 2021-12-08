One of the problems with the naive approach is the following: if a word has not occurred in the training sample of the 
`Spam` class, its probability is:
$$P(word|Spam)=0$$

It's called the problem of zero occurrences. It leads to the impossibility of classifying a message
with this word, as it will have a zero probability in all classes. The problem may not be handled by means
of analyzing a large number of documents because it's impossible to make a training sample that will include 
all possible words, including synonyms, neologisms, typos, etc.

One of common solutions is [adaptive smoothing](https://en.wikipedia.org/wiki/Laplace_smoothing), or Laplace smoothing –
a technique that allows smoothing categorical data (i.e., data qualitatively characterizing the researched process or object
that cannot be represented quantitatively).
Given the observed ${\textstyle \textstyle {\mathbf {x} \ =\ \left\langle x_{1},\,x_{2},\,\ldots ,\,x_{d}\right\rangle }}$ 
from a multi-nominal distribution with ${\textstyle \textstyle {N}}$ 
tests, the “smoothed” data variant will provide the following assessment:


$$\hat{\theta_i}={\frac{x_i + \alpha}{N+\alpha d}}\qquad (i=1, \ldots , d),$$


where $α > 0$ is the smoothing parameter, and $α = 0$ corresponds to the absence of smoothing.

Essentially, if $α = 1$, we pretend that each word has occurred one time more often, i.e.,
we add 1 to each word's occurrence. Thus, the words that did not occur at the training stage of the model get a small
but not zero probability. To balance it, we increase
the number of possible words in the denominator so that the division result doesn't exceed 1:

$$P(w_i|c)={\frac{W_{ic}+1}{\sum_{i'\in|V|}{(W_{i'c}+1)}}}={\frac{W_{ic}+1}{|V|+\sum_{i'\in|V|}{W_{i'c}}}}$$

where: 

$P(w_i|c)$ is the probability of a word's occurrence in a class;

$W_{ic}$ the number of occurrences of the $i$-th word in the messages of the class $c$;

$V$ is the list of all unique words.


### Task
Update the implementation of the `fit` method so that it uses Laplace smoothing.
The `alpha` parameter is already initialized in the code.

<div class="hint">
You need to change the initial implementation of the <code>likelihood</code> attribute, as well as the calculation
of the denominator value. </div>

<div class="hint">
Initially, the <code>likelihood</code> array must be filled not with zeros but with $\alpha=1$. </div>

To see how your code works, you can run `task.py`.
In this task, you don't need to modify `task.py`. Mind the change of probability values compared to the previous 
step.
