The naive Bayes algorithm is one of the most common algorithms in natural language processing: in
these tasks, it is superior to many others. Due to that, naive Bayes has a wide range of applications in
the field of spam filtering (detecting spam in emails) and text sentiment analysis (social media analysis, identification of
clients' positive and negative opinions).
However, it also has other fields of application.

- The naive Bayes algorithm is quickly trained, so it can be used to **classify data in real time**.
  
- The naive Bayes classifier accompanied by 
  collaborative filtering allows realising a **recommendation system**. 
  It filters the information new to the user on the basis of the user's predicted
  opinion on it, which is calculated with machine learning methods and intellectual data analysis.
  Such systems are widely used by services like
  Netflix, Amazon, Facebook, and Google.
  
- Naive Bayes allows predicting probabilities for multiple values of the target variable,
  which makes possible **multi-class classifications**.
  
- [Bayesian inference in phylogeny](https://en.wikipedia.org/wiki/Bayesian_inference_in_phylogeny) allows getting most probable
  phylogenetic trees according to given original data – DNA or protein sequences in researched organisms and in the evolutionary
  model of nucleotide substitutions. The high speed of the algorithm and its
  possible integration with the [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) methods
  (Markov chain Monte Carlo) make naive Bayes one of the most popular methods of
  statistical inference. The development of the MCMC methods allows calculating
  large hierarchical models that require integrating hundreds or even thousands of
  unknown parameters.

### Naive Bayes: pros and cons

**Pros**:
- Classification, including multi-class classification, is easy and fast.
- It works well with categorical characteristics.
- When the independence assumption is met, the naive Bayes classifier works
  better than other algorithms, such as, for example, logistic regression,
  while requiring a smaller volume of data for training.

**Cons**:
- The assumption of independent characteristics is a limitation. In reality,
  sets of totally independent characteristics are extremely rare.
- There is a problem of "zero frequency" although it is solved with smoothing.
- This algorithm works with continuous characteristics worse than with categorical ones.
  Continuous characteristics imply normal distribution, which is a strong assumption.
  
When solving real-life problems that require the Bayes method, you don't really need to write all your code from scratch.
You can use, for example, the [sklearn.naive_bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) module of the scikit-learn library where the necessary algorithms are already implemented.
It's an open-source library, which provides a wide range of training algorithms.

