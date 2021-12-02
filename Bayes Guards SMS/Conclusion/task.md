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
  
- [Bayesian inference in phylogeny](https://ru.wikipedia.org/wiki/%D0%91%D0%B0%D0%B9%D0%B5%D1%81%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9_%D0%BF%D0%BE%D0%B4%D1%85%D0%BE%D0%B4_%D0%B2_%D1%84%D0%B8%D0%BB%D0%BE%D0%B3%D0%B5%D0%BD%D0%B5%D1%82%D0%B8%D0%BA%D0%B5) allows getting most probable
  phylogenetic trees according to given original data – DNA or protein sequences in researched organisms and in the evolutionary
  model of nucleotide substitutions. The high speed of the algorithm and its
  possible integration with the [MCMC](https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%80%D0%BA%D0%BE%D0%B2%D1%81%D0%BA%D0%B0%D1%8F_%D1%86%D0%B5%D0%BF%D1%8C_%D0%9C%D0%BE%D0%BD%D1%82%D0%B5-%D0%9A%D0%B0%D1%80%D0%BB%D0%BE) methods
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
- There is a problem of "zero frequency"; however, it is solved with smoothing.
- This algorithm works with continuous characteristics worse than with categorical ones.
  Continuous characteristics imply normal distribution, which is a strong assumption.
  
When solving real-life problems that require the Bayes method, you don't really need to write all your code from scratch.
You can use, for example, the [sklearn.naive_bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) module of the scikit-learn library where the necessary algorithms are already implemented.
It's an open-source library, which provides a wide range of training algorithms.


> <i>This course is currently in the Alpha version. You can help us improve it by answering questions after each task in the following
> <a href="https://docs.google.com/forms/d/e/1FAIpQLSd3V5XUAMjCyU4uOuri9WKBEXpVsRfzCfMfVtnS8AzjqdXqFw/viewform?usp=sf_link">form</a>.
> Thanks! :) </i>
