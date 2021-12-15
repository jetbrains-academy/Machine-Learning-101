<h2>Numerical assessment of algorithm quality</h2>

In the simplest case, a quality metric might be the proportion of the correctly classified objects.
$$Accuracy=\frac{P}{N}$$

where $P$ is the number of correctly classified objects, and 
$N$ is the testing sample size. 

This metric is simple and good, but it has a peculiarity: all objects are assigned the same weight, which might be incorrect
in the case of imbalanced distribution of classes in the training sample. In such a case, the classifier will have more information 
about certain classes, and within those classes it will make better decisions. As a result, even if the overall accuracy is 90%, 
with regard to some classes the algorithm will work poorly, with the accuracy lower than 50%.
Such problems may be avoided if we choose a different approach towards quality assessment.

<details>
<summary><b><a href="https://en.wikipedia.org/wiki/Precision_and_recall">Precision and Recall</a></b></summary>
<p>Precision and recall, which we have already encountered in previous lessons, are used in the assessment of most information retrieval algorithms.
They may be used both independently and as a basis for complex metrics, such as
F-score or R-Precision. Just to remind you: precision (within a class) is the proportion of objects really belonging to that class 
among all objects assigned to that class by the algorithm. Recall is the proportion of objects belonging to a class and identified by the classifier among
all objects of that class in the testing sample.</p>
</details>

<details>
<summary><b><a href="https://en.wikipedia.org/wiki/Confusion_matrix">Confusion Matrix</a></b></summary>
<p>In practice, precision and recall values are conveniently calculated with the help of the confusion matrix.
When the number of classes is relatively small (not more than 100-150), it allows visualizing
the results of the algorithm's work..</p>
<p>Confusion matrix is a matrix of order N х N, where N is the number of classes. Matrix columns represent the reality, and matrix rows
– the classifier's decisions. When an object from the testing sample is classified, the number at the intersection of the class row returned by the algorithm and the
class column the object really belongs to, increases. Consequently,
if the classifier identifies most objects correctly, the diagonal elements of the matrix will be salient.</p>
</details>

<details>
<summary><b><a href="https://en.wikipedia.org/wiki/F-score">F-score</a></b></summary>
<p>The higher the precision and recall are, the better. However, in practice, maximum precision and maximum recall cannot be achieved
simultaneously. That's why it would be good to have a metric which combines the algorithm's precision and recall. That's exactly what the
F-score is: it's a harmonic mean between precision and recall. It tends to zero if the precision or the recall
tend to zero.</p>
$$F = 2\frac{Precision * Recall}{Precision + Recall}$$
</details>

<h2>Task</h2>

In this task, we assess the classification quality by mere calculation of the proportion of correctly classified objects among all
objects in the sample.

In the `evaluate.py` file, implement the `accuracy` function, which passes the testing sample through the algorithm, compares 
the received class labels with the real ones, and returns the proportion of correctly classified objets.

To see the results of your code's work, you can add the following lines to the `main` block in `task.py` block and run it:

```python
print("Accuracy:")
print(accuracy(nn, X_test, y_test))
```
Variables required for the correct work of this code were introduced in previous steps; if you haven't worked with `task.py` yet, pay attention to them:
```python
X, y = read_data('iris.csv')
trainX, trainY, testX, testY = train_test_split(X, y, 0.7)
nn = NN(len(X[0]), 5, 1)
nn.train(trainX, trainY)
```
Try running the code in `task.py` several times to see how `accuracy` changes on each run.