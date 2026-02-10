Now, it's about time to evaluate the quality of our classifier, i.e., its *precision* and *recall*. Formally, these metrics are calculated in the following manner:
$$
\operatorname{Precision} = \frac{TP}{TP + FP} \\\\
\operatorname{Recall} = \frac{TP}{TP + FN}
$$

Here,
- $TP$ (true positives) is the number of elements the classifier **correctly** assigned to the class $c$;
- $FP$ (false positives) is the number of elements the classifier **incorrectly** assigned to the class $c$;
- $FN$ (false negatives) is the number of elements the classifier **incorrectly** did not assign to the class $c$.

**Precision** defines how reliable the classifier is. In our case, precision is the fraction of wines correctly assigned to class 1 among all wines assigned to class 1. The higher the precision of an algorithm, the fewer mistakes it will make in classifying elements.


**Recall** shows the number of class elements correctly assigned by the algorithm to that class. In our case, it is the fraction of the wines correctly assigned by the algorithm to class 1 among the actual number of wines in class 1. The higher the recall, the less frequently the algorithm misses elements that should be assigned to this or that class.

### Task
Implement the body of a function that calculates precision and recall for each class according to the predictions made by `knn`. The function may be found in the `metrics.py` file.
The function needs to return a list of triples `(class, precision, recall)`.
Thus, for each class, we will know the following: how precisely the algorithm assigns a wine to this class and with what probability it will miss such a wine.


The function may look as follows:

    # Let y_pred be the result of applying the k-nearest neighbors algorithm to the testing sample X_test.
    
    y_pred = knn(X_train, y_train, X_test, k, dist)

    def precision_recall(y_pred, y_test):
        class_precision_recall = []
        for c in range(n_classes):
            # … 
            
        return class_precision_recall

Don't forget to import the implemented function to `task.py` in order to use it in the main program and output the result.

<div class="hint">
You can calculate the value of <code>n_classes</code> using <code>y_test</code>:
<pre>
<code>
    n_classes = len(set(y_test))
</code>
</pre>
or
<pre>
<code>
    import numpy as np
    n_classes = len(np.unique(y_test))
</code>
</pre>
</div>

Here we use the <a href="https://numpy.org/doc/1.18/reference/generated/numpy.unique.html">numpy.unique</a> function, which returns all unique elements of an array. It helps us get the list of all classes represented in the sample.
