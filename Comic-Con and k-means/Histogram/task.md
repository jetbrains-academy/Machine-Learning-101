To evaluate the work of the algorithm, we will now use a histogram.

A histogram is a graphical representation of table data, which allows displaying quantitative characteristics of a certain feature as bars with proportional areas.

For clarity, we will build a histogram with bars of equal height, the area of each bar depending on the number of pixels assigned to a respective cluster. Thus, we will see the prevailing colors in the image recolored by the algorithm.


### Task
Implement the function `centroid_histogram(labels)` to build a histogram displaying the number of pixels assigned to each cluster. The function should return a vector whose $i$-th place indicates the number of pixels assigned to a cluster with the index `i`. You can find the function template in the `plotting.py` file.

You also need to complete the `plot_colors(hist, centroids)` function in the same file. It takes the number of points assigned to clusters as well as their centroids and displays a histogram with bars proportional to the number of points of each color in the image. 

A histogram example:
![Histogram](barchart.png)

<div class="hint">
To complete the task, you will need the <a href="https://numpy.org/doc/stable/reference/generated/numpy.unique.html#numpy.unique">np.unique</a> function, which takes an array as input and returns only its unique elements (as an array). If you pass the <code>return_counts=true</code> flag to the function, it will also calculate the number of occurrences for each unique element.
</div>

To see the results of your code, add the following lines in `task.py` and run it:
1. Necessary import:
 ```python
from plotting import plot_colors, centroid_histogram
```
2. Add the lines for getting the result in the `main` block **instead** of those added in the previous step:
```python
(pixel_labels, centroids) = k_means(image, 4, euclidean_distance)
print(pixel_labels)
hist = centroid_histogram(pixel_labels)
plot_colors(hist, centroids)
```
The histogram will show up in the list of task files on the left.