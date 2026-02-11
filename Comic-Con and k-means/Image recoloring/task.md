Now we only need to implement the function that will recolor the image with the required number of colors.

### Task

Implement the `recolor(image, n_colors)` function, which takes an image as a numpy-array and the number of colors as input. The function should return the array of an image whose colors were changed by the `k_means` method. You can find the function template in the `processing.py` file. The `process_image` function opens the image, calls `recolor`, and saves the file.

In order to save the image, we will first create a ```Pillow.Image``` object (we've already encountered it in the **Reading an image** task) with the help of the ```fromarray``` method, [which creates](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray) an image from an array. Then, we will save the image using the ```image.save``` method.


To see the results of your code, add the following lines in `task.py` and run it:
1. Necessary import:
 ```python
from processing import process_image
```
2. Add the line for getting the result in the `main` block **instead** of those added in the previous step:
```python
process_image(image)
```
The image will show up in the list of task files on the left.


<div class="column" style="float: left;width: 45%;padding: 5px;">
    <img src="superman-batman.png" alt="Original image" style="width:100%">
    <p style="text-align:center;">Original image</p>
</div>
<div class="column" style="float: left;width: 45%;padding: 5px;">
    <img src="superman-batman-after.png" alt="16-color image" style="width:100%">
    <p style="text-align:center;">8-color image</p>
</div>
