Now, we've decided to visit a Comic-Con. To get ready, we need to Batman and Superman t-shirts. What we've got is just one picture with both characters,
`superman-batman.png`, and an 8-color textile printer.
With the help of the k-means clustering method, we need to make a new picture of our characters in just 8 colors.

The color information of each pixel is encoded according to the [RGB](https://ru.wikipedia.org/wiki/RGB) model
— a color model consisting of three components: R — red, G — green, and B — blue.
Each RGB component may take values `0` $\dots $ `255`. This allows us to decode`256*256*256` colors.
 
To change the color scheme, we will use the [k-means](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_k-%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B8%D1%85) algorithm — a clustering method that rearranges clusters (groups of close values) and recalculates their centers in each iteration.
 

### Task

First, we need to realize a function that reads a file and transforms it into a two-dimensional matrix ($M \times N	$, 3). Here, $M$ и $N$ represent the image height and width in pixels, and 3 is the number of R, G and B components for each pixel.

The function signature is the following:

    def read_image(path):
        # ...
        return image

Here we might use the [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) function, which changes the array shape. Initially, the array we get from the picture is three-dimensional, with the order of (M, N, 3).
<br/>

<div class="hint">
When you work with images, the
<a href="https://pillow.readthedocs.io/en/stable/">Pillow</a> library may come in handy. For example, its <code>open</code> function reads an image from a given path and returns an <a href="https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=open#PIL.Image.Image">Image object</a>, which may be represented as an array with the help of the <code>NumPy.array()</code> function. 
</div>


> <i>This course is currently in the Alpha version. You can help us improve it by answering questions after each task in the following
> <a href="https://docs.google.com/forms/d/e/1FAIpQLSfQxVgSmjAXNyxqoF5O4XxXSYWvHv2UcLsqfyt0MtE6u9820A/viewform?usp=sf_link">form</a>.
> Thanks! :) </i>