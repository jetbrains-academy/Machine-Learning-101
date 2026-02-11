import numpy as np
from PIL import Image, ImageDraw


def centroid_histogram(labels):
    # Here we retrieve the counts of unique entries in the
    # labels array (those are the clusters, and we need to know how many
    # objects belong to each of them).
    unique, counts = #TODO
    return counts


# This function will allow us to draw the clusters received from the
# k-means as bars in a histogram.
def plot_colors(hist, centroids):
    # The bar is an array representing the underlying rectangle behind
    # the histogram. It is scaled to have a width of 500.
    bar = np.zeros((50, 500, 3), dtype=np.uint8)
    # We create an Image object from the array.
    hist_plot = Image.fromarray(bar)
    # We create a drawing instance of the hist_plot.
    draw = ImageDraw.ImageDraw(hist_plot)
    # The x-axis position where we start to draw
    start_x = 0
    # The overall sum of the samples belonging to all of the clusters
    sum_hist = np.sum(hist)
    for (percent, color) in zip(hist, centroids):
        # The ending x-axis position is obtained
        # by adding the percentage of all cluster points
        # of a given color in the overall count
        # (scaled to have a width 500)
        # to the starting position.
        end_x = #TODO
        # Here we draw the current rectangle with a height of 50 and color
        # defined by the clusters' centroid.
        draw.rectangle(((int(start_x), 0), (int(end_x), 50)), tuple(color))
        # We set the starting point for the next bar to the ending point
        # of the current one
        start_x = #TODO

    # ...and we save the image as a .png.
    hist_plot.save("histogram.png")
