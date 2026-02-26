import numpy as np
from PIL import Image, ImageDraw


def centroid_histogram(labels):
    # Here, we count the occurrences of each unique label in the
    # labels array to find out how many objects
    # belong to each cluster.
    unique, counts = #TODO
    return counts


# This function allows us to visualize the cluster distribution from
# K-Means as a bar chart.
def plot_colors(hist, centroids):
    # The bar is an array used to draw the rectangular background behind
    # the histogram. It is scaled to a width of 500 pixels.
    bar = np.zeros((50, 500, 3), dtype=np.uint8)
    # We create an Image object from the array.
    hist_plot = Image.fromarray(bar)
    # We create a drawing object for the hist_plot.
    draw = ImageDraw.ImageDraw(hist_plot)
    # The starting x-coordinate for the drawing
    start_x = 0
    # The total number of samples across all clusters
    sum_hist = np.sum(hist)
    for (percent, color) in zip(hist, centroids):
        # The ending x-coordinate is calculated
        # by adding the cluster's
        # share of the total points of a given color
        # (scaled to a maximum width of 500)
        # to the starting position.
        end_x = #TODO
        # Here, we draw the current rectangle with a height of 50, using the color
        # defined by the clusters' centroid.
        draw.rectangle(((int(start_x), 0), (int(end_x), 50)), tuple(color))
        # We set the starting position for the next bar to the end
        # of the current one.
        start_x = #TODO

    # We save the image as a .png.
    hist_plot.save("histogram.png")
