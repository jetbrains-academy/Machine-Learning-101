import numpy as np
from PIL import Image, ImageDraw


def centroid_histogram(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return counts


def plot_colors(hist, centroids):
    bar = np.zeros((50, 500, 3), dtype=np.uint8)
    hist_plot = Image.fromarray(bar)
    draw = ImageDraw.ImageDraw(hist_plot)
    start_x = 0
    sum_hist = np.sum(hist)
    for (percent, color) in zip(hist, centroids):
        end_x = start_x + percent * 500 / sum_hist
        draw.rectangle(((int(start_x), 0), (int(end_x), 50)), tuple(color))
        start_x = end_x

    hist_plot.save("histogram.png")