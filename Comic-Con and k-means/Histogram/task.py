from PIL import Image
import numpy as np
from plotting import plot_colors, centroid_histogram
from clustering import k_means
from distances import euclidean_distance

def read_image(path='superman-batman.png'):
    # Here we need to read the image using the PIL function open.
    image = Image.open(path)
    # We reshape the image array into one with the (M x N, 3)
    # shape.
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    (pixel_labels, centroids) = k_means(image, 4, euclidean_distance)
    print(pixel_labels)
    hist = centroid_histogram(pixel_labels)
    plot_colors(hist, centroids)
