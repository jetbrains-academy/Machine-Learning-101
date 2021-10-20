import numpy as np
from PIL import Image
from clustering import k_means
from distances import euclidean_distance


IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024


def recolor(image, n_colors):
    (labels, centroids) = k_means(image.astype(np.int64), n_colors, euclidean_distance)
    return centroids[labels]


def process_image(image):
    recolored_image = recolor(image, 8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).astype('uint8')
    image = Image.fromarray(recolored_image)
    image.save("recolored-superman-batman.png")
