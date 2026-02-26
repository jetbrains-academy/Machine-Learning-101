import numpy as np
from PIL import Image
from clustering import k_means
from distances import euclidean_distance


IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024


def recolor(image, n_colors):
    # Here, we bring everything together by applying the k_means function to the image
    # to reduce its palette to the specified number of colors.
    # Note that it is better to cast the image array to a consistent type by calling .astype(np.int64).
    (labels, centroids) = k_means(image.astype(np.int64), n_colors, euclidean_distance)
    return centroids[labels]


# Call this function inside the main method in task.py to recolor the image!
def process_image(image):
    recolored_image = recolor(image, 8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).astype('uint8')
    image = Image.fromarray(recolored_image)
    image.save("recolored-superman-batman.png")
