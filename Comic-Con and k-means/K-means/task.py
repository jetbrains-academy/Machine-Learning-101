from PIL import Image
import numpy as np
from distances import euclidean_distance
from clustering import k_means

def read_image(path='superman-batman.png'):
    # Here we need to read the image using the PIL function open.
    image = Image.open(path)
    # We reshape the image array into one with the (M x N, 3)
    # shape.
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    (centroids, labels) = k_means(image, 4, euclidean_distance)
    print("Cluster centers:")
    for label in labels:
        print(label)