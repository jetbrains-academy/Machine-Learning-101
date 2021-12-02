from PIL import Image
import numpy as np
from distances import euclidean_distance
from clustering import k_means


def read_image(path='superman-batman.png'):
    image = Image.open(path)
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    (centroids, labels) = k_means(image, 4, euclidean_distance)
    print("Cluster centers:")
    for label in labels:
        print(label)
