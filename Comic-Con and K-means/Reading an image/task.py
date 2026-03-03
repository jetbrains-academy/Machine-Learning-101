from PIL import Image
import numpy as np


def read_image(path='superman-batman.png'):
    # Here, we load the image using PIL's open function.
    image = Image.open(path)
    # We reshape the image into an (M x N, 3)
    # array.
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    # Take a look at what the image looks like in the form of an array.
    print(image)
