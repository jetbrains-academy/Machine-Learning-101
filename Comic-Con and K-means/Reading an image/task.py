from PIL import Image
import numpy as np


def read_image(path='superman-batman.png'):
    # Here we need to read the image using the PIL function open.
    image = #TODO
    # We reshape the image array into one with the (M x N, 3)
    # shape.
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    # Take a look at what the image looks like in the form of an array.
    print(image)
