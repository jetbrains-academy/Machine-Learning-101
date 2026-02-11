from PIL import Image
import numpy as np
from processing import recolor

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024


def read_image(path='superman-batman.png'):
    # Here we need to read the image using the PIL function open.
    image = Image.open(path)
    # We reshape the image array into one with the (M x N, 3)
    # shape.
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    recolored_image = recolor(image, 8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).astype('uint8')
    image = Image.fromarray(recolored_image)
    image.save("recolored-superman-batman.png")
