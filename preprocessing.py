from PIL import Image
import numpy as np
from resizeimage import resizeimage


def from_image_to_np(image):
    image_np = np.array(image)

    return image_np


def image_resizing(path, input_width, input_height):
    image = Image.open(path)
    resized_image = resizeimage.resize_cover(image, [input_width, input_height])

    return resized_image
