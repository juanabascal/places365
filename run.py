import tensorflow as tf
import numpy as np
from PIL import Image
import preprocessing

from model.google_net import GoogleNet
from model.vgg_16 import VGG16

input_width = 224
input_height = 224
num_channels = 3

image_path = "./data/castle.jpg"

# Load the input
resized_image = preprocessing.image_resizing(image_path, input_width, input_height)
np_image = preprocessing.from_image_to_np(resized_image).reshape(1, input_width, input_height, num_channels)

def main(unused_arguments):
    input_data = tf.placeholder(tf.float32, shape=[None, input_width, input_height, 3])

    net = VGG16({'data': input_data})

    sess = tf.Session()
    net.load('data/vgg16_places365.npy', sess)
    sess.run(net.get_output(), feed_dict={input_data: np_image})


if __name__ == "__main__":
    tf.app.run()