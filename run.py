import tensorflow as tf
import numpy as np
from PIL import Image

from model.google_net import GoogleNet

# Load the input
image = Image.open("./data/castle.jpg")
np_image = np.array(image)
image_tensor = tf.convert_to_tensor(np_image, np.float32)


def main(unused_arguments):
    input_data = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    google_net = GoogleNet({'data': input_data})

    with tf.Session as sess:
        google_net.load('./mynet.npy', sess)
        sess.run(google_net.get_output(), feed_dict={input_data: image_tensor})


if __name__ == "__main__":
    tf.app.run()