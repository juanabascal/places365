import tensorflow as tf
import preprocessing
import numpy as np

import sys

import model.helper as model_helper

image_path = "./data/castle.jpg"
weights_path = "./data/vgg16_places365.npy"
labels_path = "./data/labels.txt"


def main(unused_arguments):
    helper = model_helper.Helper('vgg16')

    # Load the input
    resized_image = preprocessing.image_resizing(image_path, helper.net.input_width, helper.net.input_height)
    np_image = preprocessing.from_image_to_np(resized_image).reshape(1,
                                                                     helper.net.input_width,
                                                                     helper.net.input_width,
                                                                     helper.net.num_channels)

    input_data = tf.placeholder(tf.float32, shape=[None,
                                                   helper.net.input_width,
                                                   helper.net.input_height,
                                                   helper.net.num_channels])

    net = helper.net_build(input_data)

    with tf.Session() as sess:
        net.load(weights_path, sess)
        predictions = sess.run(net.get_output(), feed_dict={input_data: np_image})
        print_results(predictions)


def print_results(predictions):
    # Convert predictions to numpy
    np_pred = np.array(predictions).flatten()

    # Take the indexes of the 5 highest values
    np_index = np_pred.argsort()[-5:][::-1]

    # Load the labels in memory
    labels = open(labels_path, "r").readlines()

    for i in range(0, 5):
        print"Es un {} con probabilidad de {}".format(labels[np_index[i]], np_pred[np_index[i]])
        print"---------------"


if __name__ == "__main__":
    tf.app.run()
