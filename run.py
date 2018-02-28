import tensorflow as tf
import preprocessing

from model.vgg_16 import VGG16

input_width = 224
input_height = 224
num_channels = 3

image_path = "./data/stadium.jpg"


def main(unused_arguments):
    # Load the input
    resized_image = preprocessing.image_resizing(image_path, input_width, input_height)
    np_image = preprocessing.from_image_to_np(resized_image).reshape(1, input_width, input_height, num_channels)

    input_data = tf.placeholder(tf.float32, shape=[None, input_width, input_height, 3])

    # Load the net
    net = VGG16({'data': input_data})

    with tf.Session() as sess:
        net.load('data/vgg16_places365.npy', sess)
        predictions = sess.run(net.get_output(), feed_dict={input_data: np_image})

        print"{} - {}".format(predictions.argmax(), predictions[0, predictions.argmax()])


if __name__ == "__main__":
    tf.app.run()