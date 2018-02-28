# TF Places365
Using places365 pre-trained models in TensorFlow. Previously converted from Caffe to TensorFlow format.
## Dependencies
You must need to have the following packages installed in order to run the predictor.
* Python 2.7
* TensorFlow
* Numpy
* Pillow
* resizeimage
## Getting the model
You will need to download the places365 model in Caffe from their GitHub repository https://github.com/CSAILVision/places365. You will need both the deploy file and the weights file. Moreover, you can find here the labels for the different categories of places365.

After that you need to convert the Caffe model to a TensorFlow model using the https://github.com/ethereon/caffe-tensorflow caffe-tensorflow converter. You can find a guide to use the conversor here https://docs.google.com/document/d/1UhPTyDTFJHDx94yDH8x_dCnNpM9K9irAKZSvn97ps6c/edit?usp=sharing.
## Running
For running the program you will need to declare the image path, the converted weights paths and the labels path. Note that the input's width and height are set for a VGG16 model, you may have to adjust them to your model's input.
