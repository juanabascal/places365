from model.vgg_16 import VGG16
from model.googlenet import GoogleNet

MODELS = {'vgg16', 'googlenet'}


class Helper:

    def __init__(self, model_name):
        if model_name is 'vgg16':
            self.net = VGG16
        if model_name is 'googlenet':
            self.net = GoogleNet

    def net_build(self, data):
        return self.net({'data': data})
