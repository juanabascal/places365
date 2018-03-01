from model.vgg_16 import VGG16
from model.googlenet import GoogleNet

MODELS = {'vgg16', 'googlenet'}


def net_build(data, type):
    if type not in MODELS:
        return None
    if type is 'vgg16':
        return VGG16({'data': data})
    if type is 'googlenet':
        return GoogleNet({'data': data})
