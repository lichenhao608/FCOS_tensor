import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Conv2D
from keras.applications.resnet50 import ResNet50
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1


def backbone(img):
    '''
    Input:
        img: a [batch, hight, width, channels] image tensor

    Output:
        features: a list of features getting from layer C3, C4, and C5
    '''

    features = []
    for _ in range(3):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            img, _ = resnet_v1.resnet_v1_101(img)
            features.append(img)
    return features


def fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)


def heads():
    return
