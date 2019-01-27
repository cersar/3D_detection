from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Reshape, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
import tensorflow as tf


def l2_normalize(x):
    return tf.nn.l2_normalize(x, dim=2)


def bbox_3D_net(input_shape=(224, 224, 3), vgg_weights=None, freeze_vgg=False, bin_num=6):
    vgg16_model = VGG16(include_top=False, weights=vgg_weights, input_shape=input_shape)

    if freeze_vgg:
        for layer in vgg16_model.layers:
            layer.trainable = False

    x = Flatten()(vgg16_model.output)

    dimension = Dense(512)(x)
    dimension = LeakyReLU(alpha=0.1)(dimension)
    dimension = Dropout(0.5)(dimension)
    dimension = Dense(3)(dimension)
    dimension = LeakyReLU(alpha=0.1, name='dimension')(dimension)

    orientation = Dense(256)(x)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.5)(orientation)
    orientation = Dense(bin_num * 2)(orientation)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Reshape((bin_num, -1))(orientation)
    orientation = Lambda(l2_normalize, name='orientation')(orientation)

    confidence = Dense(256)(x)
    confidence = LeakyReLU(alpha=0.1)(confidence)
    confidence = Dropout(0.5)(confidence)
    confidence = Dense(bin_num, activation='softmax', name='confidence')(confidence)

    model = Model(vgg16_model.input, outputs=[dimension, orientation, confidence])
    return model


