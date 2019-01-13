from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Reshape, Lambda
from keras.models import Model
import tensorflow as tf


def l2_normalize(x):
    return tf.nn.l2_normalize(x, dim=2)


def bbox_3D_net(input_shape=(224, 224, 3),bin_num=2):
    inputs = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)

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

    model = Model(inputs, outputs=[dimension, orientation, confidence])
    return model


