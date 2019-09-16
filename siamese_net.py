import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
import numpy as np


def initialize_weights(shape, name = None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def initialize_bias(shape, name = None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_model(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation = 'relu', input_shape = input_shape, kernel_initializer = initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation = 'relu', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation = 'relu', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation = 'relu', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation = 'sigmoid', kernel_regularizer = l2(1e-3), bias_initializer = initialize_bias, kernel_initializer = initialize_weights))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation = 'sigmoid', bias_initializer = initialize_bias)(L1_distance)

    siamese_net = Model(inputs= [left_input, right_input], outputs = prediction)

    return siamese_net
