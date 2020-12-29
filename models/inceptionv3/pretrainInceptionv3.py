# Copyright (C) 2020 LE MANH THANG. All right reserved.
# Module: pretrain.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:  Inception V3 pretrain and transfer learning

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

def Inceptionv3Pretrain(width, height, depth, classes):
    base_model = InceptionV3(input_shape = (width, height, depth), include_top = False, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(base_model.input, x)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])
    return model
