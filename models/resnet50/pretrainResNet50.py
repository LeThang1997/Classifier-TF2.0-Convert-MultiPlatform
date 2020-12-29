# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: pretrain.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:  ResNet50 pretrain and transfer learning

from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf

def ResNet50Pretrain(width, height, depth, classes):
    base_model = ResNet50(input_shape=(width, height, depth), include_top=False, weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(base_model.input, x)
    model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])
    return model