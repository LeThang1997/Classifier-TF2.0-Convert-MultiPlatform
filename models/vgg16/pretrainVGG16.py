# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: pretrain.py
# Author: ThangLMb
# Created: 28/12/2020
# Description: VGG16 pretrain and transfer learning

from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf

def VGG16Pretrain(width, height, depth, classes):
    #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    base_model = VGG16(input_shape = (width, height, depth), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False
        print(layer)
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(base_model.input, x)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
    return model