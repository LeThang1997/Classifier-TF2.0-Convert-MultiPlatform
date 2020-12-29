# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: Mymodel.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:

import tensorflow as tf

def create_model(width, height, depth, classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(width, height, depth)))
    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.1))
    model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.1))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(width, height, depth)))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(classes))
    model.add(tf.keras.layers.Activation("softmax"))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

