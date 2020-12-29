
# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: prepare_data.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:

import tensorflow as tf
import cv2

def create_data(data_dir, width, height, batch_size):
    #data_dir = "/media/thanglmb/Bkav/AICAM/TrainModels/TF2/dataset/gender/data"
    print("[INFO] - Directory data: ", data_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print("[INFO] - List of classes: ", class_names)
    with open('labels.txt', 'w') as f:
        for item in class_names:
            f.write("%s\n" % item)
    print("[INFO] - Create labels file seccessfully\n ")
 
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # normalize data
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_normalized_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    print("[INFO] - Prepared data for training successfully\n ")
    return train_normalized_ds, val_normalized_ds
