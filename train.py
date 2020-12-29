# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: train.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:

from models.mymodel import MyModel
from models.vgg16 import pretrainVGG16
from models.inceptionv3 import pretrainInceptionv3
from models.resnet50 import pretrainResNet50
import prepare_data
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

BATCH_SIZE = 32
EPOCHS = 10
HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 2
#---------------------------------------------------------------------------------------------------------
# Create model
#---------------------------------------------------------------------------------------------------------
# my model
#model = MyModel.create_model(width=100, height=100, depth=3, classes=2)
#

# VGG16 pretrain
model = pretrainVGG16.VGG16Pretrain(width=WIDTH, height=HEIGHT, depth=DEPTH, classes=NUM_CLASSES)

# InceptionV3 pretrain
#model = pretrain.Inceptionv3Pretrain(width=WIDTH, height=HEIGHT, depth=DEPTH, classes=NUM_CLASSES)

# ResNet50 pretrain
#model = pretrain.ResNet50Pretrain(width=WIDTH, height=HEIGHT, depth=DEPTH, classes=NUM_CLASSES)

model.summary()
print("[INFO] - Created model successfully\n")

#---------------------------------------------------------------------------------------------------------
# Create data
#---------------------------------------------------------------------------------------------------------
# train_ds, val_ds = prepare_data.create_data("/media/thanglmb/Bkav/AICAM/TrainModels/dataset/gender/data", WIDTH, HEIGHT, BATCH_SIZE)
# image_batch, labels_batch = next(iter(train_ds))
# first_image = image_batch[0]

data_dir = "/media/thanglmb/Bkav/AICAM/TrainModels/dataset/gender/data"
train_dir = "/media/thanglmb/Bkav/AICAM/TrainModels/dataset/DogCat/dataset/training_set"
validation_dir = "/media/thanglmb/Bkav/AICAM/TrainModels/dataset/DogCat/dataset/test_set"
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(data_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))

#---------------------------------------------------------------------------------------------------------
# Train model
#---------------------------------------------------------------------------------------------------------
model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=EPOCHS)
test_img_path = "/media/thanglmb/Bkav/AICAM/TrainModels/dataset/gender/men/00000001.jpg"
img = tf.keras.preprocessing.image.load_img(
    test_img_path, target_size=(HEIGHT, WIDTH)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
img_array = img_array / 255.0
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("[INFO] - The testing result: ",score)
#---------------------------------------------------------------------------------------------------------
# Save model and export
#---------------------------------------------------------------------------------------------------------
model.save('ResNet50.h5')
model.save('Savedmodel')
print("[INFO] - Saved model successfully\n")

frozen_out_path = "/media/thanglmb/Bkav/AICAM/TrainModels/TF2/Classifier/export"
frozen_graph_filename = "frozen_graph"
# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
#full_model = full_model.get_concrete_function(
#    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
full_model = full_model.get_concrete_function(tf.TensorSpec(shape=(1, WIDTH, HEIGHT, DEPTH), name='Input', dtype=model.inputs[0].dtype))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def(add_shapes=True)
layers = [op.name for op in frozen_func.graph.get_operations()]
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
# Save its text representation
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir=frozen_out_path,
#                   name=f"{frozen_graph_filename}.pbtxt",
#                   as_text=True)
print("[INFO] - Export graph model successfully\n")

#---------------------------------------------------------------------------------------------------------
# Chart
#---------------------------------------------------------------------------------------------------------
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
'''