# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: test-h5.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import ImageFont, ImageDraw, Image
from numpy import exp

class_names = ['man', 'woman']
#class_names = ['cats', 'dogs', 'woman']
#class_names = ['black', 'blue', 'brown', 'green', 'purple', 'red', 'white', 'yellow']
model = load_model('/media/thanglmb/Bkav/AICAM/TrainModels/TF2/Classifier/ResNet50.h5')
model.summary()
# calculate the softmax of a vector
def softmax(vector):
	e = exp(vector)
	return e / e.sum()
    
#image_path = '/media/thanglmb/DATA/MyProject/AI_DL/DogCat-Classifier/archive/test/cats/cat.10.jpg'
image_path = '/media/thanglmb/Bkav/AICAM/TrainModels/dataset/gender_dataset_face/man/man_33.jpg'
img1 = cv2.imread(image_path, 1)
img_color = cv2.resize(img1 , (500,500))
img = image.load_img(image_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
img_array = tf.expand_dims(img_tensor, 0) # Create a batch
#predict
'''
# if non softmax
predictions = model.predict(img_tensor)
print("precdict", predictions)
print("precdict[0]", predictions[0])
#score = tf.nn.softmax(predictions[0])
score = softmax(predictions)
print("SUM: ", np.sum(score))
print("Score", score)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
'''
predictions = model.predict(img_tensor)
print("precdict", predictions)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(predictions)], 100 * np.max(predictions))
)
cv2.putText(img_color,class_names[np.argmax(predictions)],(200,480),cv2.FONT_HERSHEY_COMPLEX,2,(20,3,97),3)

while(True):
    cv2.imshow("Color", img_color)
    #cv2.imwrite("/media/thanglmb/ThangLMb/MyProject/AI/TF2/GenderClassifier/testColor/test.png", img_color)
    if cv2.waitKey(0)==27:
        break
    #closing all open windows  
    cv2.destroyAllWindows() 
# print(
#     "This image is %.2f percent woman and %.2f percent man."
#     % (100 * (1 - score), 100 * score)
# )

# #plt.imshow(img_tensor[0])
# plt.matshow(first_layer_activation[0,: , :, 31], cmap='viridis')
# plt.show()
