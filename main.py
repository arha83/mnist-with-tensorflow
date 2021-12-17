import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, applications
import cv2 as cv
import numpy as np

# loading dataset:
print('### loading dataset...')
(trainImages, trainLabels), (testImages, testLables)= datasets.mnist.load_data()
# normalizing dataset:
print('### resizing dataset...')
trainImages, testImages= trainImages/255.0, testImages/255.0

# building the model:
print('### building the model...')
model= models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())


# training:
print('### training...')
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=1, validation_data=(testImages, testLables))
model.save(os.path.dirname(os.path.abspath(__file__))+'\\myModel')