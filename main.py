import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, applications
import cv2 as cv
import numpy as np

'''
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

classes= [0,1,2,3,4,5,6,7,8,9]

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
model= models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),'myModel'))
print('### training...')
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=3, validation_data=(testImages, testLables))
model.save(os.path.dirname(os.path.abspath(__file__))+'\\myModel')

# predicting:
print('### predicting...')
for i in range(10):
    image= cv.imread(os.path.dirname(os.path.abspath(__file__))+f'\\test images\\{str(i)}.png')
    image= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    pre= np.expand_dims(image, 0)
    predictions= model.predict([pre])
    maxi= np.where(predictions[0] == np.amax(predictions[0]))[0][0]
    #for i in range(10): print(classes[i], ': ', predictions[0][i], sep='')
    print(f'closest class to {i}: {classes[maxi]}')
