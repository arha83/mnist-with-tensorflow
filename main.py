import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses
import cv2 as cv
import numpy as np

# loading dataset:
print('### loading dataset...')
(trainImages, trainLabels), (testImages, testLables)= datasets.mnist.load_data()
# normalizing dataset:
print('### resizing dataset...')
trainImages, testImages= trainImages/255.0, testImages/255.0