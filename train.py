#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:42:46 2019

@author: marki

Implementation of Training Script 
"""

import matplotlib 
matplotlib.use("Agg")

from pyimagesearch.resnet import ResNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True, 
                help = "path to input dataset")
ap.add_argument("-a", "--augment", type = int, default = -1,
                help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument("-p", "--plot", type = str, default = "plot.png",
                help="path to output loss/accuracy plot")

args = vars(ap.parse_args())

# Initialize the initial learning rate, batch size, and number of epochs
# to train for

INIT_LR = 1e-1
BATCH_SIZE = 8
EPOCHS = 50

print("[INFO] loading images .. ")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    # Extract the class label from the filename, load the image, and
    # Resize it to be a fixed 64x64 pixels
    
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(64,64))
    
    data.append(image)
    labels.append(label)
# Convert the data int a Numpy Array, then preprocess it by scaling 
# all pixel intensities between [0,1]

data = np.array(data, dtype = "float")/255.0


le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels,2)

# Partition dat into training and testing splits using 75% of the 
# data for training and the remaining 25% for testing

trainX, testX, trainY,testY = train_test_split(data, labels,
                                               test_size=.25, randome_state=42)





