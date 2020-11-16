#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:20:14 2020

@author: miles
"""

import numpy as np
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


training_data = np.load('../training_data/training.npy')
training_labels = np.load('../training_data/labels2D.npy')

network = models.Sequential()
network.add(layers.Dense(150, activation='relu', input_shape=(15 * 10,)))
network.add(layers.Dense(600, activation='relu', input_shape=(150 * 4,)))
network.add(layers.Dense(150, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


training_data = np.transpose(training_data, (1,0,2))
training_data = training_data.reshape((3600,15*10))

network.fit(training_data, training_labels, epochs=10, batch_size=360)
network.save('learning-strategy-nn.h5')

