#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:20:14 2020

@author: miles
"""
import os
import argparse
from utils.file import load_from_json
from utils.data import subsampling
import numpy as np
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import backend as K

def train(num_wds, training_data, training_labels, validation_data, validation_labels):
    
    network = models.Sequential()
    network.add(layers.Dense(num_wds * 10, activation='tanh'))#, input_shape=(num_wds * 10,)))
    network.add(layers.Dense(num_wds * 40, activation='tanh'))#, input_shape=(num_wds * 40,)))
    network.add(layers.Dense(num_wds * 20, activation='relu'))
    network.add(layers.Dense(num_wds * 10, activation='relu'))
    network.add(layers.Dense(10, activation='sigmoid'))
    network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('>Training')
    history = network.fit(training_data, training_labels, epochs=10, batch_size=100)
    history.history
    
    print('>Evaluate on validation data')
    results = network.evaluate(validation_data, validation_labels, batch_size=100)
    print('>Test loss, Test acc:', results)
    
    network.save('learning-strategy-nn.h5')
