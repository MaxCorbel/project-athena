#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:14:50 2020

@author: miles
"""
import numpy as np

samples = np.load('training_data/training.npy')
labels = np.load('training_data/training_labels.npy')
print(samples.shape)
print(labels.shape)
