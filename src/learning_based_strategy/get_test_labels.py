#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:08:08 2020

@author: miles
"""
import numpy as np

file = 'training_data/training_labels.npy'
labels = np.load(file)
arr = []
for label in labels:
    temp = [0,0,0,0,0,0,0,0,0,0]
    temp[label] = 1;
    arr.append(temp)
    
new_arr = np.array(arr)
np.save(file='training_data/labels2D.npy', arr=new_arr)
