#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:53:22 2020

@author: miles
"""
import os
import glob
import numpy as np

samples = glob.glob('samples/*test_samples*.npy')
labels = glob.glob('samples/*test_labels*.npy')
sorted_samples = []
sorted_labels = []
for sample in samples:
    pref = sample.split('_test_samples')[0].split('/')[1].replace('.npy','')
    sorted_samples.append(sample)
    for label in labels:
        pref2 = label.split('_test_labels')[0].split('/')[1].replace('.npy','')
        if pref == pref2:
            sorted_labels.append(label)
    
samples_dat = [np.load(dat) for dat in sorted_samples]
labels_dat = [np.load(dat) for dat in sorted_labels]
samples_dat = np.concatenate(samples_dat,axis=0)
labels_dat = np.concatenate(labels_dat)
print(samples_dat.shape)
samples = []
labels = []
for i in range(0, len(labels_dat), 90):
    samples.append(samples_dat[i])
    labels.append(labels_dat[i])
labels_file = os.path.join('testing/', 'test_labels.npy')
samples_file = os.path.join('testing/', 'test_samples.npy')
labels = np.array(labels)
samples = np.array(samples)
print(samples.shape)
np.save(file=labels_file,arr=labels)
np.save(file=samples_file,arr=samples)
    