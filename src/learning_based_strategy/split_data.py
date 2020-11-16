#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:49:52 2020

@author: miles
"""
import os
from utils.file import load_from_json
from utils.data import subsampling
import numpy as np

data_configs = load_from_json('../configs/experiment/data-mnist.json')

path = 'samples/'

#get data files, only take the AE type for the filename for matching later
data_files = [os.path.join(data_configs.get('dir'), ae_file) for ae_file in data_configs.get('ae_files')]
filenames = [ae_file.split('-')[-1].replace('.npy','') for ae_file in data_configs.get('ae_files')]
#Get respective label files
label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
labels = np.load(label_file)

#Subsample from each AE file
for file, filename in zip(data_files, filenames):

    data = np.load(file)
    subsampling(data, labels, 10, 0.2, path, filename)
