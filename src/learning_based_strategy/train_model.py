#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:50:25 2020

@author: miles
"""
import os
import argparse
from utils.file import load_from_json
from utils.data import get_indv_correct_labels
from models.nn import train
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/experiment/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/data-mnist.json',
                        help='Folder where test data stored in.')

    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()
    
    trans_configs = load_from_json(args.trans_configs)
    data_configs = load_from_json(args.data_configs)
    num_wds = len(trans_configs.get('active_wds'))
    
    training_dir = data_configs.get('training_dir')
    t_data = os.path.join(training_dir, data_configs.get('training_samples_file'))
    t_data = np.load(t_data)
    t_data = np.reshape(t_data, (len(t_data), num_wds*10))
    t_labels = os.path.join(training_dir, data_configs.get('training_labels_file'))
    t_labels = np.load(t_labels)
    #t_labels = get_indv_correct_labels(t_labels)
    
    v_data = os.path.join(training_dir, data_configs.get('validation_samples_file'))
    v_data = np.load(v_data)
    v_data = np.reshape(v_data, (len(v_data), num_wds*10))
    v_labels = os.path.join(training_dir, data_configs.get('validation_labels_file'))
    v_labels = np.load(v_labels)
    #v_labels = get_indv_correct_labels(v_labels)
    
    train(num_wds, t_data, t_labels, v_data, v_labels)