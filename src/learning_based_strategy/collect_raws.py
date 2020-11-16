"""
Code pieces for collecting raw values from WDs on the input(s).
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import argparse
import numpy as np
import os
import time
import glob

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def collect_raw_prediction(trans_configs, model_configs, data_configs, use_logits=False, output_dir='testing'):
    """

    :param trans_configs:
    :param model_configs:
    :param data_configs:
    :param use_logits: Boolean. If True, the model will return logits value (before ``softmax``),
                    return probabilities, otherwise.
    :return:
    """
    # load the pool and create the ensemble
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=True,
                        use_logits=use_logits,
                        wrap=True
                        )
    athena = Ensemble(classifiers=list(pool.values()),
                      strategy=ENSEMBLE_STRATEGY.MV.value)

    #get samples and respective labels
    samples = glob.glob('samples/*training_samples*.npy')
    labels = glob.glob('samples/*training_labels*.npy')
    
    #sort based on type of attack
    sorted_samples = []
    sorted_labels = []
    for sample in samples:
        pref = sample.split('_training_samples')[0].split('/')[1].replace('.npy','')
        sorted_samples.append(sample)
        for label in labels:
            pref2 = label.split('_training_labels')[0].split('/')[1].replace('.npy','')
            if pref == pref2:
                sorted_labels.append(label)
    
    #load data and labels, concatenate into single numpy array for easy looping
    samples_dat = [np.load(dat) for dat in sorted_samples]
    labels_dat = [np.load(dat) for dat in sorted_labels]
    samples_dat = np.concatenate(samples_dat,axis=0)
    labels_dat = np.concatenate(labels_dat)
    samples = []
    labels = []
    #Generate raw predictions from each WD for each AE
    for i in range(0, len(labels_dat), 100):
        raw_preds = athena.predict(x=samples_dat[i], raw=True)
        samples.append(raw_preds)
        labels.append(labels_dat[i])

    #Write out raw predictions to training_data directory
    samples = np.concatenate(samples,axis=1)
    labels = np.array(labels)
    samples_file = os.path.join('training_data/', 'training.npy')
    np.save(file=samples_file,arr=samples)
    labels_file = os.path.join('training_data/','training_labels.npy')
    np.save(file=labels_file,arr=labels)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    """
    configurations under ``configs/demo`` are for demo.
    """

    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/experiment/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/experiment/model-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/demo/data-mnist.json',#'samples/',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()

    print('------AUGMENT SUMMARY-------')
    print('TRANSFORMATION CONFIGS:', args.trans_configs)
    print('MODEL CONFIGS:', args.model_configs)
    print('DATA CONFIGS:', args.data_configs)
    print('OUTPUT ROOT:', args.output_root)
    print('DEBUGGING MODE:', args.debug)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    trans_configs = load_from_json(args.trans_configs)
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)
 #   data_configs = args.data_configs

    # collect probabilites
    collect_raw_prediction(trans_configs=trans_configs,
                           model_configs=model_configs,
                           data_configs=data_configs)