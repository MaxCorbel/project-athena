"""
@author: miles
"""

import argparse
import numpy as np
import os
import time
import glob

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.data import subsampling
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def collect_raw_prediction(trans_configs, model_configs, data_configs, use_logits=False, output_dir=None):
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
    wds = list(pool.values())
    athena = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.LEARNING.value)
    
    print('>Weak defenses to train model on: ', type(wds), type(wds[0]))

    #get samples and respective labels
    print('>Getting benign samples and labels from: {}'.format(data_configs.get('dir')))
    
    samples_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    labels_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    samples = np.load(samples_file)
    labels = np.load(labels_file)
    
    #Get subsamples, save labels to training directory
    training_dir = data_configs.get('training_dir')
    
    print('>Getting subsamples of benign for training and validation, ratio: {}'.format('.2'))
    
    samples_training, samples_validation, _ = subsampling(samples, labels, 10, ratio=0.2, filepath=training_dir, filename='_labels.npy')
    
    print('>Labels for training and validation saved to {} and {}'.format(training_dir+data_configs.get('training_labels_file'),
                                                                          training_dir+data_configs.get('validation_labels_file')))
    
    #Get raws for training and validation
    print('>Getting raw predictions for training')
    
    raws_training = athena.predict(x=samples_training, raw=True)
    raws_training = np.transpose(raws_training, (1, 0, 2))
    raws_training_file = os.path.join(training_dir, data_configs.get('training_samples_file'))
    np.save(file=raws_training_file,arr=raws_training)
    
    print('>Raw predictions for training saved to {}'.format(training_dir+data_configs.get('training_samples_file')))
    print('>Getting raw predictions for validation')
    
    raws_validation = athena.predict(x=samples_validation, raw=True)
    raws_validation = np.transpose(raws_validation, (1, 0, 2))
    raws_validation_file = os.path.join(training_dir, data_configs.get('validation_samples_file'))
    np.save(file=raws_validation_file,arr=raws_validation)
    
    print('>Raw predictions for validation saved to {}'.format(training_dir+data_configs.get('validation_samples_file')))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/experiment/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/experiment/model-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/data-mnist.json',
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
    output_dir = data_configs.get('training_dir')

    # collect probabilites
    collect_raw_prediction(trans_configs=trans_configs,
                           model_configs=model_configs,
                           data_configs=data_configs,
                           output_dir=output_dir)