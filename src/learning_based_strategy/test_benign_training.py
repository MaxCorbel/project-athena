#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:27:21 2020

@author: miles
"""
import argparse
import numpy as np
import os
import time
import json

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.data import subsampling
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def evaluate_strategy(trans_configs, model_configs,
                 data_configs, save=False, output_configs=None):

    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # get the CNN undefended model (UM)
    cnn_configs = model_configs.get('cnn')
    file = os.path.join(cnn_configs.get('dir'), cnn_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load weak defenses into a pool
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=cnn_configs,
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble and learning based strategy from the WD pool
    wds = list(pool.values())
    print(">>> wds:", type(wds), type(wds[0]))
    learning_based = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.LEARNING.value)
    ensemble = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)
    
    '''
    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)
    '''

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)
    
    '''
    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)
    '''

    output_dir = output_configs.get('dir')
    ae_outputs = output_configs.get('files_by_ae')
    ae_list = data_configs.get('ae_files')
    
    # Evaluate AEs.
    i = 0
    while i < len(ae_list):
        
        #Get AE
        results = {}
        ae_list = data_configs.get('ae_files')
        ae_file_name = ae_list[i]
        ae_variant = ae_file_name.rsplit('-', 1)[-1].split('_',1)[-1].replace('.npy','')
        ae_file = os.path.join(data_configs.get('dir'), ae_file_name)
        x_adv = np.load(ae_file)
        
        #Get subsample of AE
        ae_sub, _, labels_sub = subsampling(x_adv, labels, 10, 0.8)

        # evaluate the undefended model on the AE
        print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
        pred_adv_um = undefended.predict(ae_sub)
        err_um = error_rate(y_pred=pred_adv_um, y_true=labels_sub)
        # track the result
        results['UM'] = err_um
        
        # evaluate ensemble on the AE
        print('>>> Evaluating Ensemble on [{}], it may take a while...'.format(ae_file))
        pred_adv_ens = ensemble.predict(ae_sub)
        err_ens = error_rate(y_pred=pred_adv_ens, y_true=labels_sub)
        results['Ensemble'] = err_ens
        
        # evaluate the learning based strategy on the AE
        print(">>> Evaluating learning based model on [{}], it may take a while...".format(ae_file))
        pred_adv_lrn = learning_based.predict(ae_sub)
        err_lrn = error_rate(y_pred=pred_adv_lrn, y_true=labels_sub)
        # track the result
        results['Learning based'] = err_lrn
        
        # evaluate the baseline on the AE
        print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
        pred_adv_bl = baseline.predict(ae_sub)
        err_bl = error_rate(y_pred=pred_adv_bl, y_true=labels_sub)
        # track the result
        results['PGD-ADT'] = err_bl
        
        i = i + 1
        
        if save:
            for key in ae_outputs:
                if key in ae_file_name:
                    outfile = os.path.join(output_dir, ae_outputs.get(key))
                    if os.path.exists(outfile):
                        with open(outfile, 'r+') as f:
                            data = json.load(f)
                            data[ae_variant] = results
                            f.seek(0)
                            json.dump(data, f)
                    else:
                        with open(outfile, 'w') as f:
                            data = {}
                            data[ae_variant] = results
                            json.dump(data, f)
        print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")


    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/experiment/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/demo/hybrid-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/data-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='../configs/experiment/results.json',
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
    results_configs = load_from_json(args.output_root)

    evaluate_strategy(trans_configs=trans_configs,
                 model_configs=model_configs,
                 data_configs=data_configs,
                 save=True,
                 output_configs=results_configs)
