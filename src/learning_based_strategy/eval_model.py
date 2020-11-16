"""
A sample to evaluate model on dataset
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import argparse
import numpy as np
import os
import time
import json

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def evaluate_cnn(trans_configs, model_configs,
                 data_configs, save=False, output_dir=None):
    """
    Apply transformation(s) on images.
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    cnn_configs = model_configs.get('cnn')
    file = os.path.join(cnn_configs.get('dir'), cnn_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)

    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=cnn_configs,
                        active_list=True,
                        wrap=True)
    wds = list(pool.values())
    ensemble = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.LEARNING.value)
    
    samples = 'testing/test_samples.npy'
    samples = np.load(samples)
    
    labels = 'testing/test_labels.npy'
    labels = np.load(labels)

    results = {}
    pred_um = undefended.predict(samples)
    err_um = error_rate(y_pred=pred_um, y_true=labels)
    results['Undefended'] = err_um


    pred_ens = ensemble.predict(samples)
    err_ens = error_rate(y_pred=pred_ens, y_true=labels)
    results['Ensemble'] = err_ens
    
    pred_bl = baseline.predict(samples)
    err_bl = error_rate(y_pred=pred_bl, y_true=labels)
    results['PGD-ADT'] = err_bl

    res = json.dumps(results)
    f = open('results.json', 'w')
    f.write(res)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    """
    configurations under ``configs/demo`` are for demo.
    """

    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/experiment/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/demo/hybrid-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/demo/data-mnist.json',
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

    # -------- Evaluate CNN ATHENA -------------
    evaluate_cnn(trans_configs=trans_configs,
                 model_configs=model_configs,
                 data_configs=data_configs,
                 save=False,
                 output_dir=args.output_root)

