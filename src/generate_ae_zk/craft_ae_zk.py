"""
A sample to generate adversarial examples in the context of zero-knowledge threat model.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import sys

import argparse
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import json

from utils.model import load_lenet, load_pool
from utils.file import load_from_json
from utils.metrics import error_rate
from utils.data import subsampling
from attacks.attack import generate
from models.athena import Ensemble, ENSEMBLE_STRATEGY



def generate_ae(model_configs, trans_configs, data, labels, attack_configs, save=False, output_dir=None):
    """
    Generate adversarial examples
    :param model: WeakDefense. The targeted model.
    :param data: array. The benign samples to generate adversarial for.
    :param labels: array or list. The true labels.
    :param attack_configs: dictionary. Attacks and corresponding settings.
    :param save: boolean. True, if save the adversarial examples.
    :param output_dir: str or path. Location to save the adversarial examples.
        It cannot be None when save is True.
    :return:
    """
    cnn = os.path.join(model_configs.get('dir'), model_configs.get('um_file'))
    #Get undefended model
    um = load_lenet(file=cnn, wrap=True)
    
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)
    
    # Load pool of WD's for Ensemble
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=True,
                        wrap=True)
    wds = list(pool.values())
    ens = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)

    
    img_rows, img_cols = data.shape[1], data.shape[2]
    num_attacks = attack_configs.get("num_attacks")
    
    #Get subsample for generating attacks
    print('> Getting subsamples of bs for AE generation')
    data, labels = subsampling(data, labels, 10, ratio=0.2, filepath='samples/')
    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    attack_dict = attack_configs.get('attacks')
    for attack_type in attack_dict:
        attacks = attack_dict[attack_type]
        if not os.path.isdir(os.path.join(output_dir, attack_type)):
            os.mkdir(os.path.join(output_dir, attack_type))
        type_results = {}
        print('> Generating AEs for {}'.format(attack_type))
        for attack in attacks:
            results = {}
            data_adv = generate(model=um,
                                data_loader=data_loader,
                                a_type=attack_type,
                                attack_args=attack
                                )
            # predict the adversarial examples for UM
            print('> Getting predictions from Undefended Model')
            pred_um = um.predict(data_adv)
            pred_um = np.asarray([np.argmax(p) for p in pred_um])

            err_um = error_rate(y_pred=pred_um, y_true=labels)
            results['UM'] = err_um
            print(">>> UM error rate:", err_um)
            
            # predict for ensemble
            print('> Getting predictions from Ensemble')
            pred_ens = ens.predict(data_adv)
            pred_ens = np.asarray([np.argmax(p) for p in pred_ens])
            err_ens = error_rate(y_pred=pred_ens, y_true=labels)
            results['Ensemble'] = err_ens
            print('> Ensemble error rate:', err_ens)
            
            # predict for baseline
            pred_bl = baseline.predict(data_adv)
            pred_bl = np.asarray([np.argmax(p) for p in pred_bl])
            err_bl = error_rate(y_pred=pred_bl, y_true=labels)
            print('> Baseline error rate:',err_bl)
            results['Baseline'] = err_bl
            
            type_results[attack.get('description')] = results
            
            if save:
                if output_dir is None:
                    raise ValueError("Cannot save images to a none path.")
                    
                # plotting an examples
                img = data_adv[0].reshape((img_rows, img_cols))
                plt.imshow(img, cmap='gray')
                title = '{}'.format(attack.get('description'))
                plt.title(title)
                #Save picture of an AE
                plt.savefig(os.path.join(output_dir,'{}/{}.png'.format(attack_type, attack.get('description'))), dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
                # save the adversarial example
                file = os.path.join(output_dir, "{}/{}.npy".format(attack_type, attack.get('description')))
                print("Save the adversarial examples to file [{}].".format(file))
                np.save(file, data_adv)
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")
            with open(os.path.join(output_dir, '{}/fgsm_results.json'.format(attack_type)), 'w') as f:
                json.dump(type_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/demo/model-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/data-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='../configs/experiment/attack-zk-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='examples/',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)
    parser.add_argument('--trans-configs', default='../configs/experiment/athena-mnist.json')

    args = parser.parse_args()

    print("------AUGMENT SUMMARY-------")
    print("MODEL CONFIGS:", args.model_configs)
    print("DATA CONFIGS:", args.data_configs)
    print("ATTACK CONFIGS:", args.attack_configs)
    print("OUTPUT ROOT:", args.output_root)
    print("DEBUGGING MODE:", args.debug)
    print('TRANS CONFIGS:', args.trans_configs)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)
    attack_configs = load_from_json(args.attack_configs)
    trans_configs = load_from_json(args.trans_configs)

    # load the benign samples
    data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    data_bs = np.load(data_file)
    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    generate_ae(model_configs=model_configs, trans_configs=trans_configs, data=data_bs, labels=labels, attack_configs=attack_configs, save=True, output_dir=args.output_root)
