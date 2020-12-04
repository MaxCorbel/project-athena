#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:05:15 2020

@author: miles
"""

import os
import json
from utils.file import load_from_json
import matplotlib.pyplot as plt
import numpy as np

results_configs = load_from_json('../configs/experiment/results.json')
results_dir = results_configs.get('dir')
ae_type_files = results_configs.get('files_by_ae')

for ae_type in ae_type_files:
    res_file = ae_type_files[ae_type]
    results = load_from_json(os.path.join(results_dir, res_file))
    um = []
    ens = []
    lrn = []
    pgdadt = []
    for key in results:
        var_res = results[key]
        um.append(var_res['UM'])
        ens.append(var_res['Ensemble'])
        lrn.append(var_res['Learning based'])
        pgdadt.append(var_res['PGD-ADT'])
    x=np.arange(len(results))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x, um, width=.25, label='UM')
    ax.bar(x+.25, ens, width=.25, label='Ensemble', align='center')
    ax.bar(x+.5, lrn, width=.25, label='Learning based')
    ax.bar(x+.75, pgdadt, width=.25, label='PGD-ADT')
    plt.xticks(range(len(results)), list(results.keys()))
    plt.yscale('log')
    ax.set_ylabel('Error rate')
    ax.set_xlabel('Variant')
    ax.set_title(ae_type)
    plt.legend()
    plt.savefig(os.path.join(results_dir, '{}.png'.format(ae_type)), dpi=300, bbox_inches='tight')
    plt.show()

        