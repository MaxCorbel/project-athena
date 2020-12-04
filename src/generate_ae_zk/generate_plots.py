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

attack_type = input('enter attack type: ')

results = load_from_json('examples/{}/{}_results.json'.format(attack_type, attack_type))
um = []
ens = []
pgdadt = []
for key in results:
    var_res = results[key]
    um.append(var_res['UM'])
    ens.append(var_res['Ensemble'])
    pgdadt.append(var_res['Baseline'])
x=np.arange(len(results))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x, um, width=.25, label='UM')
ax.bar(x+.25, ens, width=.25, label='Ensemble', align='center')
ax.bar(x+.5, pgdadt, width=.25, label='Baseline')
plt.xticks(range(len(results)), list(results.keys()))
plt.yscale('log')
ax.set_ylabel('Error rate')
ax.set_xlabel('Variant')
ax.set_title(attack_type)
plt.legend()
plt.savefig(os.path.join('examples/{}'.format(attack_type), '{}.png'.format(attack_type)), dpi=300, bbox_inches='tight')
plt.show()

        