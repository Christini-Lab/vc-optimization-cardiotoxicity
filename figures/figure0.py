from os import listdir, mkdir
import h5py
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable
import numpy as np
from scipy.stats import ttest_ind
from operator import itemgetter
from itertools import groupby
import pickle
import pandas as pd
import seaborn as sns

from figs_cell_objects import ExpDat

import os
up1 = os.path.abspath('..')
os.sys.path.insert(0, up1)
from mod_kernik import KernikModel
from mod_protocols import SpontaneousProtocol


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]

    return np.array(new_vals)


def plot_fig_0a():
    path = 'exp_data'
    files = ['021921_1_control',
             '031521_2_verapamil',
             '031721_3_verapamil',
             '033021_4_control',
             '040521_1_quinidine',
             '042721_5_quinine']
    drug = 'Drug'


    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    type_trial = 'spont'

    for f in files:
        cell = ExpDat(path, f, drug)

        dat = cell.get_single_aps(is_paced=False, ap_window=1500)['Pre-drug']

        window = 10

        t = moving_average(dat['Time (s)'].values, window)
        c = moving_average(dat['Current (pA/pF)'].values, window)
        v = moving_average(dat['Voltage (V)'].values, window)

        ax.plot(t*1000, v*1000)#, col[it], label=label[it])

    ax.set_ylim(-80, 45)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)


    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(f'./fig0-data/{type_trial}.svg', format='svg')
    plt.show()


def plot_fig_0b():
    mod = KernikModel()
    proto = SpontaneousProtocol()

    tr = mod.generate_response(proto, is_no_ion_selective=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    idx_start = np.abs((tr.t-250)).argmin()
    idx_end = np.abs((tr.t-1250)).argmin()
    ax.plot(tr.t[idx_start:idx_end]-505, tr.y[idx_start:idx_end], 'k')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)

    ax.set_ylim(-80, 45)
    ax.set_xlim(-1000, 1000)
    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(f'./fig0-data/kernik.svg', format='svg')
    plt.show()


plot_fig_0a()
plot_fig_0b()
