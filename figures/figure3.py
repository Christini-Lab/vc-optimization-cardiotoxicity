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


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs] 

    return np.array(new_vals)


def plot_fig_3ab():
    path = 'exp_data'
    files = ['021921_1_control',
             '031521_2_verapamil',
             '031721_3_verapamil',
             '033021_4_control',
             '040521_1_quinidine',
             '042721_5_quinine']
    drug = 'Drug'


    for type_trial in ['spont', 'paced']:
        if type_trial == 'spont':
            fig, axs = plt.subplots(1, 1, figsize=(12, 8))
            axs = [axs]
        else:
            fig, axs = plt.subplots(1, 1, sharex=True, figsize=(12, 8))
            axs = [axs]

        for f in files:
            cell = ExpDat(path, f, drug)

            if type_trial == 'paced':
                dat = cell.get_single_aps()['Pre-drug']
            if type_trial == 'spont':
                dat = cell.get_single_aps(is_paced=False, ap_window=1500)['Pre-drug']

            window = 10

            t = moving_average(dat['Time (s)'].values, window)
            c = moving_average(dat['Current (pA/pF)'].values, window)
            v = moving_average(dat['Voltage (V)'].values, window)

            axs[0].plot(t*1000, v*1000)#, col[it], label=label[it])

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        axs[0].set_xlabel('Time (ms)', fontsize=14)
        axs[0].set_ylabel('Voltage (mV)', fontsize=14)


        plt.rcParams['svg.fonttype'] = 'none'

        plt.savefig(f'./fig3-data/{type_trial}.svg', format='svg')
        plt.show()


def plot_fig_3c():
    cell_stats = pd.read_csv('exp_data/cell_stats.csv')

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    columns = ['apd20', 'apd90']
    x_labels = ['APD20 (ms)', 'APD90 (ms)']

    for i, ax in enumerate(axs):
        ax.hist(cell_stats[columns[i]], rwidth=.9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.set_xlabel(x_labels[i], fontsize=14)

        if np.mod(i, 2) == 0:
            ax.set_ylabel('Count', fontsize=14)



    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(f'./fig3-data/feature_histograms.svg', format='svg')
    plt.show()


def main():
    plot_fig_3ab()
    plot_fig_3c()


if __name__ == '__main__':
    main()
