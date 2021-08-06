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

from cell_models import kernik, paci_2018, ohara_rudy
from cell_models.ga import target_objective
from cell_models.rtxi.rtxi_data_exploration import explore_data, get_exp_as_df
from cell_models.ga.target_objective import TargetObjective
from cell_models import protocols

from cell_objects import *


VIABLE_CELLS = {'control': [7, 13, 14 ,25 ,27 ,28 ,40 ,41 ,43 ,44],
                'cisapride': [8 , 16 , 18 , 21 , 22 , 23],
                'verapamil': [30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38],
                'quinidine': [48 , 49 , 51 , 52 , 54 , 55],
                'quinine': [57 , 58 , 61 , 62 , 63 , 64 , 65 , 66 , 67]}

def plot_fig_hists():
    cell_stats = pd.read_csv('../analyze_experiments/results/cell_stats.csv')
    #plt.style.use('seaborn')

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs = [item for sublist in axs for item in sublist]
    columns = ['rmp', 'apa', 'apd20', 'apd90']
    x_labels = ['RMP (mV)', 'APA (mV)', 'APD20 (ms)', 'APD90 (ms)']

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

    plt.savefig(f'./figS9-data/feature_histograms.svg', format='svg')
    plt.show()
