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



def get_cells():
    """
        returns Beautiful table and all cells in cell_objects list
    """
    table = BeautifulTable()
    table.column_headers = ["#", "Drug", "Min(Rm)", "Rm/Ra", "AP", "VC", "Drug Exp", "Comp", "OED"]


    path = f'../analyze_experiments/results'
    files = [f for f in listdir(f'{path}/cells')
            if 'h5' in f]
    files = [f.split('.')[0] for f in files]
    files.sort()
    cell_objects = []
    for j, f in enumerate(files):
        f = f.split('.')[0]
        drug = f.split('_')[-1].capitalize()

        cell_object = ExpDat(path, f, drug)

        art_params = cell_object.artefact_parameters

        vc_rm = ''

        if 'vcp_70_70' in cell_object.trials['Pre-drug'].keys():
            phases = ['Pre-drug']
            if 'vcp_70_70' in cell_object.trials['Post-drug'].keys():
                phases.append('Post-drug')

            vc_rm = ''
            for i, tri in enumerate(phases):
                trial = cell_object.trials[tri]['vcp_70_70']
                rm_b = cell_object.artefact_parameters[:, 0] <= trial
                rm = cell_object.artefact_parameters[:, 3][rm_b][-1]
                if i == 0:
                    vc_rm += f'{rm}'
                else:
                    vc_rm += f'-->{rm}' 

        min_rm_over_ra = (art_params[:, 3] / art_params[:, 2]).min()

        is_ap = ('Y' if 'paced' in cell_object.trials['Pre-drug'].keys() else ' ')
        is_vc = ('Y' if 'vcp_70_70' in
                cell_object.trials['Pre-drug'].keys() else ' ')
        is_drug = ('Y' if 'paced' in 
                cell_object.trials['Post-drug'].keys() else ' ')
        is_comp = ('Y' if 'rscomp_80' in 
                cell_object.trials['Compensation'].keys() else ' ')
        is_oed = ('Y' if 'proto_O' in 
                cell_object.trials['Pre-drug'].keys() else ' ')

        table.append_row([j, cell_object.drug, vc_rm, min_rm_over_ra,
            is_ap, is_vc, is_drug, is_comp, is_oed])
        cell_objects.append(cell_object)

    return table, cell_objects, files


def pick_cell():
    print(table)

    which_cell = int(input("Which cell would you like to plot? "))

    print(f"You selected: {files[which_cell]}")

    return 


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs] 

    return np.array(new_vals)


def plot_fig_3ab():
    table, cell_objects, files = get_cells()
    which_cells = [13, 30, 35, 40, 49, 66]

    for type_trial in ['spont', 'paced']:
        if type_trial == 'spont':
            fig, axs = plt.subplots(1, 1, figsize=(12, 8))
            axs = [axs]
        else:
            fig, axs = plt.subplots(1, 1, sharex=True, figsize=(12, 8))
            axs = [axs]

        for cell_num in which_cells:
            cell = cell_objects[cell_num]

            if type_trial == 'paced':
                dat = cell.get_single_aps()['Pre-drug']
            if type_trial == 'spont':
                dat = cell.get_single_aps(is_paced=False, ap_window=1500)['Pre-drug']

            window = 10

            t = moving_average(dat['Time (s)'].values, window)
            c = moving_average(dat['Current (pA/pF)'].values, window)
            v = moving_average(dat['Voltage (V)'].values, window)

            #if type_trial != 'spont':
            #    axs[1].plot(t*1000, c)#, col[it], label=label[it])


            axs[0].plot(t*1000, v*1000, label=cell_num)#, col[it], label=label[it])

            #it += 1


        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        #if type_trial != 'spont':
        #    axs[1].set_xlabel('Time (ms)', fontsize=14)
        #    axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
        #    axs[0].set_ylabel('Voltage (mV)', fontsize=14)
        #else:
        axs[0].set_xlabel('Time (ms)', fontsize=14)
        axs[0].set_ylabel('Voltage (mV)', fontsize=14)


        #if type_trial == 'spont':
        #    plt.legend(fontsize=14)

        plt.rcParams['svg.fonttype'] = 'none'

        plt.savefig(f'./fig3-data/{type_trial}.svg', format='svg')
        #axs[0].legend()
        plt.show()


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

    plt.savefig(f'./fig3-data/feature_histograms.svg', format='svg')
    plt.show()


def main():
    plot_fig_3ab()
    plot_fig_hists()


if __name__ == '__main__':
    main()
