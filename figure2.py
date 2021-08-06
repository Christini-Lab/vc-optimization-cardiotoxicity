from os import listdir, mkdir
import h5py
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable
import numpy as np
from scipy.stats import ttest_ind
from operator import itemgetter
from itertools import groupby
import pickle

from cell_models import kernik, paci_2018, ohara_rudy
from cell_models.ga import target_objective
from cell_models.rtxi.rtxi_data_exploration import explore_data, get_exp_as_df
from cell_models.ga.target_objective import TargetObjective
from cell_models import protocols

from cell_objects import *


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


def plot_fig_2():
    table, cell_objects, files = get_cells()
    which_cell = 65

    cell = cell_objects[which_cell]

    col = ['k', 'r']
    label = ['No Drug', 'Quinine']

    fig, axes = plt.subplots(5, 1,
            gridspec_kw={'height_ratios': [8, 4, 4, 4, 4]}, figsize=(4.5, 9))

    for i, type_trial in enumerate(['spont', 'paced', 'vcp_70_70']):
        if type_trial == 'spont':
            axs = [axes[0]]
        else:
            if type_trial == 'paced':
                axs = axes[1:3]
            else:
                axs = axes[3:5]

        nd_trial = cell.trials['Pre-drug'][type_trial]
        drug_trial = cell.trials['Post-drug'][type_trial]

        if type_trial == 'paced':
            dats = cell.get_single_aps()
        if type_trial == 'spont':
            dats = cell.get_single_aps(is_paced=False)
        if type_trial == 'vcp_70_70':
            dats = cell.get_vc_data()

        it = 0

        for k, dat in dats.items():
            if k == 'Washoff':
                continue
            
            window = 10
            if type_trial == 'vcp_70_70':
                window = 40

            t = moving_average(dat['Time (s)'].values, window)
            if type_trial == 'paced':
                t_c = moving_average(dat['Time (s)'].values, 2)
                c = moving_average(dat['Current (pA/pF)'].values, 2)
            else:
                c = moving_average(dat['Current (pA/pF)'].values, window)
                t_c = t

            v = moving_average(dat['Voltage (V)'].values, window)

            if type_trial != 'spont':
                axs[1].plot(t_c*1000, c, col[it], label=label[it])

            if type_trial == 'vcp_70_70':
                if k == 'Post-drug':
                        continue

            axs[0].plot(t*1000, v*1000, col[it], label=label[it])

            it += 1


        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if type_trial == 'vcp_70_70':
            axs[1].set_ylim(-10, 10)

        if type_trial != 'spont':
            axs[1].set_xlabel('Time (ms)', fontsize=14)
            axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
            axs[0].set_ylabel('Voltage (mV)', fontsize=14)
        else:
            axs[0].set_xlabel('Time (ms)', fontsize=14)
            axs[0].set_ylabel('Voltage (mV)', fontsize=14)


        if type_trial == 'spont':
            axes[0].legend(fontsize=14)

        plt.rcParams['svg.fonttype'] = 'none'



    plt.savefig(f'./fig2-data/figure2.svg', format='svg')

    fig.tight_layout()
    plt.show()


def main():
    plot_fig_2()


if __name__ == '__main__':
    main()
