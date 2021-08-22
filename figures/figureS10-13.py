from os import listdir, mkdir
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from operator import itemgetter
from itertools import groupby
import pickle
from random import shuffle
import pandas as pd
import seaborn as sns

from utility_funcs import get_cell_objects

import os
up1 = os.path.abspath('..')
os.sys.path.insert(0, up1)
import mod_kernik as kernik
import mod_protocols as protocols


def plot_subtracted_data(drug, p_val=.05):
    files, cell_objects = get_cell_objects()

    if (drug == 'cisapride'):
        add_plt = 1
    elif  (drug == 'verapamil'):
        add_plt = 1
    elif  (drug == 'quinidine'):
        add_plt = 3
    elif  (drug == 'quinine'):
        add_plt = 3
    else:
        print('Invalid drug name')
        return

    fig, axs = plt.subplots(2+add_plt, 1, sharex=True, figsize=(12, 8))

    drug_cols = {'control': 'k', 'cisapride': 'c', 'verapamil': 'r', 'quinidine': 'g', 'quinine': 'b'}

    drug_sub_dat = {'control': [], 'cisapride': [], 'verapamil': [], 'quinidine': [], 'quinine': []}

    for i, cell_object in enumerate(cell_objects):
        if cell_object.drug.lower() not in ['control', drug]:
            continue

        vc_dat = cell_object.get_vc_data(is_filtered=False)
        recorded_data = vc_dat['Pre-drug']

        dat = cell_object.get_subtracted_drug_data('pred_comp')
        dat['Time (s)'] = (dat['Time (s)'] - dat['Time (s)'].min())

        drug_sub_dat[cell_object.drug.lower()
                ].append(dat['Current (pA/pF)'].values)

    mod_k = kernik.KernikModel(is_exp_artefact=True)
    proto = pickle.load(open('exp_data/ga_results/shortened_trial_steps_ramps_200_50_4_-120_60_500_artefact_True_short.pkl', 'rb'))
    trk = mod_k.generate_response(proto, is_no_ion_selective=False)
    start_idx = np.abs(trk.t - 400).argmin()
    t = trk.t[start_idx:]-400

    if drug == 'cisapride':
        axs[2].plot(t, trk.current_response_info.get_current('I_Kr')[start_idx:])
    elif drug == 'verapamil':
        axs[2].plot(t, trk.current_response_info.get_current('I_CaL')[start_idx:])
    elif drug == 'quinidine':
        axs[2].plot(t, trk.current_response_info.get_current('I_Kr')[start_idx:])
        axs[3].plot(t, trk.current_response_info.get_current('I_To')[start_idx:])
        axs[4].plot(t, trk.current_response_info.get_current('I_Ks')[start_idx:])
    elif drug == 'quinine':
        axs[2].plot(t, trk.current_response_info.get_current('I_Kr')[start_idx:])
        axs[3].plot(t, trk.current_response_info.get_current('I_F')[start_idx:])
        axs[4].plot(t, trk.current_response_info.get_current('I_CaL')[start_idx:])
    else:
        return

    axs[0].plot(recorded_data['Time (s)']*1000,
            recorded_data['Voltage (V)']*1000)

    spans = get_subtracted_functional_t(drug_sub_dat, drug_name=drug, p=p_val,
             consec_pts=5)


    cols = ['b', 'r', 'g']
    labs = [r'Cisapride $\mu_{\Delta I_m}$ p<' + f'{p_val}',
                r'Verapamil $\mu_{\Delta I_m}$ p<' + f'{p_val}',
                r'Quinidine $\mu_{\Delta I_m}$ p<' + f'{p_val}',
                r'Quinine $\mu_{\Delta I_m}$ p<' + f'{p_val}'
                ]
    if drug == 'cisapride':
        drug_spans = [spans['cisapride']]
        cols = ['c']
        labs = [labs[0]]
        ax_rng = range(2, 3)
    elif drug == 'verapamil':
        drug_spans = [spans['verapamil']]
        cols = ['r']
        labs = [labs[1]]
        ax_rng = range(2, 3)
    elif drug == 'quinidine':
        drug_spans = [spans['quinidine']]
        cols = ['g']
        labs = [labs[2]]
        ax_rng = range(2, 5)
    elif drug == 'quinine':
        drug_spans = [spans['quinine']]
        cols = ['b']
        labs = [labs[3]]
        ax_rng = range(2, 5)

    for i, spans in enumerate(drug_spans):
        for j, span in enumerate(spans): 
            axs[0].axvspan(
                    recorded_data['Time (s)'].values[span[0]]*1000,
                    recorded_data['Time (s)'].values[span[1]]*1000,
                    color=cols[i], alpha=.3,
                    label=(labs[i] if j == 0 else None))
            for k in ax_rng:
                axs[k].axvspan(
                        recorded_data['Time (s)'].values[span[0]]*1000,
                        recorded_data['Time (s)'].values[span[1]]*1000,
                        color=cols[i], alpha=.3)

    i = -1

    num_cells = {'control': len(drug_sub_dat['control']),
                 'cisapride': len(drug_sub_dat['cisapride']),
                 'verapamil': len(drug_sub_dat['verapamil']),
                 'quinidine': len(drug_sub_dat['quinidine']),
                 'quinine': len(drug_sub_dat['quinine'])}

    if drug == 'a':
        for k, v in drug_sub_dat.items():
            avg_arr = np.array(v).mean(0)
            i += 1

            axs[1].plot(recorded_data['Time (s)']*1000,
                    avg_arr, color=drug_cols[k],
                    label=f'{k} (n={num_cells[k]})')
    else:
        avg_arr = np.array(drug_sub_dat['control']).mean(0)

        axs[1].plot(recorded_data['Time (s)']*1000,
                avg_arr, color=drug_cols['control'],
                label=f'Control (n={num_cells["control"]})')

        avg_arr = np.array(drug_sub_dat[drug]).mean(0)

        axs[1].plot(recorded_data['Time (s)']*1000,
                avg_arr, color=drug_cols[drug],
                label=f'{drug.capitalize()} (n={num_cells[drug]})')


    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axs[0].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_ylabel(r'$\mu_{\Delta I_m}$ (pA/pF)', fontsize=14)
    axs[1].axhline(0, color='grey')
    axs[1].set_ylim(-5, 5)

    if drug == 'cisapride':
        axs[2].set_ylabel(r'$I_{Kr}$ (pA/pF)', fontsize=14)
        axs[2].set_xlabel('Time (ms)', fontsize=14)
    elif drug == 'verapamil':
        axs[2].set_ylabel(r'$I_{CaL}$ (pA/pF)', fontsize=14)
        axs[2].set_xlabel('Time (ms)', fontsize=14)
    elif drug == 'quinidine':
        axs[2].set_ylabel(r'$I_{Kr}$ (pA/pF)', fontsize=14)
        axs[3].set_ylabel(r'$I_{to}$ (pA/pF)', fontsize=14)
        axs[4].set_ylabel(r'$I_{Ks}$ (pA/pF)', fontsize=14)
        axs[4].set_xlabel('Time (ms)', fontsize=14)
    elif drug == 'quinine':
        axs[2].set_ylabel(r'$I_{Kr}$ (pA/pF)', fontsize=14)
        axs[3].set_ylabel(r'$I_{f}$ (pA/pF)', fontsize=14)
        axs[4].set_ylabel(r'$I_{CaL}$ (pA/pF)', fontsize=14)
        axs[4].set_xlabel('Time (ms)', fontsize=14)

    axs[0].legend(loc=1)
    axs[1].legend(loc=1)

    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(f'./figS10-13-data/{drug}_functional_T.svg', format='svg')

    plt.show()


def get_subtracted_functional_t(drug_sub_dat, drug_name, nperm=200, p=.05,
        consec_pts=5):
    idxs = len(drug_sub_dat['control'][0])
    q = 1-p

    sub_dmso = np.array(drug_sub_dat['control'])

    drug_spans = []
    drug_q_t = []

    drugs = [drug_name]

    span_dat = {}

    for drug in drugs:
        sub_drug = np.array(drug_sub_dat[drug])
        if sub_drug.size == 0:
            drug_spans.append([])
            continue

        drug_p = []
        mixed_array = np.concatenate((sub_drug, sub_dmso))
        n_obs = int(mixed_array.shape[0])
        list_of_vals = list(range(0, n_obs))
        null_t_vals = []
        null_t = []

        for i in range(0, nperm):
            shuffle(list_of_vals)
            arr_1 = mixed_array[list_of_vals[0:int(n_obs/2)], :]
            arr_2 = mixed_array[list_of_vals[int(n_obs/2):], :]

            curr_t_vals = np.abs(ttest_ind(arr_1, arr_2).statistic)

            null_t_vals.append(curr_t_vals)
            null_t.append(curr_t_vals.max())

        null_t_vals = np.array(null_t_vals)

        t_vals = np.abs(ttest_ind(sub_dmso, sub_drug).statistic)
        t_obs = max(t_vals)

        pval = np.mean(t_obs < null_t)
        qval = np.quantile(null_t, q)

        pval_pts = [np.mean(null_t_vals[:, i] < t) for i, t in enumerate(t_vals)]
        qval_pts = [np.quantile(null_t_vals[:, i],q) for i, t in enumerate(t_vals)]

        drug_q_t.append([qval_pts, t_vals])

        drug_mask = np.where(t_vals > qval_pts)

        ranges = []
        for k, g in groupby(enumerate(drug_mask[0]), lambda ix : ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            ranges.append((group[0], group[-1]))

        new_ranges = []
        for ra in ranges:
            if (ra[1] - ra[0]) > consec_pts:
                new_ranges.append(ra)

        span_dat[drug] = new_ranges

    return span_dat 


def get_simple_t_span(drug_sub_dat, nperm=200, p=.05, consec_pts=5):
    idxs = len(drug_sub_dat['Control'][0])

    sub_dmso = np.array(drug_sub_dat['control'])

    drug_spans = []

    for drug in ['cisapride', 'verapamil', 'quinidine', 'quinine']:
        sub_drug = np.array(drug_sub_dat[drug])
        if sub_drug.size == 0:
            drug_spans.append([])
            continue

        p_vals = ttest_ind(sub_dmso, sub_drug).pvalue

        drug_mask = np.where(p_vals < p)[0]

        ranges = []
        for k, g in groupby(enumerate(drug_mask), lambda ix : ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            ranges.append((group[0], group[-1]))

        new_ranges = []
        for ra in ranges:
            if (ra[1] - ra[0]) > consec_pts:
                new_ranges.append(ra)

        drug_spans.append(new_ranges)

    return drug_spans[0], drug_spans[1], drug_spans[2], drug_spans[3]


def main():
    plot_subtracted_data(drug='cisapride', p_val=.05) #SX
    plot_subtracted_data(drug='verapamil', p_val=.05) #SXI
    plot_subtracted_data(drug='quinidine', p_val=.05) #SXII
    plot_subtracted_data(drug='quinine', p_val=.05) #SXIII


if __name__ == '__main__':
    main()


