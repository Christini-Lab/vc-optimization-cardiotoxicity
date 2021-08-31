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
from random import shuffle, seed

from figs_cell_objects import *
from utility_funcs import get_cell_objects


def pick_cell():
    print(table)

    which_cell = int(input("Which cell would you like to plot? "))

    print(f"You selected: {files[which_cell]}")

    return 


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs] 

    return np.array(new_vals)


def plot_fig_5a():
    files, cell_objects = get_cell_objects()

    which_cell = 37
    cell = cell_objects[which_cell]

    col = ['k', 'r']
    label = ['No Drug', 'Quinine']
    type_trial = 'vcp_70_70'

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    nd_trial = cell.trials['Pre-drug'][type_trial]
    drug_trial = cell.trials['Post-drug'][type_trial]

    dats = cell.get_vc_data()

    i = 0

    for k, dat in dats.items():
        if k == 'Washoff':
            continue
        
        window = 2 

        t = moving_average(dat['Time (s)'].values[0:20000], window)
        c = moving_average(dat['Current (pA/pF)'].values[0:20000], window)
        v = moving_average(dat['Voltage (V)'].values[0:20000], window)

        if i == 0:
            axs[0].plot(t*1000, v*1000, col[i], label=label[i])

        axs[1].plot(t*1000, c, col[i], label=label[i])

        i += 1

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    axs[1].set_ylim(-10, 10)

    axs[1].set_xlabel('Time (ms)', fontsize=14)
    axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
    axs[0].set_ylabel('Voltage (mV)', fontsize=14)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.legend()

    plt.savefig(f'./fig5-data/whole_vcp.svg', format='svg')
    plt.show()
    

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 8))

    i = 0

    for k, dat in dats.items():
        if k == 'Washoff':
            continue
        
        window = 2 

        t = moving_average(dat['Time (s)'].values, window)
        c = moving_average(dat['Current (pA/pF)'].values, window)

        min_idx = np.abs(t - .85).argmin()
        max_idx = np.abs(t - .89).argmin()

        ax.plot(t[min_idx:max_idx]*1000, c[min_idx:max_idx], col[i], label=label[i])

        i += 1

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Current (pA/pF)', fontsize=14)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.legend()

    plt.savefig(f'./fig5-data/zoomed_vcp.svg', format='svg')
    plt.show()


def plot_fig_5b():
    drug_change_data = pd.read_csv('./exp_data/cell_change_stats.csv')

    feature_significance = []
    drug_arr = ['Cisapride', 'Verapamil', 'Quinidine', 'Quinine']

    for feature in drug_change_data.columns[3:]:
        for i, drug in enumerate(drug_arr):
            control = drug_change_data.loc[
                    drug_change_data['drug_type'] == 'Control']
            drug_dat = drug_change_data.loc[drug_change_data['drug_type'] == drug]
            t_vals = ttest_ind(control[feature].values, drug_dat[feature].values)

            if t_vals.pvalue < .05:
                print(f'The p-value when comparing the {feature} of {drug} is {t_vals.pvalue}')
                feature_significance.append([drug, feature, t_vals.pvalue])

    significance_df = pd.DataFrame(feature_significance,
            columns=['drug', 'feature', 'p_value'])

    drug_index = dict(zip(drug_arr, [1, 2, 3, 4]))

    #FEATURES: 'vc_kr_avg_change' 'vc_cal_avg_change' 'vc_na_min_change'
    # 'vc_to_avg_change' 'vc_k1_avg_change' 'vc_f_avg_change' 'vc_ks_avg_change'
    # 'vc_weird_k' 'dvdt_max' 'rmp' 'apa' 'apd10' 'apd20' 'apd30' 'apd60'
    # 'apd90' 'triangulation': []

    for feature in ['vc_kr1_avg_change']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        drug_change_data[feature] = drug_change_data[feature]

        #sns.set_style('white')
        sns.pointplot(x='drug_type', y=feature, data=drug_change_data, join=False, capsize=.2, markers='_')
        sns.swarmplot(x='drug_type', y=feature, data=drug_change_data, size=10, color='.15')

        x_vals = []
        y_vals = []

        for i, row in significance_df.iterrows():
            if row.feature != feature:
                continue

            x_vals.append([0, drug_index[row.drug]])

        for i, x_val in enumerate(x_vals):
            change_range = (drug_change_data[feature].max() -
                                drug_change_data[feature].min())
            y = drug_change_data[feature].max() + (i+.1) * change_range * .08
            h, col =  change_range*.05, 'k'


            x1, x2 = x_val[0], x_val[1]
            all_x = [x1, x1, x2, x2]
            all_y = [y, y+h, y+h, y]
            plt.plot(all_x, all_y, lw=1.5, color=col)
            print(f'y: {y}, h: {h}')
            plt.text((x1+x2)*.5, y+.5*h, "*", ha='center', va='bottom', color=col, fontsize=22)
            print(y+h)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel('Drug', fontsize=18)
        ax.set_ylabel(f'Change in IKr Segment (pA/pF)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f'Change in IKr Segment', fontsize=18)
        plt.rcParams['svg.fonttype'] = 'none'

        plt.savefig(f'./fig5-data/{feature}.svg', format='svg')
        plt.show()


def plot_fig_5cde(p_val=.05):
    files, cell_objects = get_cell_objects()

    w_models = 'n'
    drug_switch = {'c': 'Cisapride',
                   'v': 'Verapamil',
                   'qd': 'Quinidine',
                   'qn': 'Quinine',
                   'a': 'All'}

    for which_drug in ['c', 'qd','qn']:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        plt.rcParams['svg.fonttype'] = 'none'


        drug_cols = {'Control': 'k', 'Cisapride': 'c',
                'Verapamil': 'r', 'Quinidine': 'g', 'Quinine': 'b'}

        drug_sub_dat = {'Control': [], 'Cisapride': [],
                'Verapamil': [], 'Quinidine': [], 'Quinine': []}

        for i, cell_object in enumerate(cell_objects):
            drug_type = cell_object.drug

            idx_range = [8500, 8900]

            vc_dat = cell_object.get_vc_data(is_filtered=False)

            recorded_data = vc_dat['Pre-drug'][idx_range[0]:idx_range[1]]

            dat = cell_object.get_subtracted_drug_data('pred_comp')
            dat = (dat.iloc[idx_range[0]:idx_range[1]]
                    .copy().reset_index(drop=True))
            dat['Time (s)'] = (dat['Time (s)'] - dat['Time (s)'].min())

            drug_sub_dat[drug_type].append(dat['Current (pA/pF)'].values)


        spans = get_subtracted_functional_t(drug_sub_dat, p=p_val,
                 consec_pts=10, drug_name=drug_switch[which_drug])

        cols = ['b', 'r', 'g']
        labs = [r'Cisapride $\mu_{\Delta I_m}$ p<' + f'{p_val}',
                    r'Verapamil $\mu_{\Delta I_m}$ p<' + f'{p_val}',
                    r'Quinidine $\mu_{\Delta I_m}$ p<' + f'{p_val}',
                    r'Quinine $\mu_{\Delta I_m}$ p<' + f'{p_val}'
                    ]
        if which_drug == 'c':
            drug_spans = [spans['Cisapride']]
            cols = ['c']
            labs = [labs[0]]
            ax_rng = range(2, 3)
        elif which_drug == 'qd':
            drug_spans = [spans['Quinidine']]
            cols = ['g']
            labs = [labs[2]]
            ax_rng = range(2, 5)
        elif which_drug == 'qn':
            drug_spans = [spans['Quinine']]
            cols = ['b']
            labs = [labs[3]]
            ax_rng = range(2, 6)


        axs[0].plot(recorded_data['Time (s)']*1000,
                recorded_data['Voltage (V)']*1000)
        for i, spans in enumerate(drug_spans):
            for j, span in enumerate(spans): 
                axs[0].axvspan(
                        recorded_data['Time (s)'].values[span[0]]*1000,
                        recorded_data['Time (s)'].values[span[1]]*1000,
                        color=cols[i], alpha=.3,
                        label=(labs[i] if j == 0 else None))

        i = -1

        num_cells = {'Control': len(drug_sub_dat['Control']),
                     'Cisapride': len(drug_sub_dat['Cisapride']),
                     'Verapamil': len(drug_sub_dat['Verapamil']),
                     'Quinidine': len(drug_sub_dat['Quinidine']),
                     'Quinine': len(drug_sub_dat['Quinine'])}

        avg_arr = np.array(drug_sub_dat['Control']).mean(0)

        axs[1].plot(recorded_data['Time (s)']*1000,
                avg_arr, color=drug_cols['Control'],
                label=f'Control (n={num_cells["Control"]})')

        avg_arr = np.array(drug_sub_dat[drug_switch[which_drug]]).mean(0)

        axs[1].plot(recorded_data['Time (s)']*1000,
                avg_arr, color=drug_cols[drug_switch[which_drug]],
                label=f'{drug_switch[which_drug]} (n={num_cells[drug_switch[which_drug]]})')

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axs[0].set_ylabel('Voltage (mV)', fontsize=14)
        axs[1].set_ylabel(r'$\mu_{\Delta I_m}$ (pA/pF)', fontsize=14)
        axs[1].axhline(0, color='grey')
        axs[1].set_ylim(-2.5, 2.5)

        axs[1].set_xlabel('Time (ms)', fontsize=14)

        axs[0].legend(loc=1)
        axs[1].legend(loc=1)
        plt.savefig(f'./fig5-data/{drug_switch[which_drug]}.svg', format='svg')
        plt.show()


def get_subtracted_functional_t(drug_sub_dat, nperm=200, p=.05,
        consec_pts=10, drug_name='All'):
    seed(1)
    idxs = len(drug_sub_dat['Control'][0])
    q = 1-p

    sub_dmso = np.array(drug_sub_dat['Control'])

    drug_spans = []
    drug_q_t = []

    if drug_name == 'All':
        drugs = ['Cisapride', 'Verapamil', 'Quinidine', 'Quinine']
    else:
        drugs = [drug_name.capitalize()]

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


def main():
    plot_fig_5a()
    plot_fig_5b()
    plot_fig_5cde()


if __name__ == '__main__':
    main()
