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
from random import shuffle


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


def plot_fig_6a():
    table, cell_objects, files = get_cells()
    which_cell = 65

    cell = cell_objects[which_cell]

    col = ['k', 'r']
    label = ['No Drug', 'Quinine']
    type_trial = 'vcp_70_70'

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    nd_trial = cell.trials['Pre-drug'][type_trial]
    drug_trial = cell.trials['Post-drug'][type_trial]

    dats = cell.get_vc_data()

    i = 0

    for k, dat in dats.items():
        if k == 'Washoff':
            continue
        
        window = 2 

        t = moving_average(dat['Time (s)'].values, window)
        c = moving_average(dat['Current (pA/pF)'].values, window)
        v = moving_average(dat['Voltage (V)'].values, window)

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

    plt.savefig(f'./fig6-data/whole_vcp.svg', format='svg')
    plt.show()
    

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 8))

    i = 0

    for k, dat in dats.items():
        if k == 'Washoff':
            continue
        
        window = 2 

        t = moving_average(dat['Time (s)'].values, window)
        c = moving_average(dat['Current (pA/pF)'].values, window)

        min_idx = np.abs(t - 3.5).argmin()
        max_idx = np.abs(t - 6.5).argmin()

        ax.plot(t[min_idx:max_idx]*1000, c[min_idx:max_idx], col[i], label=label[i])

        i += 1

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Current (pA/pF)', fontsize=14)
    ax.set_ylim(-15, 15)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.legend()

    plt.savefig(f'./fig6-data/zoomed_vcp.svg', format='svg')
    plt.show()


def plot_fig_6b():
    drug_change_data = pd.read_csv('../analyze_experiments/results/cell_change_stats.csv')

    feature_significance = []
    drug_arr = ['cisapride', 'verapamil', 'quinidine', 'quinine']

    for feature in drug_change_data.columns[2:]:
        for i, drug in enumerate(drug_arr):
            control = drug_change_data.loc[
                    drug_change_data['drug_type'] == 'control']
            drug_dat = drug_change_data.loc[drug_change_data['drug_type'] == drug]
            try:
                t_vals = ttest_ind(control[feature].values, drug_dat[feature].values)
            except:
                continue

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

    for feature in ['vc_f_avg_change']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        drug_change_data[feature] = drug_change_data[feature]

        #sns.set_style('white')
        sns.pointplot(x='drug_type', y=feature, data=drug_change_data, join=False, capsize=.2, markers='_')
        sns.swarmplot(x='drug_type', y=feature, data=drug_change_data, color='.15', size=10)

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

        plt.rcParams['svg.fonttype'] = 'none'
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel('Drug', fontsize=18)
        ax.set_ylabel(f'Change in I_F Segment (pA/pF)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f'Change in I_F Segment', fontsize=18)
        plt.savefig(f'./fig6-data/{feature}.svg', format='svg')
        plt.show()


def plot_fig_6c(p_val=.05):
    table, cell_objects, files = get_cells()

    drug_switch = {'c': 'Cisapride',
                   'v': 'Verapamil',
                   'qd': 'Quinidine',
                   'qn': 'Quinine',
                   'a': 'All'}
    all_viable_cells = [v for k, v in VIABLE_CELLS.items()]
    all_viable_cells = [v for sub in all_viable_cells for v in sub]

    which_drug = 'qn'

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    drug_cols = {'Control': 'k', 'Cisapride': 'c',
            'Verapamil': 'r', 'Quinidine': 'g', 'Quinine': 'b'}

    drug_sub_dat = {'Control': [], 'Cisapride': [],
            'Verapamil': [], 'Quinidine': [], 'Quinine': []}

    for i, cell_object in enumerate(cell_objects):
        if i not in all_viable_cells:
            continue

        drug_type = cell_object.drug

        idx_range = [35000, 65000]

        vc_dat = cell_object.get_vc_data()

        recorded_data = vc_dat['Pre-drug'][idx_range[0]:idx_range[1]]


        dat = cell_object.get_subtracted_drug_data('pred_comp')
        dat = (dat.iloc[idx_range[0]:idx_range[1]]
                .copy().reset_index(drop=True))
        dat['Time (s)'] = (dat['Time (s)'] - dat['Time (s)'].min())

        drug_sub_dat[drug_type].append(dat['Current (pA/pF)'].values)


    spans = get_subtracted_functional_t(drug_sub_dat, p=p_val,
             consec_pts=15, drug_name=drug_switch[which_drug])

    lab = r'Quinine $\mu_{\Delta I_m}$ p<' + f'{p_val}'

    drug_spans = [spans['Quinine']]
    col = 'b'

    axs[0].plot(recorded_data['Time (s)']*1000,
            recorded_data['Voltage (V)']*1000)

    mod_k = kernik.KernikModel(is_exp_artefact=True)
    #mod_p = paci_2018.PaciModel(is_exp_artefact=True)
    proto = pickle.load(open('../run_vc_ga/results/trial_steps_ramps_200_50_4_-120_60/shortened_trial_steps_ramps_200_50_4_-120_60_500_artefact_True_short.pkl', 'rb'))
    trk = mod_k.generate_response(proto, is_no_ion_selective=False)
    t_mod = trk.t
    i_mod = trk.current_response_info.get_current('I_F')

    min_idx = np.abs(t_mod - 3900).argmin()
    max_idx = np.abs(t_mod - 6900).argmin()

    #axs[0].plot(t_mod[min_idx:max_idx]-400, trk.command_voltages[min_idx:max_idx])
    axs[2].plot(t_mod[min_idx:max_idx]-400, i_mod[min_idx:max_idx])
    axs[2].axhline(0, color='grey')

    for i, spans in enumerate(drug_spans):
        for j, span in enumerate(spans): 
            axs[0].axvspan(
                    recorded_data['Time (s)'].values[span[0]]*1000,
                    recorded_data['Time (s)'].values[span[1]]*1000,
                    color=col, alpha=.2)

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
    
    plt.rcParams['svg.fonttype'] = 'none'

    axs[0].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_ylabel(r'$\mu_{\Delta I_m}$ (pA/pF)', fontsize=14)
    axs[1].axhline(0, color='grey')
    axs[1].set_ylim(-2.5, 2.5)

    axs[2].set_xlabel('Time (ms)', fontsize=14)
    axs[2].set_ylabel('iPSC-CM Model I_F', fontsize=14)

    axs[0].legend(loc=1)
    axs[1].legend(loc=1)
    plt.savefig(f'./fig6-data/funny_model.svg', format='svg')
    plt.show()


def get_subtracted_functional_t(drug_sub_dat, nperm=200, p=.05,
        consec_pts=5, drug_name='All'):
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
    #plot_fig_6a()
    plot_fig_6b()
    #plot_fig_6c()


if __name__ == '__main__':
    main()
