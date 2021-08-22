from os import listdir, mkdir
import h5py
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from operator import itemgetter
from itertools import groupby
import pickle
import pandas as pd
import seaborn as sns

from utility_funcs import get_cell_objects, save_cell_stats
from figs_cell_objects import ExpDat


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs] 

    return np.array(new_vals)


def figure4_ad():
    files, cell_objects = get_cell_objects()
    for k_dr, which_cell in {'Cisapride': 6, 'Verapamil': 12, 'Quinidine': 29,
            'Quinine': 37}.items():
        cell = cell_objects[which_cell]

        col = ['k', 'r']
        label = ['No Drug', k_dr] 

        type_trial = 'paced'
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 8))

        nd_trial = cell.trials['Pre-drug'][type_trial]
        drug_trial = cell.trials['Post-drug'][type_trial]

        dats = cell.get_single_aps()

        i = 0
        for k, dat in dats.items():
            if k == 'Washoff':
                continue
            
            window = 10
            t = moving_average(dat['Time (s)'].values[1000:4000], window)
            t_c = moving_average(dat['Time (s)'].values[1000:4000], 2)
            c = moving_average(dat['Current (pA/pF)'].values[1000:4000], 2)
            v = moving_average(dat['Voltage (V)'].values[1000:4000], window)

            #axs[1].plot(t_c*1000, c, col[i], label=label[i])
            ax.plot(t*1000, v*1000, col[i], label=label[i])
            
            i+=1

        #for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Voltage (mV)', fontsize=14)
        #axs[1].set_ylabel('Current (pA/pF)', fontsize=14)

        plt.rcParams['svg.fonttype'] = 'none'
        ax.legend()

        plt.savefig(f'./fig4-data/{k_dr}.svg', format='svg')
        plt.show()


def figure4_ef():
    if 'cell_change_stats.csv' not in listdir('./exp_data'):
        save_cell_stats()

    drug_change_data = pd.read_csv('exp_data/cell_change_stats.csv')

    feature_significance = []
    drug_arr = ['Cisapride', 'Verapamil', 'Quinidine', 'Quinine']

    all_drug_data = []

    for feature in drug_change_data.columns[2:]:
        for i, drug in enumerate(drug_arr):
            control = drug_change_data.loc[
                    drug_change_data['drug_type'] == 'Control']
            drug_dat = drug_change_data.loc[drug_change_data['drug_type'] == drug]
            try:
                t_vals = ttest_ind(control[feature].values, drug_dat[feature].values)
            except:
                continue
            all_drug_data.append(drug_dat)

            #if feature == 'apd20':
            #    if drug == 'Verapamil':
            #        import pdb
            #        pdb.set_trace()
            

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
    lw = 1
    plt.rcParams["font.family"] = "geneva"

    for feature in ['apd20', 'apd90']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.rcParams['svg.fonttype'] = 'none'

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        drug_change_data[feature] = drug_change_data[feature]

        drug_change_data[feature] = drug_change_data[feature] * 100

        sns.pointplot(x='drug_type', y=feature, data=drug_change_data, join=False, capsize=.2, markers='_')
        sns.swarmplot(x='drug_type', y=feature, data=drug_change_data, size=10, color='.15')
        x_vals = []
        y_vals = []

        plt.setp(ax.lines,linewidth=lw)  # set lw for all lines of g axes

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
            plt.plot(all_x, all_y, lw=lw, color=col)
            print(f'y: {y}, h: {h}')
            plt.text((x1+x2)*.5, y+.5*h, "*", ha='center', va='bottom', color=col, fontsize=24)
            print(y+h)

        ax.set_xlabel('Drug', fontsize=18)
        if 'apd' in feature:
            ax.set_ylabel(f'% Change in {feature.upper()}', fontsize=18)
        else:
            ax.set_ylabel(f'Change in {feature.upper()}', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f'Change in {feature.upper()}', fontsize=18)
        plt.savefig(f'./fig4-data/{feature}.svg', format='svg')
        plt.show()


def main():
    figure4_ad()
    figure4_ef()


if __name__ == '__main__':
    main()
