from os import listdir
from figs_cell_objects import ExpDat
import numpy as np
import pandas as pd


def get_cell_objects():
    path = './exp_data'
    files = [f for f in listdir(f'{path}/cells') if 'h5' in f]
    files = [f.split('.')[0] for f in files]
    files.sort()
    cell_objects = []

    for f in files:
        f = f.split('.')[0]
        drug = f.split('_')[-1].capitalize()

        cell_objects.append(ExpDat(path, f, drug))

    return files, cell_objects


def save_cell_stats():
    f_dir = 'exp_data'
    files, cell_objects = get_cell_objects()

    pre_drug_stats = {
            'cell_num': [],
            'file': [],
            'drug_type': [],
            'vc_kr1_avg_change': [],
            'vc_kr2_avg_change': [],
            'vc_cal_avg_change': [],
            'vc_na_min_change': [],
            'vc_to_avg_change': [],
            'vc_k1_avg_change': [],
            'vc_f_avg_change': [],
            'vc_ks_avg_change': [],
            'vc_weird_k': [],
            'dvdt_max': [],
            'rmp': [],
            'apa': [],
            'apd10': [],
            'apd20': [],
            'apd30': [],
            'apd60': [],
            'apd90': [],
            'triangulation': []
            }

    all_drug_change_stats = {}

    drug_change_stats = {
        'cell_num': [],
        'file': [],
        'drug_type': [],
        'vc_kr1_avg_change': [],
        'vc_kr2_avg_change': [],
        'vc_cal_avg_change': [],
        'vc_na_min_change': [],
        'vc_to_avg_change': [],
        'vc_k1_avg_change': [],
        'vc_f_avg_change': [],
        'vc_ks_avg_change': [],
        'vc_weird_k': [],
        'dvdt_max': [],
        'rmp': [],
        'apa': [],
        'apd10': [],
        'apd20': [],
        'apd30': [],
        'apd60': [],
        'apd90': [],
        'triangulation': []
        }

    channel_key = {
        'vc_kr1_avg_change': 'I_Kr_1',
        'vc_kr2_avg_change': 'I_Kr_2',
        'vc_cal_avg_change': 'I_CaL',
        'vc_na_min_change': 'I_Na',
        'vc_to_avg_change': 'I_To',
        'vc_k1_avg_change': 'I_K1',
        'vc_f_avg_change': 'I_F',
        'vc_ks_avg_change': 'I_Ks',
        'vc_weird_k': 'I_K_weird'}


    for idx, cell in enumerate(cell_objects):
        print(idx)
        f = files[idx]

        ap_data = cell.get_paced_stats()

        for k_stats, v_stats in drug_change_stats.items():
            if k_stats in ['cell_num', 'drug_type', 'file',
                           'vc_kr1_avg_change', 'vc_kr2_avg_change',
                           'vc_cal_avg_change',
                           'vc_na_min_change', 'vc_to_avg_change',
                           'vc_k1_avg_change', 'vc_f_avg_change',
                           'vc_ks_avg_change', 'vc_weird_k',
                           ]:
                continue

            pre_val = np.mean(ap_data['Pre-drug'][k_stats])
            post_val = np.mean(ap_data['Post-drug'][k_stats])

            pct_change = (post_val - pre_val) / pre_val
            pre_drug_stats[k_stats].append(pre_val)
            drug_change_stats[k_stats].append(pct_change)

        pre_drug_stats['cell_num'].append(idx)
        drug_change_stats['cell_num'].append(idx)
        pre_drug_stats['file'].append(f)
        drug_change_stats['file'].append(f)
        pre_drug_stats['drug_type'].append(cell.drug)
        drug_change_stats['drug_type'].append(cell.drug)

        for k_n in ['vc_kr1_avg_change', 'vc_kr2_avg_change',
                    'vc_cal_avg_change', 'vc_na_min_change',
                    'vc_to_avg_change', 'vc_k1_avg_change', 'vc_f_avg_change',
                    'vc_ks_avg_change', 'vc_weird_k']:
            i_name = channel_key[k_n]
            curr_change = cell.get_vc_curr_change(i_name)
            pre_drug_stats[k_n].append(curr_change[0])
            drug_change_stats[k_n].append(curr_change[1])
        #continue

    pd.DataFrame(pre_drug_stats).to_csv(
            f'{f_dir}/cell_stats.csv', index=False)
    pd.DataFrame(drug_change_stats).to_csv(
            f'{f_dir}/cell_change_stats.csv', index=False)
