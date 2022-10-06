from os import listdir, rename, mkdir
from figs_cell_objects import ExpDat
import figs_heka_reader as heka_reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def ipsc_to_csv(f_name):
    types = ['spont']
    
    cell = ExpDat('./exp_data', f_name, f_name.split('_')[-1])

    cell.write_cell_data()

        


def write_ipsc_to_excel(f_name):
    types = ['paced', 'vcp_70_70']
    
    cell = ExpDat('./exp_data', f_name, f_name.split('_')[-1])

    if f_name not in listdir('./ipsc_csv'):
        mkdir(f'./exp_data/new_ipsc_csv/{f_name}')
    else:

        if not listdir(f'./new_ipsc_csv/{f_name}'):
            return


    for i, type_trial in enumerate(types):
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
            
            if type_trial == 'vcp_70_70':
                dat.to_csv(f'./exp_data/new_ipsc_csv/{f_name}/{k.lower()}_vc_proto.csv')
            else:
                dat.to_csv(
                        f'.exp_data/new_ipsc_csv/{f_name}/{k.lower()}_{type_trial}.csv')


def write_hcn_to_excel():
    file_dat = {'210617_003': [1],
                '210617_004': [1],
                '210616_006': [2],
                '210616_003': [1],
                '210616_001': [2, 3]}
    all_conc_pts = []
    all_drug_traces = []

    for f, v in file_dat.items():
        for ch in v:
            file_channel = f'File: {f}, Channel: {ch}'
            print(file_channel)
            #hcn_pharm_df = get_pharm_dat(f, ch)
            hcn_iv_df = get_iv_dat(f, ch)

            #hcn_pharm_df.to_csv(f'./hcn_csv/pharm_{f}_{ch}.csv')
            hcn_iv_df.to_csv(f'./hcn_csv/iv_{f}_{ch}.csv')


def get_pharm_dat(f, ch, is_shown=False):
    file_name = f'./hcn_results/{f}.dat'
    bundle = heka_reader.Bundle(file_name)
    pharm_meta = pd.read_csv(f'./hcn_results/{f}_1HCNPharm.xls', sep='\t', index_col=False)

    capacitances = {'210617_003_1': 6.57E-12,
                    '210617_004_1': 11.11E-12,
                    '210616_006_2': 10.27E-12,
                    '210616_003_1': 3.6E-12,
                    '210616_001_2': 5.61E-12,
                    '210616_001_3': 6.16E-12}

    avg_curr_dat = {}
    drug_trace_dat = {}
    start_mean = 140000
    end_mean = 160000
    capacitance = capacitances[f'{f}_{ch}']

    for sweep in range(0, pharm_meta.shape[0]):
        conc = pharm_meta[f'Concentration({ch})'].iloc[sweep]
        skip_concentrations = [5E-5, 2E-3]#, 1E-6, 8E-4]
        if (conc in skip_concentrations):
            continue

        try:
            trace = bundle.data[0, 2, sweep, ch-1]
        except:
            continue

        trace = trace / capacitance

        avg_curr = trace[start_mean:end_mean].mean()

        conc = conc * 1E6

        if conc not in avg_curr_dat.keys():
            avg_curr_dat[conc] = [avg_curr]
            drug_trace_dat[conc] = [trace]
        else:
            avg_curr_dat[conc].append(avg_curr)
            drug_trace_dat[conc].append(trace)


    trace_times = np.linspace(0, len(trace) / 25000, len(trace))*1000

    all_dat_dict = {}
    all_dat_dict['Time_ms'] = trace_times

    for k, v in drug_trace_dat.items():
        for i, curr_trace in enumerate(v):
            all_dat_dict[f'conc_{k}_sweep_{i+1}_pApF'] = curr_trace



    hcn_dat_df = pd.DataFrame(all_dat_dict)


    return hcn_dat_df


def get_iv_dat(f, ch, is_shown=False):
    file_name = f'hcn_results/{f}.dat'
    
    capacitances = {'210617_003_1': 6.57E-12,
                    '210617_004_1': 11.11E-12,
                    '210616_006_2': 10.27E-12,
                    '210616_003_1': 3.6E-12,
                    '210616_001_2': 5.61E-12,
                    '210616_001_3': 6.16E-12}

    bundle = heka_reader.Bundle(file_name)
    iv_nums = [num for num in range(-20, -130, -10)]
    iv_traces = {}

    capacitance = capacitances[f'{f}_{ch}']


    for i, v in enumerate(iv_nums):
        trace = bundle.data[0, 1, i, ch-1] / capacitance
        if i == 0:
            trace_times = np.linspace(0, len(trace) / 25000, len(trace))
            iv_traces['Time_ms'] = trace_times

        iv_traces[f'{v}_mV'] = trace 

    iv_df = pd.DataFrame(iv_traces)

    return  iv_df


#write_hcn_to_excel()
ipsc_to_csv('021121_1_cisapride')
