from cell_models.rtxi.rtxi_data_exploration import explore_data, get_exp_as_df
from cell_models.ga.target_objective import TargetObjective
from cell_models import protocols
from cell_models import kernik, paci_2018, ohara_rudy
from cell_models.ga import target_objective

from os import listdir, mkdir
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from beautifultable import BeautifulTable
import numpy as np
from scipy.signal import argrelextrema, find_peaks
import xlrd
import re
import pandas as pd


class ExpDat():
    def __init__(self, path, f_name, drug):
        self.path = path
        self.f_name = f_name
        self.h5file = h5py.File(f'{path}/cells/{f_name}.h5', 'r')
        self.xl_file = f'{path}/cell_metadata/{f_name}.csv' 
        self.drug = drug

        self.set_trials()
        self.set_artefact_params()
        self.cm = self.artefact_parameters[:, 1].mean()
        

    def set_trials(self):
        df = pd.read_csv(self.xl_file, header=None)

        all_exp_data = {'Pre-drug':{}, 'Post-drug': {}, 'Washoff': {},
                'Compensation': {}}

        for i, row in df.iterrows():
            if row.values[0] in ['Pre-drug', 'Post-drug',
                    'Washoff', 'Compensation']:
                stage_key = row.values[0]

            if pd.isna(row.values[1]):
                continue

            if row.values[1] == 'Trial':
                continue

            if '[' in row.values[1]:
                continue

            trial = row.iloc[0:4].values
            t_name = trial[0]
            if not pd.isna(trial[2]):
                dat = [int(float(trial[1]))]
                rng_vals = re.split('\[|,|\]', trial[2].replace(' ', ''))
                rng = [float(rng_vals[1]), float(rng_vals[2])]
                dat.append(rng)

                if not pd.isna(trial[2]):
                    dat.append(trial[3])
            else:
                dat = int(float(trial[1]))

            all_exp_data[stage_key][t_name] = dat
        
        self.trials = all_exp_data


    def select_data_to_plot(self, return_functions=False):
        prompt_text = "What type of data would you like to plot (select a number from the list below)?"

        i = 1

        functions = []

        if 'spont' in self.trials['Pre-drug'].keys():
            prompt_text += f"\n  ({i}) Spontaneous AP data "
            functions.append(self.plot_spont_data)
            i += 1
        if 'paced' in self.trials['Pre-drug'].keys():
            prompt_text += f"\n  ({i}) Paced AP data "
            functions.append(self.plot_paced_data)
            i += 1
        if 'vcp_70_70' in self.trials['Pre-drug'].keys():
            prompt_text += f"\n  ({i}) Voltage clamp data "
            functions.append(self.plot_vc_data)
            i += 1
            #prompt_text += f"\n  ({i}) Voltage clamp leak comparison "
            #functions.append(self.compare_vc_with_without_leak)
            #i += 1
        if 'rscomp_80' in self.trials['Compensation'].keys():
            prompt_text += f"\n  ({i}) Compensation data "
            functions.append(self.plot_compensation)
            i += 1
            prompt_text += f"\n  ({i}) I-V curve data "
            functions.append(self.plot_i_v_data)
            i += 1
        if 'proto_B' in self.trials['Pre-drug'].keys():
            prompt_text += f"\n  ({i}) OED data "
            functions.append(self.plot_oed_data)
            i += 1

        prompt_text += "\n--> "

        if return_functions:
            return functions

        data_to_plot = int(input(prompt_text))

        is_kernik = input('Do you want to include Kernik? (y/n) ')
        if is_kernik == 'y':
            is_kernik = True
        else:
            is_kernik = False

        functions[data_to_plot - 1](with_kernik=is_kernik)


    def set_artefact_params(self):
        #workbook = xlrd.open_workbook(self.xl_file)
        #sheet = workbook.sheet_by_name('Sheet1')
        df = pd.read_csv(self.xl_file, header=None)

        all_params = []
        for rownum in range(1, 10):
            if pd.isna(df.iloc[rownum, 6]):
                break

            artefact_params = [float(v) for v in df.iloc[rownum, 6:10]]
            artefact_params[0] = int(artefact_params[0])


            all_params.append(artefact_params)
        
        self.artefact_parameters = np.array(all_params)

    
    def create_target_objective(self, recorded_data):
        peak_changes = find_peaks(
                recorded_data['Voltage (V)'].diff().abs().values,
                height=.00008, width=1)[0]
        v_vals = recorded_data['Voltage (V)'].values
        t_vals = recorded_data['Time (s)'].values
        start_index = 0
        prev_index = 0
        start_voltage = v_vals[0]

        proto = []
        for val in peak_changes:
            prev_index = val

            d = (t_vals[val] - t_vals[start_index]) * 1000 + .4
            start_voltage = v_vals[start_index] * 1000
            end_voltage = v_vals[val-5] * 1000
            
            if start_voltage == v_vals[val-5]*1000:
                #Is step
                p = protocols.VoltageClampStep(voltage=start_voltage, duration=d)
            else:
                p = protocols.VoltageClampRamp(
                        voltage_start=start_voltage,
                        voltage_end=end_voltage,
                        duration=d)
            proto.append(p)

            start_voltage = v_vals[val+5]
            start_index = val + 5

        protocol = protocols.VoltageClampProtocol(proto) 

        return protocol
        #return TargetObjective(recorded_data['Time (s)'].values*1000,
        #                    recorded_data['Voltage (V)'].values*1000,
        #                    recorded_data['Current (pA/pF)'].values,
        #                    "Voltage Clamp",
        #                    "Experimental")


    def get_data_from_h5(self, trial):
        recorded_data = get_exp_as_df(self.h5file, trial, self.cm,
                        is_filtered=True)

        recorded_data['Time (s)'] = recorded_data['Time (s)'] - recorded_data['Time (s)'].min()

        return recorded_data


    def get_subtracted_drug_data(self, trial_type):
        dats = self.get_vc_data()
        data_nd = dats['Pre-drug']
        data_drug = dats['Post-drug']

        subtracted_data = data_nd.copy()
        subtracted_data['Current (pA/pF)'] = data_drug['Current (pA/pF)'] - data_nd['Current (pA/pF)'] 

        return subtracted_data


    def get_kernik_response(self, target, with_artefact=False,
            artefact_params=None, ss_iters=1):
        if with_artefact:
            mod = kernik.KernikModel(is_exp_artefact=True,
                        exp_artefact_params=artefact_params)
        else:
            mod = kernik.KernikModel()

        mod.find_steady_state(max_iters=ss_iters)
        return mod.generate_response(target, is_no_ion_selective=False)


    def get_single_aps(self, is_paced=True, ap_window=None):
        if is_paced:
            trial_type = 'paced'
        else:
            trial_type = 'spont'

        trials = []
        data_trials = ['Pre-drug']
        trials.append(self.trials['Pre-drug'][trial_type])

        aps = {}

        if trial_type in self.trials['Post-drug'].keys():
            trials.append(self.trials['Post-drug'][trial_type])
            data_trials.append('Post-drug')

        if trial_type in self.trials['Washoff'].keys():
            trials.append(self.trials['Washoff']['paced'])
            data_trials.append('Washoff')

        for i, v in enumerate(trials):
            recorded_data = get_exp_as_df(self.h5file, v[0], self.cm,
                    is_filtered=True, t_range=v[1])
            if is_paced:
                ap1 = find_peaks(-recorded_data['Current (pA/pF)'].values,
                        height=0, distance=1000)[0][5]

                new_dat = recorded_data.iloc[ap1-1500:ap1+4000].copy()

                new_dat['Time (s)'] = new_dat['Time (s)'] - new_dat['Time (s)'].min()

            else:
                if ((ap_window is not None) and
                        (recorded_data['Voltage (V)'].max()<0)):
                    mid_idx = int(recorded_data.shape[0] / 2)
                    start_idx = mid_idx - int(ap_window*10 / 2)
                    end_idx = mid_idx + int(ap_window*10 / 2)
                    new_dat = recorded_data.iloc[start_idx:end_idx].copy().reset_index()
                    mid_idx = int(new_dat.shape[0]/2)
                    new_dat['Time (s)'] -= new_dat['Time (s)'][mid_idx]
                else:
                    ap_peaks = find_peaks(recorded_data['Voltage (V)'].values,
                            height=0, distance=2500)[0]
                    if len(ap_peaks) < 4:
                        new_dat = recorded_data.iloc[start_idx:end_idx].copy().reset_index()
                        mid_idx = int(new_dat.shape[0]/2)
                        new_dat['Time (s)'] -= new_dat['Time (s)'][mid_idx]
                    else:
                        mid_idx = ap_peaks[3]

                        recorded_data['Time (s)'] -= recorded_data[
                                'Time (s)'].iloc[mid_idx]
                        start_idx = int(ap_peaks[3] - (ap_peaks[3] - ap_peaks[2]) / 2)
                        end_idx = int(ap_peaks[3] + (ap_peaks[4] - ap_peaks[3]) / 2)

                        new_dat = recorded_data.iloc[start_idx:end_idx].copy()
            
            aps[data_trials[i]] = new_dat

        return aps


    def get_vc_data(self, is_filtered=True):
        trials = []
        data_trials = ['Pre-drug']
        trials.append(self.trials['Pre-drug']['vcp_70_70'])

        if 'vcp_70_70' in self.trials['Post-drug'].keys():
            trials.append(self.trials['Post-drug']['vcp_70_70'])
            data_trials.append('Post-drug')

        if 'vcp_70_70' in self.trials['Washoff'].keys():
            trials.append(self.trials['Pre-drug']['vcp_70_70'])
            data_trials.append('Washoff')

        dats = {} 
        t_rnge = [44000, 132000]

        for i, trial in enumerate(trials):
            dat = get_exp_as_df(self.h5file, trial, self.cm,
                    is_filtered=is_filtered
                    ).iloc[t_rnge[0]:t_rnge[1]].reset_index(drop=True)

            dat['Time (s)'] = dat['Time (s)'] - dat['Time (s)'].min()

            dats[data_trials[i]] = dat

        return dats


    def plot_spont_data(self, with_kernik=False, is_shown=True):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 8))
        i=0
        titles = ['Pre-Drug Spontaneous Activity',
                    'Post-Drug Spontaneous Activity']
        axs[0].set_title(titles[0])
        axs[1].set_title(titles[1])

        if 'spont' in self.trials['Pre-drug'].keys():
            spont_nd_trial = self.trials['Pre-drug']['spont']
        if 'spont' in self.trials['Post-drug'].keys():
            spont_drug_trial = self.trials['Post-drug']['spont']
        else:
            spont_drug_trial = None

        for v in [spont_nd_trial, spont_drug_trial]:
            if v is None:
                continue

            recorded_data = get_exp_as_df(self.h5file, v[0], self.cm,
                    is_filtered=True, t_range=v[1])

            recorded_data['Time (s)'] = recorded_data['Time (s)'] - recorded_data['Time (s)'].min()
            
            axs[i].plot(recorded_data['Time (s)']*1000,
                        recorded_data['Voltage (V)']*1000, 'k')
            plt.title(titles[i])
            i += 1

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axs[0].set_xlabel('Time (ms)', fontsize=14)
        axs[1].set_xlabel('Time (ms)', fontsize=14)
        axs[0].set_ylabel('Voltage (mV)', fontsize=14)

        axs[0].set_xlim(0, 8000)
        axs[1].set_xlim(0, 8000)
        axs[0].set_ylim(-90, 50)

        fig.suptitle("Spontaneous Data", fontsize=22)
        if is_shown:
            plt.show()
        else:
            return fig


    def plot_paced_data(self, with_kernik=False, save_to=None, is_shown=True, with_washoff=True, is_single_ap=False):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        i=0

        styles = []
        trials = []

        trials.append(self.trials['Pre-drug']['paced'])
        styles.append(['k', '-', 'Pre-Drug'])

        if 'paced' in self.trials['Post-drug'].keys():
            trials.append(self.trials['Post-drug']['paced'])
            styles.append(['r', '-', f'{self.drug}'])

        if with_washoff:
            if 'paced' in self.trials['Washoff'].keys():
                trials.append(self.trials['Washoff']['paced'])
                styles.append(['slategrey', '--', 'Wash-off'])

        for i, v in enumerate(trials):
            col, st, lab = styles[i]
            recorded_data = get_exp_as_df(self.h5file, v[0], self.cm,
                    is_filtered=True, t_range=v[1])
            zero_idx = recorded_data[
                        recorded_data['Current (pA/pF)'] < -4].iloc[0]
            recorded_data['Time (s)'] = (recorded_data['Time (s)'] -
                    zero_idx['Time (s)'])

            axs[0].plot(recorded_data['Time (s)']*1000,
                        recorded_data['Voltage (V)']*1000, color=col,
                        linestyle=st)
            axs[1].plot(recorded_data['Time (s)']*1000,
                        recorded_data['Current (pA/pF)'], color=col,
                        linestyle=st, label=f'{lab}, Ishi={v[2]}')
            i += 1

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if is_single_ap:
            axs[1].set_xlim(-50, 400)
        else:
            axs[1].set_xlim(500, 4500)

        axs[1].set_xlabel('Time (ms)', fontsize=14)
        axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
        axs[0].set_ylabel('Voltage (mV)', fontsize=14)

        plt.legend()
        fig.suptitle("Paced Activity", fontsize=22)
        if save_to is not None:
            plt.savefig(f'{save_to}.svg', format='svg')

        if is_shown:
            plt.show()
        else:
            return fig


    def plot_vc_data(self, with_kernik=False, is_shown=True, save_to=None, with_washoff=True):
        if is_shown:
            currents = input('Which currents do you want to consider? Separate them with a comma (I_Kr,I_CaL,I_Na,I_To,I_K1,I_F,I_Ks) ')
            each_curr = currents.split(',')

            if currents == '':
                num_axs = 2
                each_curr = []
            else:
                num_axs = 2+len(each_curr)
        else:
            num_axs = 2
            each_curr = []

            
        fig, axs = plt.subplots(num_axs, 1, sharex=True, figsize=(12, 8))

        styles = []
        trials = []

        styles.append(['k', '-', 'Pre-Drug'])

        if 'vcp_70_70' in self.trials['Post-drug'].keys():
            styles.append(['r', '-', f'{self.drug}'])

        if 'vcp_70_70' in self.trials['Washoff'].keys():
            styles.append(['slategrey', '--', 'Wash-off'])

        data_keys = ['data_nd', 'data_drug', 'data_washoff']
        data_dict = {}
        dats = []

        dats = self.get_vc_data()


        axs[0].plot(dats['Pre-drug']['Time (s)']*1000,
                dats['Pre-drug']['Voltage (V)']*1000)

        for i, dat in enumerate(dats):
            col, st, labs = styles[i]
            axs[1].plot(dat['Time (s)']*1000,
                    dat['Current (pA/pF)'],
                    color = col,
                    linestyle=st,
                    label=labs)

        if with_kernik:
            target = self.create_target_objective(data_dict['data_nd'])
            tr_baseline_k = self.get_kernik_response(target,
                    with_artefact=False)
            tr_baseline_k_artefact = self.get_kernik_response(target,
                    with_artefact=True)

            labs = ['Kernik No Artefact', 'Kernik With Artefact']
            col = ['dodgerblue', 'dodgerblue']
            st = ['-', '--']
            for i, tr in enumerate([tr_baseline_k, tr_baseline_k_artefact]):
                #axs[0].plot(tr.t, tr.y, color=col[i], linestyle=st[i])
                #axs[1].plot(tr.t,
                #            tr.current_response_info.get_current_summed(),
                #            color=col[i],
                #            linestyle=st[i],
                #            label=labs[i])
                for j, curr in enumerate(each_curr):
                    curr_vals = tr.current_response_info.get_current(curr)
                    axs[j+2].plot(tr.t, curr_vals, color=col[i],
                                    linestyle=st[i], label=labs[i])

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        for i, c in enumerate(each_curr):
            axs[2+i].set_ylabel(f'{c} (pA/pF)', fontsize=20)

        axs[0].set_ylabel('Voltage (mV)', fontsize=20)
        axs[1].set_ylabel('Current (pA/pF)', fontsize=20)

        if len(each_curr) > 0:
            axs[num_axs-1].set_xlabel('Time (ms)', fontsize=20)
        else:
            axs[1].set_xlabel('Time (ms)', fontsize=20)
        axs[1].legend()

        fig.suptitle("VC Drug Study Data", fontsize=22)

        plt.legend(loc=1)
        if save_to is not None:
            axs[1].set_ylim(-12, 12)
            plt.savefig(f'{save_to}.svg', format='svg')

        if is_shown:
            plt.show()
        else:
            axs[1].set_ylim(-12, 12)
            return fig


    def plot_oed_data(self, with_subtracted=False, is_shown=True,
            with_kernik=False):
        #TODO: with_kernik
        with_subtracted=False
        if with_subtracted:
            fig, axs = plt.subplots(3, 2, sharex=True,
                    figsize=(12, 8))
        else:
            fig, axs = plt.subplots(2, 2, sharex=True,
                    figsize=(12, 8))

        styles = []
        trials = []

        trials.append(self.trials['Pre-drug']['proto_B'])
        trials.append(self.trials['Pre-drug']['proto_O'])

        data_keys = ['protocol-B', 'protocol-O']
        data_dict = {}
        dats = []

        for i, trial in enumerate(trials):
            dat = get_exp_as_df(self.h5file, trial, self.cm,
                    is_filtered=True)

            dat['Time (s)'] = dat['Time (s)'] - dat['Time (s)'].min()

            data_dict[data_keys[i]] = dat
            dats.append(dat)


        axs[0][0].plot(data_dict['protocol-B']['Time (s)']*1000,
                data_dict['protocol-B']['Voltage (V)']*1000, 'k')
        axs[0][1].plot(data_dict['protocol-O']['Time (s)']*1000,
                data_dict['protocol-O']['Voltage (V)']*1000, 'k')

        for i, dat in enumerate(dats):
            axs[1][i].plot(dat['Time (s)']*1000,
                    dat['Current (pA/pF)'], 'k')

        for axes in axs:
            for ax in axes:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

        axs[0][0].set_ylabel('Voltage (V)', fontsize=20)
        axs[1][0].set_ylabel('Current (pA/pF)', fontsize=20)
        axs[1][0].set_xlabel('Time (ms)', fontsize=20)
        axs[1][1].set_xlabel('Time (ms)', fontsize=20)

        axs[0][0].set_title('Protocol B', fontsize=20)
        axs[0][1].set_title('Protocol O', fontsize=20)
        #plt.suptitle(f"{self.f_name} | Ra={self.r_access} | Cm={self.cm}", fontsize=16)
        fig.suptitle("OED Data", fontsize=22)

        if is_shown:
            plt.legend(loc=1)
            plt.show()
        else:
            axs[1][0].set_ylim(-12, 12)
            axs[1][1].set_ylim(-12, 12)
            return fig


    def compare_vc_with_without_leak(self, with_kernik=False, is_shown=True):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

        styles = []
        trials = []

        trials.append(self.trials['Pre-drug']['vcp_70_70'])
        styles.append(['k', '-', 'Pre-Drug'])
        styles.append(['b', '--', 'Pre-Drug Leak Removed'])

        if 'vcp_70_70' in self.trials['Post-drug'].keys():
            trials.append(self.trials['Post-drug']['vcp_70_70'])
            styles.append(['r', '-', f'{self.drug}'])
            styles.append(['r', '--', f'{self.drug} Leak Removed'])

        t_rnge = [39000, 140000]

        data_keys = ['nd', 'nd_no_leak', 'drug', 'drug_no_leak']
        data_dict = {}
        dats = []

        for i, trial in enumerate(trials):
            dat = get_exp_as_df(self.h5file, trial, self.cm,
                    is_filtered=True
                    ).iloc[t_rnge[0]:t_rnge[1]].reset_index(drop=True)

            dat['Time (s)'] = dat['Time (s)'] - dat['Time (s)'].min()

            data_dict[data_keys[i]] = dat
            dats.append(dat)

            art_params = self.artefact_parameters 
            rm_val = art_params[art_params[:,0]<=trials[0]][-1, 3]

            dat = self.remove_leak(dat, rm_val)

            data_dict[data_keys[i*2+1]] = dat
            dats.append(dat)

        axs[0].plot(data_dict['nd']['Time (s)']*1000,
                data_dict['nd']['Voltage (V)']*1000)

        for i, dat in enumerate(dats):
            col, st, labs = styles[i]
            axs[1].plot(dat['Time (s)']*1000,
                    dat['Current (pA/pF)'],
                    color = col,
                    linestyle=st,
                    label=labs)

        with_kernik=True
        #if with_kernik:
        #    target = self.create_target_objective(data_dict['nd'])
        #    tr_baseline_k = self.get_kernik_response(target,
        #            with_artefact=False)
        #    #tr_baseline_k_artefact = self.get_kernik_response(target,
        #    #        with_artefact=True)


        #    labs = ['Kernik No Artefact', 'Kernik With Artefact']
        #    col = ['dodgerblue', 'dodgerblue']
        #    st = ['-', '--']
        #    #for i, tr in enumerate([tr_baseline_k, tr_baseline_k_artefact]):
        #    for i, tr in enumerate([tr_baseline_k]):
        #        axs[0].plot(tr.t, tr.y, color=col[i], linestyle=st[i])
        #        axs[1].plot(tr.t,
        #                    tr.current_response_info.get_current_summed(),
        #                    color=col[i],
        #                    linestyle=st[i],
        #                    label=labs[i])

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axs[0].set_ylabel('Voltage (V)', fontsize=20)
        axs[1].set_ylabel('Current (pA/pF)', fontsize=20)
        axs[1].set_xlabel('Time (ms)', fontsize=20)

        fig.suptitle("Rm Leak Subtracted", fontsize=22)

        plt.legend(loc=1)
        if is_shown:
            plt.show()
        else:
            axs[1].set_ylim(-12, 12)
            return fig


    def plot_compensation(self, with_kernik, is_shown=True):
        if with_kernik:
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        else:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

        labs = {
                'rspre_0': 'Rspred=0, Rscomp=0',
                'rspre_20': 'Rspred=20, Rscomp=0',
                'rspre_40': 'Rspred=40, Rscomp=0',
                'rspre_60': 'Rspred=60, Rscomp=0',
                'rspre_80': 'Rspred=80, Rscomp=0',
                'rscomp_0': 'Rspred=70, Rscomp=0',
                'rscomp_20': 'Rspred=70, Rscomp=20',
                'rscomp_40': 'Rspred=70, Rscomp=40',
                'rscomp_60': 'Rspred=70, Rscomp=60',
                'rscomp_80': 'Rspred=70, Rscomp=80'
                }
        labs = {
                'rscomp_0': 'Rspred=70, Rscomp=0',
                'rscomp_20': 'Rspred=70, Rscomp=20',
                'rscomp_40': 'Rspred=70, Rscomp=40',
                'rscomp_60': 'Rspred=70, Rscomp=60',
                'rscomp_80': 'Rspred=70, Rscomp=80'
                }

        cols = []

        for k, v in self.trials['Compensation'].items():
            recorded_data = get_exp_as_df(self.h5file, v, self.cm,
                    is_filtered=True)
            num = int(k.split('_')[1])/10
            num = .15+(num/10)

            #if 'pre' in k:
            #    axs[1].plot(recorded_data['Time (s)']*1000,
            #            recorded_data['Current (pA/pF)'],
            #            color=(num, 0, num),
            #            label=labs[k])
            #    cols.append((num, 0, num))
            #else:
            #    axs[1].plot(recorded_data['Time (s)']*1000,
            #            recorded_data['Current (pA/pF)'],
            #            color=(0, num, 0),
            #            label=labs[k])
            #    cols.append((0, num, 0))
            if 'comp' in k:
                axs[1].plot(recorded_data['Time (s)']*1000,
                        recorded_data['Current (pA/pF)'],
                        color=(0, num, 0),
                        label=labs[k])
                cols.append((0, num, 0))


        axs[0].plot(recorded_data['Time (s)']*1000,
                recorded_data['Voltage (V)']*1000,
                label=k)

        if with_kernik:
            if recorded_data['Time (s)'].values[-1] > 0:
                steps = []
                for v_st in range(1, 11):
                    steps.append(protocols.VoltageClampStep(-80, 400))
                    steps.append(protocols.VoltageClampStep(-80+v_st*10, 50))
                for v_st in range(12, 15, 2):
                    steps.append(protocols.VoltageClampStep(-80, 400))
                    steps.append(protocols.VoltageClampStep(-80+v_st*10, 50))

            steps.append(protocols.VoltageClampStep(-80, 400))

            proto = protocols.VoltageClampProtocol(steps)

            comp_predrs = [0, .2, .4, .6, .8, .7, .7, .7, .7, .7]
            comp_rs =     [0,  0,  0,  0,  0,  0, .2, .4, .6, .8]
            comp_predrs = [.7, .7, .7, .7, .7]
            comp_rs =     [0, .2, .4, .6, .8]
            traces = []
            traces_p = []
            for i, lab in enumerate(labs):
                params = {'G_Na': .8}
                art_params = {'c_m': self.cm,
                            #'r_access_star': float( self.r_access.split(" ")[-1])*1E-3,
                            'r_access_star': 10E-3,
                            'comp_rs': comp_rs[i],
                            'comp_predrs': comp_predrs[i]}
                mod = kernik.KernikModel(updated_parameters=params,
                        is_exp_artefact=True,
                            exp_artefact_params=art_params)
                tr = mod.generate_response(proto, is_no_ion_selective=False)
                traces.append(tr)
                #mod = paci_2018.PaciModel(updated_parameters=params,
                #        is_exp_artefact=True,
                #        exp_artefact_params=art_params)
                #tr = mod.generate_response(proto, is_no_ion_selective=False)
                #traces_p.append(tr)
                print(f'Through simulation {i} of length {len(labs)}')

            labs = [lab for lab in labs.values()]
            for i, tr in enumerate(traces):
                #axs[0].plot(tr.t, tr.y, color=cols[i])
                axs[2].plot(tr.t, tr.current_response_info.get_current_summed(),
                            color=cols[i],
                            label=labs[i])

            #for i, tr in enumerate(traces_p):
            #    axs[0].plot(tr.t*1000, tr.y*1000, color=cols[i])
            #    axs[3].plot(tr.t*1000,
            #            tr.current_response_info.get_current_summed(),
            #                color=cols[i],
            #                label=labs[i])

            mod = kernik.KernikModel(updated_parameters=params)
            tr = mod.generate_response(proto, is_no_ion_selective=False)
            axs[0].plot(tr.t, tr.y, 'k--')
            axs[2].plot(tr.t,
                        tr.current_response_info.get_current_summed(),
                        'k--')
            #mod = paci_2018.PaciModel(updated_parameters=params)
            #tr = mod.generate_response(proto, is_no_ion_selective=False)
            #axs[3].plot(tr.t*1000,
            #            tr.current_response_info.get_current_summed(),
            #            'k--')

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axs[0].set_ylabel('Voltage (V)', fontsize=14)
        axs[1].set_ylabel('Current (pA/pF)', fontsize=14)

        if with_kernik:
            axs[2].set_ylabel(r'Kernik $I_{out}$ (pA/pF)', fontsize=14)
            axs[2].set_ylim(-200, 20)

            axs[2].set_xlabel('Time (s)', fontsize=14)
            #axs[3].set_ylabel(r'Paci $I_{out}$ (pA/pF)', fontsize=14)
            #axs[3].set_ylim(-200, 20)
        else:
            axs[1].set_xlabel('Time (s)', fontsize=14)

        axs[1].set_ylim(-200, 20)

        plt.legend(loc=1)
        if is_shown:
            plt.show()
        else:
            return fig


    def plot_i_v_data(self, with_kernik=False, is_shown=True):
        titles = ["Prediction = 20, Comp = 0", "Prediction = 40, Comp = 0", "Prediction = 60, Comp = 0", "Prediction = 80, Comp = 0", "Prediction = 70, Comp = 0", "Prediction = 70, Comp = 20", "Prediction = 70, Comp = 40", "Prediction = 70, Comp = 60", "Prediction = 70, Comp = 80"]

        get_input_str = "What amount of compensation do you want (will always display no comp)? \n"

        for i, title in enumerate(titles):
            get_input_str += f' {i+1}. {title} \n'

        get_input_str += "--> "

        comp_num = int(input(get_input_str))

        keys = {1: 'rspre_20',
                2: 'rspre_40',
                3: 'rspre_60',
                4: 'rspre_80',
                5: 'rscomp_0',
                6: 'rscomp_20',
                7: 'rscomp_40',
                8: 'rscomp_60',
                9: 'rscomp_80'}
        
        trial_to_plot = keys[comp_num]

        main_one = 'rspre_0'

        fig, axs = plt.subplots(2, 2, sharey=True, figsize=(12, 8))

        recorded_data = get_exp_as_df(self.h5file,
                self.trials['Compensation']['rspre_0'],
                self.cm, is_filtered=True)

        if recorded_data['Time (s)'].max() > 5:
            step_times = [400+450*t for t in range(0, 12)]
            step_voltages = [-70+10*t for t in range(0, 10)]
            step_voltages.append(40) 
            step_voltages.append(60) 
            denom = 15
        else:
            step_times = [398, 898, 1398, 1898, 2398]
            step_voltages = [-40, -20, 0, 20, 40]
            denom = 6
        step_range = 15 

        i = 0
        for k in ['rspre_0', trial_to_plot]:
            v = self.trials['Compensation'][k]
            recorded_data = get_exp_as_df(self.h5file, v, self.cm,
                    is_filtered=True)
            recorded_data['Time (s)'] = recorded_data['Time (s)']*1000
            recorded_data['Voltage (V)'] = recorded_data['Voltage (V)']*1000

            for j, t_step in enumerate(step_times):
                num = j / denom
                t_start_index = t_step * 10
                t_end_index = (t_step + step_range) * 10
                data_slice = recorded_data.iloc[t_start_index:t_end_index]
                data_slice['Time (s)'] = (data_slice['Time (s)'] -
                        data_slice['Time (s)'].min())
                axs[0, i].plot(data_slice['Time (s)'],
                        data_slice['Current (pA/pF)'], 
                        color=(num, num, num))
                axs[1, i].scatter(step_voltages[j],
                        data_slice['Current (pA/pF)'].min(), 
                        color=(num, num, num))

            i += 1

        for ax_row in axs:
            for ax in ax_row:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

        axs[1,0].axhline(0, color=(0,0,0))
        axs[1,1].axhline(0, color=(0,0,0))

        axs[0, 0].set_title('I-V Data for Uncompensated Condition', fontsize=16)
        axs[0, 0].set_xlabel('Time (ms)', fontsize=14)
        axs[0, 0].set_ylabel('Current (pA/pF)', fontsize=14)
        axs[0, 1].set_xlabel('Time (ms)', fontsize=14)

        axs[0, 1].set_title(titles[comp_num-1], fontsize=16)
        axs[1, 0].set_xlabel('Voltage (mV)', fontsize=14)
        axs[1, 0].set_ylabel('Max Current (pA/pF)', fontsize=14)
        axs[1, 1].set_xlabel('Voltage (mV)', fontsize=14)

        plt.show()

        if with_kernik:
            proto = []
            step_voltages = []
            for st in range(-7, 9):
                st += .1
                v = st*10
                proto.append(protocols.VoltageClampStep(voltage=-80.0, duration=300.0))
                proto.append(protocols.VoltageClampStep(voltage=v, duration=50.0))
                step_voltages.append(v)

            target = protocols.VoltageClampProtocol(proto)

            recorded_data['Time (s)'] = recorded_data['Time (s)']/1000
            recorded_data['Voltage (V)'] = recorded_data['Voltage (V)']/1000
            #target = self.create_target_objective(recorded_data)
            
            tr_baseline_k = self.get_kernik_response(target)
            alpha = int(titles[comp_num-1].split(" ")[2].split(',')[0])/100 + .1
            tr_baseline_k_artefact = self.get_kernik_response(target,
                    artefact_params={'c_m': self.cm,
                        'r_access': float(self.r_access.split(" ")[-1])*1E-3,
                        'r_access_star': float(
                            self.r_access.split(" ")[-1])*1E-3,
                        'alpha': alpha}, with_artefact=True)
            fig, axs = plt.subplots(2, 2, sharey=True, figsize=(12, 8))

            step_times = [(350*(i+1) -52) for i in range(0, len(step_voltages))]
            for i, tr in enumerate([tr_baseline_k, tr_baseline_k_artefact]):
                for j, t_step in enumerate(step_times):
                    data_slice = tr.get_i_v_in_time_range(step_times[j],
                            step_times[j]+step_range)
                    num = j / 16

                    axs[0, i].plot(data_slice['Time (ms)']-step_times[j],
                            data_slice['Current (pA/pF)'], color=(num, num, num))
                    axs[1, i].scatter(step_voltages[j],
                            data_slice['Current (pA/pF)'].min(), color=(num, num, num))

            for ax_row in axs:
                for ax in ax_row:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

            axs[1,0].axhline(0, color=(0,0,0))
            axs[1,1].axhline(0, color=(0,0,0))

            axs[0, 0].set_title('Kernik No Artefact', fontsize=16)
            axs[0, 0].set_xlabel('Time (ms)', fontsize=14)
            axs[0, 0].set_ylabel('Current (pA/pF)', fontsize=14)
            axs[0, 1].set_xlabel('Time (ms)', fontsize=14)

        axs[0, 1].set_title(f'Kernik Artefact and alpha = {alpha*100}', fontsize=16)
        axs[1, 0].set_xlabel('Voltage (mV)', fontsize=14)
        axs[1, 0].set_ylabel('Max Current (pA/pF)', fontsize=14)
        axs[1, 1].set_xlabel('Voltage (mV)', fontsize=14)

        if is_shown:
            plt.show()
        else:
            return fig


    def remove_leak(self, recorded_data, rm_val):
        new_df = recorded_data.copy()
        new_df['Current (pA/pF)'] = new_df['Current (pA/pF)'] - (new_df['Voltage (V)'] / rm_val / self.cm)*1E6

        return new_df 


    def write_cell_data(self, with_csv=True):
        path = self.path
        f_name = self.f_name
        if f'cell_csv' not in listdir(path):
            mkdir(f'{self.path}/cell_csv')

        if f'{f_name}' not in listdir(f'{path}/cell_csv'):
            mkdir(f'{path}/cell_csv/{f_name}')

        cell_folder = f'{path}/cell_csv/{f_name}'

        #pdf = backend_pdf.PdfPages(f'{path}/cell_csv/{f_name}/{self.f_name}.pdf')

        #plot_functions = self.select_data_to_plot(return_functions=True)

        #for func in plot_functions:
        #    fig = func(is_shown=False)
        #    pdf.savefig(fig)
        #pdf.close()

        if with_csv:
            for k1, v1 in self.trials.items():
                for k2, v2 in v1.items():
                    if isinstance(v2, int):
                        dat = get_exp_as_df(self.h5file, v2, self.cm,
                                is_filtered=True)
                    else:
                        dat = get_exp_as_df(self.h5file, v2[0], self.cm,
                                is_filtered=True, t_range=v2[1])

                    dat['Time (s)'] = dat['Time (s)'] - dat['Time (s)'].min()
                    dat.to_csv(f'{path}/cell_csv/{f_name}/{k1}_{k2}.csv', index=False)


    def get_paced_data(self, with_kernik=False, pace=1):
        apd_no_drug = self.get_apd_data(trial_name='paced_nd')
        apd_drug = self.get_apd_data(trial_name='paced_drug')

        apd_change = {}
        cell_text = [['APD', 'No Drug (ms)', 'Drug (ms)', '% Change']]

        for k, v in apd_no_drug.items():
            apd_change[k] = int((apd_drug[k].mean() -
                apd_no_drug[k].mean()) / apd_no_drug[k].mean()*100)
            cell_text.append([k.upper(), int(apd_no_drug[k].mean()),
                int(apd_drug[k].mean()), apd_change[k]])

        tri_nd = ((apd_no_drug['apd30'].mean() - apd_no_drug['apd20'].mean()) /
                        (apd_no_drug['apd80'].mean() -
                            apd_no_drug['apd70'].mean()))
        tri_drug = ((apd_drug['apd30'].mean() - apd_drug['apd20'].mean()) /
                        (apd_drug['apd80'].mean() - apd_drug['apd70'].mean()))
        print(f'Pre-drug triangulation is {tri_nd}')
        print(f'Post-drug triangulation is {tri_drug}')
        print(f'Change in triangulation is {(tri_drug - tri_nd) / tri_nd}')

        aps_nd = self.get_paced_aps(trial_name='paced_nd')
        aps_drug = self.get_paced_aps(trial_name='paced_drug')
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

        for ap in aps_nd:
            axs[0].plot(ap[0], ap[1], 'k')
            axs[1].plot(ap[0], ap[2], 'k')

        for ap in aps_drug:
            axs[0].plot(ap[0], ap[1], 'r')
            axs[1].plot(ap[0], ap[2], 'r')

        for ax in [axs[0], axs[1]]:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axs[0].set_ylabel('Voltage (V)', fontsize=14)
        axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
        axs[1].set_xlabel('Time (s)', fontsize=14)

        axs[2].table(cell_text[1:], colLabels=cell_text[0], loc='center')
        axs[2].axis('off')

        plt.show()
        

    def get_paced_stats(self, pace=1):
        span_ap_data = {}
        spans = []
        for span in ['Pre-drug', 'Post-drug']:
            if 'paced' not in self.trials[span].keys():
                continue
            trial = self.trials[span]['paced']
            paced_aps = self.get_paced_aps(trial, pace)

            aps_stats = {
                    'dvdt_max': [],
                    'rmp': [],
                    'apa': [],
                    'apd10': [],
                    'apd20': [],
                    'apd30': [],
                    'apd40': [],
                    'apd50': [],
                    'apd60': [],
                    'apd70': [],
                    'apd80': [],
                    'apd90': [],
                    'triangulation': []
                    }

            ap_apds = []
            for ap in paced_aps[2:5]:
                temp_t, temp_v, temp_c = ap
                didt_max_idx_plus_one = np.diff(temp_c).argmax() + 1
                dvdt_max_idx = didt_max_idx_plus_one + np.diff(
                        temp_v[didt_max_idx_plus_one:]).argmax()
                if temp_v[dvdt_max_idx] < -65:
                    dvdt_max_idx = np.diff(temp_v).argmax()
                dvdt_max = (temp_v[dvdt_max_idx+1] - temp_v[dvdt_max_idx]) / (
                        temp_t[dvdt_max_idx+1] - temp_t[dvdt_max_idx])

                temp_t = temp_t - temp_t[didt_max_idx_plus_one + 30]

                rmp = np.min(temp_v)
                apa = np.max(temp_v[(didt_max_idx_plus_one+30):]) - rmp

                apds = []
                for p in range(1, 10):
                    apds.append(get_apd(temp_t, temp_v, temp_c, p*10))

                ap_apds.append(np.array(apds)[:, 0])
                aps_stats['dvdt_max'].append(dvdt_max)
                aps_stats['rmp'].append(rmp)
                aps_stats['apa'].append(apa)

                aps_stats['triangulation'].append(
                    ((apds[2][0].mean() - apds[1][0].mean()) /
                            (apds[7][0].mean() - apds[6][0].mean())))

            ap_apds = np.array(ap_apds)
            for i in range(1, 10):
                aps_stats[f'apd{i*10}'] = ap_apds[:, i-1]

            span_ap_data[span] = aps_stats

        return span_ap_data


    def get_paced_aps(self, trial, pace=1):
        dat = get_exp_as_df(self.h5file, trial[0], self.cm,
                    is_filtered=True, t_range=trial[1])
        t = dat['Time (s)'].values * 1000
        v = dat['Voltage (V)'].values * 1000
        c = dat['Current (pA/pF)'].values

        freq = t[1] - t[0]
        pts_per_ap = int(pace/freq) * 1000
        
        stim_times = argrelextrema(c, np.less,
                order=int(pts_per_ap/1.9))[0]
        stim_idxs = stim_times[1:-1]
        
        aps = []

        for idx in stim_idxs:
            idx = idx - 500
            end_idx = idx + int(pts_per_ap/2)
            temp_t = t[idx:end_idx]
            temp_t -= temp_t.min()
            temp_v = v[idx:end_idx]
            temp_c = c[idx:end_idx]

            aps.append([temp_t, temp_v, temp_c])

        return aps


    def get_opt_vc_data(self, comp_trial='vcp_70_70'):
        trials = []
        trials.append(self.trials['Pre-drug'][comp_trial])

        if comp_trial in self.trials['Post-drug'].keys():
            trials.append(self.trials['Post-drug'][comp_trial])

        if comp_trial in self.trials['Washoff'].keys():
            trials.append(self.trials['Washoff'][comp_trial])

        t_rnge = [40000, 140000]

        data_keys = ['data_nd', 'data_drug', 'data_washoff']
        data_dict = {}
        dats = []

        for i, trial in enumerate(trials):
            dat = get_exp_as_df(self.h5file, trial, self.cm,
                    is_filtered=True
                    ).iloc[t_rnge[0]:t_rnge[1]].reset_index(drop=True)

            dat['Time (s)'] = dat['Time (s)'] - dat['Time (s)'].min()

            data_dict[data_keys[i]] = dat
            dats.append(dat)

        return data_dict


    def get_vc_curr_change(self, curr_name):
        """
            curr_name: I_Na, I_K_weird
        """
        curr_dict = {
                'I_K_weird': [.610, 1.200, 'max'],
                'I_Kr_1': [1.26, 1.265, 'avg'],
                'I_Kr_2': [1.265, 1.270, 'avg'],
                'I_CaL': [1.975, 1.985, 'avg'],
                'I_Na': [2.750, 2.770, 'min'],
                'I_To': [3.640, 3.650, 'avg'],
                'I_K1': [4.295, 4.305, 'avg'],
                'I_F': [5.5, 5.9, 'avg'],
                'I_Ks': [8.8, 9.0, 'avg'],
                }
        min_t = curr_dict[curr_name][0]
        max_t = curr_dict[curr_name][1]
        min_or_max = curr_dict[curr_name][2]

        vc_dat = self.get_opt_vc_data()

        max_curr = []

        for k, v in vc_dat.items():
            if min_or_max == 'min':
                max_curr.append(v[((v['Time (s)'] > min_t) &
                            (v['Time (s)'] < max_t))]['Current (pA/pF)'].min())
            elif min_or_max == 'avg':
                max_curr.append(v[((v['Time (s)'] > min_t) &
                            (v['Time (s)'] < max_t))]['Current (pA/pF)'].mean())
            elif min_or_max == 'auc':
                max_curr.append(v[((v['Time (s)'] > min_t) &
                        (v['Time (s)'] < max_t) &
                        (v['Current (pA/pF)'] < -5))]['Current (pA/pF)'].mean())
            else:
                max_curr.append(v[((v['Time (s)'] > min_t) &
                            (v['Time (s)'] < max_t))]['Current (pA/pF)'].max())


        return [max_curr[0], (max_curr[1] - max_curr[0])]


def get_apd(t, v, c, percent=90):
    # start measuring from 0
    t_0_idx = np.argmin(np.abs(t))

    rmp = np.min(v)
    apa = np.max(v[t_0_idx:]) - rmp

    apd_v = rmp + apa * (100-percent)/100

    apd_idx = t_0_idx + np.argmin(np.abs(v[t_0_idx:] - apd_v))

    apd = t[apd_idx]

    if apd < 3:
        dvdt_max_idx = t_0_idx + np.argmax(v[t_0_idx:])
        apd_idx = dvdt_max_idx + np.argmin(np.abs(v[dvdt_max_idx:] - apd_v))
        apd = t[apd_idx]



    if apd < 5:
        print(f'At {percent}, there is an APD of {apd}')
    
    
    return apd, apd_v
