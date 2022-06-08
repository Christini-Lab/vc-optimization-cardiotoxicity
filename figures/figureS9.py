from os import listdir
import pickle
import numpy as np
import os

import matplotlib.pyplot as plt

up1 = os.path.abspath('..')
os.sys.path.insert(0, up1)
import mod_kernik as kernik
import mod_paci_2018 as paci_2018
import mod_protocols as protocols


path_to_data = "exp_data/ga_results"

short_protocol = pickle.load(open(f"{path_to_data}/optimized_vc_proto.pkl", 'rb'))


#fig, ax = short_protocol.plot_voltage_clamp_protocol(is_plotted=False)
fig, axs = plt.subplots(8, 1, sharex=True, figsize=(8, 8), 
                            gridspec_kw={'height_ratios':[3, 1, 1, 1, 1, 1, 1, 1]})

k = kernik.KernikModel(is_exp_artefact=True)
p = paci_2018.PaciModel(is_exp_artefact=True)

tr_k = k.generate_response(short_protocol, is_no_ion_selective=False)
tr_p = p.generate_response(short_protocol, is_no_ion_selective=False)

axs[0].plot(tr_k.t, tr_k.command_voltages, 'k')
axs[0].set_ylabel('mV', fontsize=10)
#axs[1].plot(tr_k.t, tr_k.current_response_info.get_current_summed(),
#        'b', label='Kernik')
#axs[1].plot(tr_p.t*1000, tr_p.current_response_info.get_current_summed(), 'g--', label='Paci')
#axs[1].set_ylabel(f'$I_m$ (A/F)', fontsize=10)

max_k_currs = tr_k.current_response_info.get_max_current_contributions(
        time=tr_k.t, window=10, step_size=1)
max_p_currs = tr_p.current_response_info.get_max_current_contributions(
        time=tr_p.t, window=.01, step_size=.001)

print('Max Kernik currents')
print(max_k_currs)

print('Max Paci currents')
print(max_p_currs)
combined_currs = []
kernik_currs = []
paci_currs = []

curr_names = ['I_Kr', 'I_CaL', 'I_Na', 'I_To', 'I_K1', 'I_F', 'I_Ks']
titles = [r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na}$', r'$I_{to}$', r'$I_{K1}$', r'$I_{f}$', r'$I_{Ks}$']

for curr in curr_names:
    k_start = max_k_currs[max_k_currs['Current'] == curr]['Time Start']
    p_start = max_p_currs[max_p_currs['Current'] == curr]['Time Start'] * 1000
    
    if np.abs(k_start.values[0] - p_start.values[0]) < 10:
        combined_currs.append(k_start.values[0])
    else:
        kernik_currs.append(k_start.values[0])
        paci_currs.append(p_start.values[0])

cols = ['grey', 'b', 'g']
labels = ['Both', 'Kernik', 'Paci']

for i, max_array in enumerate([combined_currs, kernik_currs, paci_currs]):
    for x, t in enumerate(max_array):
        start_t = t-10
        end_t = t + 10

        if x == 0:
            axs[0].axvspan(start_t, end_t, color=cols[i], alpha=.3, label=labels[i])
        else:
            axs[0].axvspan(start_t, end_t, color=cols[i], alpha=.3)


k_contributions = tr_k.current_response_info.get_current_contributions(tr_k.t, 1, 5)
p_contributions = tr_p.current_response_info.get_current_contributions(tr_p.t, .0010, .005)

for i, curr in enumerate(curr_names):
    axs[i+1].set_ylabel(f'%{titles[i]}', fontsize=10)
    axs[i+1].set_ylim(0, 1)
    axs[i+1].plot(k_contributions['Time Mid'], k_contributions[curr], 'b', label='Kernik')
    axs[i+1].plot(1000*p_contributions['Time Mid'], p_contributions[curr], 'g--', label='Paci')

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[-1].set_xlabel('Time (ms)')

axs[0].legend()
axs[-1].legend()
plt.rcParams['svg.fonttype'] = 'none'

plt.savefig(f'./figS9-data/figureS9.svg', format='svg')
plt.show()
