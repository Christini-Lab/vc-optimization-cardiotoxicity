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

files = listdir(path_to_data)

for f in files:
    if ('shorten' in f) and ('pkl' in f):
        file_name = f

short_protocol = pickle.load(open(f"{path_to_data}/{file_name}", 'rb'))

fig, ax = short_protocol.plot_voltage_clamp_protocol(is_plotted=False)


k = kernik.KernikModel(is_exp_artefact=True)
p = paci_2018.PaciModel(is_exp_artefact=True)

tr_k = k.generate_response(short_protocol, is_no_ion_selective=False)
tr_p = p.generate_response(short_protocol, is_no_ion_selective=False)


max_k_currs = tr_k.current_response_info.get_max_current_contributions(
        time=tr_k.t, window=10, step_size=5)
max_p_currs = tr_p.current_response_info.get_max_current_contributions(
        time=tr_p.t, window=.010, step_size=.005)

combined_currs = []
kernik_currs = []
paci_currs = []

for curr in ['I_Na', 'I_CaL', 'I_Kr', 'I_K1', 'I_Ks', 'I_F', 'I_To']:
    k_start = max_k_currs[max_k_currs['Current'] == curr]['Time Start']
    p_start = max_p_currs[max_p_currs['Current'] == curr]['Time Start'] * 1000
    
    if np.abs(k_start.values[0] - p_start.values[0]) < 10:
        combined_currs.append(k_start.values[0])
    else:
        kernik_currs.append(k_start.values[0])
        paci_currs.append(p_start.values[0])

cols = ['grey', 'g', 'r']
labels = ['Both', 'Kernik', 'Paci']

for i, max_array in enumerate([combined_currs, kernik_currs, paci_currs]):
    for x, t in enumerate(max_array):
        start_t = t-10
        end_t = t + 10

        if x == 0:
            ax.axvspan(start_t, end_t, color=cols[i], alpha=.3, label=labels[i])
        else:
            ax.axvspan(start_t, end_t, color=cols[i], alpha=.3)

plt.legend()
plt.rcParams['svg.fonttype'] = 'none'

plt.savefig(f'./figS9-data/figureS9.svg', format='svg')
plt.show()
