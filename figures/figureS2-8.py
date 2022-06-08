import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

up1 = os.path.abspath('..')
os.sys.path.insert(0, up1)
import mod_kernik as kernik
import mod_paci_2018 as paci
import mod_protocols as protocols

plt.rcParams['svg.fonttype'] = 'none' 



folder = './exp_data/ga_results'
currents = ['I_Na', 'I_To', 'I_Kr', 'I_K1', 'I_CaL', 'I_Ks', 'I_F']

def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual

for i, current in enumerate(currents):
    print(current)
    ga_result = pickle.load(open(f'{folder}/ga_results_{current}.pkl', 'rb'))
    best_individual = get_high_fitness(ga_result)
    proto = best_individual.protocol
    
    k = kernik.KernikModel(is_exp_artefact=True)
    tr_k = k.generate_response(proto, is_no_ion_selective=False)

    max_k_currs = tr_k.current_response_info.get_max_current_contributions(
        time=tr_k.t, window=1, step_size=1)
    print(max_k_currs)


    p = paci.PaciModel(is_exp_artefact=True)
    tr_p = p.generate_response(proto, is_no_ion_selective=False) 

    fig, axs = plt.subplots(3, 2, figsize=(8, 6))

    labs = ['Kernik', 'Paci']
    styles = ['k-', 'g--']
    scales = [1, 1000]

    titles = [r'$I_{Na}$', r'$I_{to}$', r'$I_{Kr}$', r'$I_{K1}$', r'$I_{CaL}$', r'$I_{Ks}$', r'$I_{f}$']

    for it, tr in enumerate([tr_k, tr_p]):
        sc = scales[it]
        conts = tr.current_response_info.get_current_contributions(tr.t, 1/sc, 1/sc)

        t = sc*tr.t
        curr = tr.current_response_info.get_current_summed()
        axs[1, 0].plot(t, curr, styles[it])
        axs[2, 0].plot(sc*conts['Time Mid'], conts[current], 
                                                styles[it], label=labs[it])

        if it == 0:
            cont_st = conts[current].idxmax() - 150
            cont_end = cont_st + 300

            max_time = conts['Time Mid'][conts[current].idxmax()]
            st_time = max_time - 150
            end_time = max_time + 150
            conts_times = sc*conts['Time Mid'][cont_st:cont_end]


        axs[2, 1].plot(conts_times,
                conts[current][cont_st:cont_end], styles[it], label=labs[it])
        

        t = sc*tr.t
        start_idx = np.argmin(np.abs(t - st_time))
        end_idx = np.argmin(np.abs(t - end_time))
        t = t[start_idx:end_idx]

        curr = tr.current_response_info.get_current_summed()[start_idx:end_idx]

        axs[1, 1].plot(t, curr, styles[it])

    fs = 14

    axs[0, 0].plot(tr_p.t*1000, 1000*np.array(tr_p.command_voltages), 'k')
    axs[0, 1].plot(t, 1000*np.array(tr_p.command_voltages[start_idx:end_idx]), 'k')
    axs[0, 0].set_ylabel(r'$V_{c}$ (mV)', fontsize=fs)
    axs[1, 0].set_ylabel(r'$I_{m}$ (A/F)', fontsize=fs)
    itot_name = r'$I_{tot}$'
    axs[2, 0].set_ylabel(f'{titles[i]} contrib (%)', fontsize=fs)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 1].set_ylim(0, 1)
    axs[0, 0].axvspan(st_time, end_time, color='b', alpha=.3)

    axs[2, 0].set_xlabel('Time (ms)', fontsize=fs)
    axs[2, 1].set_xlabel('Time (ms)', fontsize=fs)
    axs[2, 0].legend()
    #fig.suptitle(titles[i], fontsize=22)

    for row in axs:
        for ax in row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    #plt.savefig(f'./figS2-8-data/{current}.svg', format='svg', transparent=True)
    #plt.show()
    plt.close()
