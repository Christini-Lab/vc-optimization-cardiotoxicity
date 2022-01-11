import pickle
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from figs_cell_objects import ExpDat
import figs_heka_reader as heka_reader



def step_1():
    path_to_data = f"./exp_data/ga_results"

    files = listdir(path_to_data)

    for f in files:
        if 'optim' in f:
            file_name = f
    
    short_protocol = pickle.load(open(f"{path_to_data}/{file_name}", 'rb'))

    print(f'The protocol is {short_protocol.get_voltage_change_endpoints()[-1]} ms')

    short_protocol.plot_voltage_clamp_protocol(is_plotted=False)

    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(f"fig1-data/figure1_step1.svg", format='svg', transparent=True)
    plt.show()


def step_2():
    path = 'exp_data'
    f = '042721_4_quinine'
    drug = 'quinine'
    cell = ExpDat(path, f, drug)

    col = ['k', 'r']
    label = ['No Drug', 'Drug']

    fig, axes = plt.subplots(5, 1,
            gridspec_kw={'height_ratios': [8, 4, 4, 4, 4]}, figsize=(4.5, 9))

    for i, type_trial in enumerate(['spont', 'paced', 'vcp_70_70']):
        if type_trial == 'spont':
            axs = [axes[0]]
        else:
            if type_trial == 'paced':
                axs = axes[1:3]
            else:
                axs = axes[3:5]

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
            
            window = 10
            if type_trial == 'vcp_70_70':
                window = 40

            t = moving_average(dat['Time (s)'].values, window)
            if type_trial == 'paced':
                t_c = moving_average(dat['Time (s)'].values, 2)
                c = moving_average(dat['Current (pA/pF)'].values, 2)
            else:
                c = moving_average(dat['Current (pA/pF)'].values, window)
                t_c = t

            v = moving_average(dat['Voltage (V)'].values, window)

            if type_trial != 'spont':
                axs[1].plot(t_c*1000, c, col[it], label=label[it])

            if type_trial == 'vcp_70_70':
                if k == 'Post-drug':
                        continue

            axs[0].plot(t*1000, v*1000, col[it], label=label[it])

            it += 1


        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if type_trial == 'vcp_70_70':
            axs[1].set_ylim(-10, 10)

        if type_trial != 'spont':
            axs[1].set_xlabel('Time (ms)', fontsize=14)
            axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
            axs[0].set_ylabel('Voltage (mV)', fontsize=14)
        else:
            axs[0].set_xlabel('Time (ms)', fontsize=14)
            axs[0].set_ylabel('Voltage (mV)', fontsize=14)


        if type_trial == 'spont':
            axes[0].legend(fontsize=14)

        plt.rcParams['svg.fonttype'] = 'none'



    plt.savefig(f'./fig1-data/figure1_step2.svg', format='svg')

    fig.tight_layout()
    plt.show()


def step_5():
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
            conc_block_pts, drug_trace_dat, conc_block_dat = get_pharm_dat(
                    f, ch)
            all_conc_pts.append(conc_block_pts)
            all_drug_traces.append(drug_trace_dat)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    plot_dose_resp(axs[0], all_conc_pts)
    plot_rep_pharm_traces(axs[1], all_drug_traces)

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.savefig(f"fig1-data/figure1_step3.svg", format='svg', transparent=True)
    plt.show()


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs] 

    return np.array(new_vals)


def get_pharm_dat(f, ch, is_shown=False):
    file_name = f'./exp_data/hcn_results/{f}.dat'
    bundle = heka_reader.Bundle(file_name)
    pharm_meta = pd.read_csv(f'./exp_data/hcn_results/{f}_1HCNPharm.xls', sep='\t', index_col=False)

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
            trace = smooth_trace(bundle.data[0, 2, sweep, ch-1], 200)
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

    trace_times = np.linspace(0, len(trace) / 25000, len(trace))

    #Fit the sigmoid

    conc_block_pts = [[], []]

    max_dmso = np.mean(avg_curr_dat[0][-2:])
    for conc, vals in avg_curr_dat.items():
        if conc == 0:
            continue
        vals_to_plot = 1 - np.mean(vals[-2:])/max_dmso
        conc_block_pts[0].append(conc)
        conc_block_pts[1].append(vals_to_plot.tolist())

    popt, pcov = fit_sigmoid(conc_block_pts)
    conc_lin = np.linspace(1, 800, 1000)
    block_lin = [sigmoid(np.log10(x), popt[0], popt[1])
            for x in conc_lin]

    conc_block_dat = np.array([conc_lin, block_lin])

    if is_shown:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].scatter(conc_block_pts[0], conc_block_pts[1], c='k')
        axs[0].plot(conc_lin, block_lin, 'k')

        for conc, traces in drug_trace_dat.items():
            if conc == 0:
                continue

            axs[1].plot(trace_times, traces[-1], label=r'{conc} $\mu$M')

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axs[0].set_xscale('log')

        f_size = 12
        axs[0].set_xlabel('Concentration (nM)', fontsize=f_size)
        axs[1].set_xlabel('Time (s)', fontsize=f_size)

        axs[0].set_ylabel('% Block', fontsize=f_size)
        axs[1].set_ylabel('Current (pA/pF)', fontsize=f_size)

        axs[0].set_ylim(0, 1)

        plt.legend()
        plt.show()

    return conc_block_pts, drug_trace_dat, np.array(conc_block_dat)


def plot_dose_resp(ax, all_conc_pts):
    font_size = 12
    # figure 1, 1
    flattened = [[], []]
    for x in all_conc_pts:
        flattened[0] += x[0]
        flattened[1] += x[1]
        ax.scatter(x[0], x[1])

    flattened = np.array(flattened)

    popt, pcov = fit_sigmoid(flattened.tolist())
    conc_lin = np.linspace(1, 800, 1000)
    block_lin = [sigmoid(np.log10(x), popt[0], popt[1])
            for x in conc_lin]

    ax.plot(conc_lin, block_lin, 'k')
    ax.set_xscale('log')
    ax.set_xlabel(r'Concentration ($\mu$M)', fontsize=font_size)
    ax.set_ylabel('% Block', fontsize=font_size)
    ax.set_ylim(0, 1)
    ax.set_xticklabels([0, 0, 1, 10, 100, 1000])

    print(f'Block at 12uM is: {sigmoid(np.log10(12), popt[0], popt[1])}')
    print(f'IC50 is {10**popt[0]}')

    print(f'Hill Coefficient is: {popt[1]}')


def plot_rep_pharm_traces(ax, all_drug_traces):
    cell_traces = all_drug_traces[2]
    start_pt = 25000
    last_pt = -100
    trace = cell_traces[0][0][start_pt:last_pt]
    trace_times = np.linspace(0, len(trace) / 25000, len(trace))*1000

    c = np.linspace(1, .2, 8)

    concentrations = [c for c in cell_traces.keys()]
    concentrations.reverse()

    for i, conc in enumerate(concentrations):
        col = (c[i], 0, 0)
        traces = cell_traces[conc]
        if conc == 0:
            ax.plot(trace_times,
                    traces[-1][start_pt:last_pt], label=f'.5% DMSO',
                    c=col)
            ax.plot(trace_times,
                    traces[0][start_pt:last_pt], label=f'Baseline',
                    c=col)
        else:
            ax.plot(trace_times, traces[-1][start_pt:last_pt],
                    label=r'{c} $\mu$M'.format(c=conc),
                    c=col)

        i += 1

    scale_line = np.array([[2000, -90], [2500, -90], [2500, -80]])
    ax.plot(scale_line[:, 0], scale_line[:, 1], 'k')

    f_size = 12
    #ax.set_xlabel('Time (ms)', fontsize=f_size)
    #ax.set_ylabel('Current (pA/pF)', fontsize=f_size)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc=(.02, -.12))


def fit_sigmoid(conc_block_pts):
    xdata, ydata = [], []

    xdata, ydata = np.log10(conc_block_pts[0]), conc_block_pts[1]

    p0 = [np.log10(40000), 1]

    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
    return popt, pcov


def sigmoid(x, k, n_):
    bottom = 0
    top = 1

    y = bottom + (top - bottom) / (1 + (10**(k - x)) ** n_)
    return y


def smooth_trace(x, w=200):
    trace = np.convolve(x, np.ones(w), mode='same') / w
    return trace


if __name__ == '__main__':
    step_1()
    step_2()
    step_5()
