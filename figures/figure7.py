import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from cell_models import protocols

import figs_heka_reader as heka_reader

#file 17_003 has channels 1, 2, and 4. Channel 1 is the only one worth using
#0, 0, 0-ch is some sort of test
#1, 0-10, 0-ch is IV data
#2, 0-, 0-ch is pharma data


#Good cells:
#--> 17_003, ch1


#6-1, 4-2, 3-4, 
file_dat = {'210617_003': [1],
            '210617_004': [1],
            '210616_006': [2],
            '210616_003': [1],
            '210616_001': [2, 3]}

capacitances = {'210617_003_1': 6.57E-12,
                '210617_004_1': 11.11E-12,
                '210616_006_2': 10.27E-12,
                '210616_003_1': 3.6E-12,
                '210616_001_2': 5.61E-12,
                '210616_001_3': 6.16E-12}


def get_iv_dat(f, ch, is_shown=False):
    file_name = f'fig7-data/{f}.dat'
    bundle = heka_reader.Bundle(file_name)
    iv_nums = [num for num in range(-20, -130, -10)]
    iv_traces = {} 
    iv_dat = {} 
    iv_tail = {}
    start_max_curr = 115000
    end_max_curr = 135000

    start_tail_curr = 137700
    end_tail_curr = 137900

    capacitance = capacitances[f'{f}_{ch}']

    for i, v in enumerate(iv_nums):
        trace = bundle.data[0, 1, i, ch-1] / capacitance
        iv_traces[v] = smooth_trace(trace, 200)
        iv_dat[v] = np.mean(trace[start_max_curr:end_max_curr])
        iv_tail[v] = np.mean(trace[start_tail_curr:end_tail_curr])

    max_tail_curr = min([i for i in iv_tail.values()])
    min_tail_curr = max([i for i in iv_tail.values()])
    trace_times = np.linspace(0, len(trace) / 25000, len(trace))

    for v, curr in iv_tail.items():
        iv_tail[v] = (curr - min_tail_curr) / (max_tail_curr - min_tail_curr)

    if is_shown:
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))

        for v, trace in iv_traces.items():
            axs[0].plot(trace_times, trace, 'k')
            axs[0].set_xlabel('Time (ms)', fontsize=12)
            axs[0].set_ylabel('Current (pA/pF)', fontsize=12)

        for v, i_max in iv_dat.items():
            axs[1].scatter(v, i_max, c='k')
            axs[1].set_xlabel('Voltage (mV)', fontsize=12)
            axs[1].set_ylabel('Current (pA/pF)', fontsize=12)

        for v, i_tail in iv_tail.items():
            axs[2].scatter(v, i_tail , c='k')
            axs[2].set_xlabel('Voltage (mV)', fontsize=12)
            axs[2].set_ylabel(r'Normalized Tail$', fontsize=12)

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.show()

    return  iv_traces, iv_dat, iv_tail


def get_pharm_dat(f, ch, is_shown=False):
    file_name = f'fig7-data/{f}.dat'
    bundle = heka_reader.Bundle(file_name)
    pharm_meta = pd.read_csv(f'./fig7-data/{f}_1HCNPharm.xls', sep='\t', index_col=False)

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


def smooth_trace(x, w=200):
    trace = np.convolve(x, np.ones(w), mode='same') / w
    return trace


def plot_all_sweeps(f, ch):
    file_name = f'fig7-data/{f}.dat'
    bundle = heka_reader.Bundle(file_name)

    for sweep in range(0, 10):
        trace = moving_average(bundle.data[0, 2, sweep, ch-1])
        plt.plot(trace)
        plt.show()


def sigmoid(x, k, n_):
    bottom = 0
    top = 1

    y = bottom + (top - bottom) / (1 + (10**(k - x)) ** n_)
    return y


def fit_sigmoid(conc_block_pts):
    xdata, ydata = [], []

    xdata, ydata = np.log10(conc_block_pts[0]), conc_block_pts[1]

    p0 = [np.log10(40000), 1]

    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
    return popt, pcov


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


def plot_pharm_voltage_proto(ax):
    steps = [protocols.VoltageClampStep(-20, 2000),
             protocols.VoltageClampStep(-110, 3500)]
    proto = protocols.VoltageClampProtocol(steps)
    proto.plot_voltage_clamp_protocol(ax=ax, is_plotted=False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_rep_iv_traces(ax, all_iv_traces):
    iv_traces = all_iv_traces[2]
    start_trace = 40000
    trace = iv_traces[-20][start_trace:]

    trace_times = np.linspace(0, len(trace) / 25000, len(trace))*1000

    for v, trace in iv_traces.items():
        ax.plot(trace_times, trace[start_trace:], 'k')

    #ax.set_xlabel('Time (ms)', fontsize=12)
    #ax.set_ylabel('Current (pA/pF)', fontsize=12)
    scale_line = np.array([[700, -100], [1200, -100], [1200, -90]])
    ax.plot(scale_line[:, 0], scale_line[:, 1], 'k')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_all_iv(ax, all_iv_dat):
    mean_currs = []
    std_currs = []

    for v in range(-20, -130, -10):
        mean_currs.append(np.mean([x[v] for x in all_iv_dat]))
        std_currs.append(np.std([x[v] for x in all_iv_dat]))

    voltages = [v for v in range(-20, -130, -10)]

    
    #ax.scatter(voltages, mean_currs, c='k')
    ax.errorbar(voltages, mean_currs, yerr=std_currs, c='k', fmt='o',
            capsize=4)
    ax.plot(voltages, mean_currs, 'k')

    ax.set_xlabel('Voltage (mV)', fontsize=12)
    ax.set_ylabel('Current (pA/pF)', fontsize=12)


def plot_all_tail(ax, all_tail_dat):
    mean_currs = []
    std_currs = []

    for v in range(-20, -130, -10):
        mean_currs.append(np.mean([x[v] for x in all_tail_dat]))
        std_currs.append(np.std([x[v] for x in all_tail_dat]))

    voltages = [v for v in range(-20, -130, -10)]

    
    ax.errorbar(voltages, mean_currs, yerr=std_currs, c='k', fmt='o',
            capsize=4)
    ax.plot(voltages, mean_currs, 'k')

    ax.set_xlabel('Voltage (mV)', fontsize=12)
    ax.set_ylabel(r'G/$G_{max}$', fontsize=12)


def plot_iv_voltage_proto(ax):
    for v in np.linspace(-20, -120, 11):
        steps = [protocols.VoltageClampStep(-20, 400),
                 protocols.VoltageClampStep(v, 3500),
                 protocols.VoltageClampStep(-50, 500)] 
        proto = protocols.VoltageClampProtocol(steps)
        proto.plot_voltage_clamp_protocol(ax=ax, is_plotted=False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)



all_conc_pts = []
all_drug_traces = []
all_iv_traces = []
all_iv_dat = []
all_iv_tail = []

plt.rcParams['svg.fonttype'] = 'none'

for f, v in file_dat.items():
    for ch in v:
        file_channel = f'File: {f}, Channel: {ch}'
        print(file_channel)
        iv_traces, iv_dat, iv_tail = get_iv_dat(f, ch)
        conc_block_pts, drug_trace_dat, conc_block_dat = get_pharm_dat(
                f, ch)
        all_conc_pts.append(conc_block_pts)
        all_drug_traces.append(drug_trace_dat)

        all_iv_traces.append(iv_traces)
        all_iv_dat.append(iv_dat)
        all_iv_tail.append(iv_tail)

#fig, axs = plt.subplots(3, 2, figsize=(12, 8),
#        gridspec_kw={'height_ratios': [1, 3, 4]})

fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(12)


ax1 = plt.subplot2grid(shape=(9, 2), loc=(0, 1))
plot_iv_voltage_proto(ax1)
ax2 = plt.subplot2grid(shape=(9, 2), loc=(1, 1), rowspan=3)
plot_rep_iv_traces(ax2, all_iv_traces)
ax3 = plt.subplot2grid(shape=(9, 2), loc=(0, 0), rowspan=4)
plot_all_iv(ax3, all_iv_dat)


ax4 = plt.subplot2grid(shape=(9, 2), loc=(5, 1))
plot_pharm_voltage_proto(ax4)
ax5 = plt.subplot2grid(shape=(9, 2), loc=(6, 1), rowspan=3)
plot_rep_pharm_traces(ax5, all_drug_traces)
ax6 = plt.subplot2grid(shape=(9, 2), loc=(5, 0), rowspan=4)
plot_dose_resp(ax6, all_conc_pts)

#IV dat: File: 210616_006, Channel: 2

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.savefig('./fig7-data/figure7.svg', format='svg')
plt.show()



fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plot_all_tail(ax, all_iv_tail)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()
