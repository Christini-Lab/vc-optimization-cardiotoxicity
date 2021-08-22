import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from cell_models import protocols

import figs_heka_reader as heka_reader


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
    file_name = f'exp_data/hcn_results/{f}.dat'
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


def smooth_trace(x, w=200):
    trace = np.convolve(x, np.ones(w), mode='same') / w
    return trace


def plot_all_sweeps(f, ch):
    file_name = f'exp_data/hcn_results/{f}.dat'
    bundle = heka_reader.Bundle(file_name)

    for sweep in range(0, 10):
        trace = moving_average(bundle.data[0, 2, sweep, ch-1])
        plt.plot(trace)
        plt.show()


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


def plot_rep_iv_tail(ax, all_iv_traces):
    iv_traces = all_iv_traces[2]
    start_trace = 136000 
    end_trace = 142000
    trace = iv_traces[-20][start_trace:end_trace]

    trace_times = np.linspace(0, len(trace) / 25000, len(trace))*1000

    scale_line = np.array([[100, -60], [200, -60], [200, -40]])

    for v, trace in iv_traces.items():
        ax.plot(trace_times, trace[start_trace:end_trace], 'k')

    ax.plot(scale_line[:, 0], scale_line[:, 1], 'k')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_all_max_cond(all_iv_tail):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_all_tail(ax, all_iv_tail)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./figS14-data/max_cond.svg', format='svg')

    plt.show()


def plot_all_rep_traces(all_iv_traces):
    fig, axs = plt.subplots(2, 1, figsize=(6, 5))

    plot_iv_voltage_proto(axs[0])
    plot_rep_iv_traces(axs[1], all_iv_traces)

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./figS14-data/rep_traces.svg', format='svg')

    plt.show()


def plot_all_zoomed_tail(all_iv_traces):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    plot_rep_iv_tail(ax, all_iv_traces)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./figS14-data/zoomed_tail.svg', format='svg')

    plt.show()


def main():
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
            all_iv_tail.append(iv_tail)
            all_iv_traces.append(iv_traces)


    #plot_all_max_cond(all_iv_tail)
    #plot_all_rep_traces(all_iv_traces)
    plot_all_zoomed_tail(all_iv_traces)


if __name__ == '__main__':
    main()


