import pickle
from os import listdir, mkdir
import matplotlib.pyplot as plt


def panel_a():
    currents = ['I_Na', 'I_To', 'I_Kr', 'I_K1', 'I_CaL', 'I_Ks', 'I_F']
    folder = './fig2-data'

    if 'contribution_plots_2a' not in listdir('./fig2-data'):
        mkdir('fig2-data/contribution_plots_2a')


    for current in currents:
        ga_result = pickle.load(open(f'exp_data/ga_results/ga_results_{current}_artefact_True', 'rb'))
        best_individual = get_high_fitness(ga_result)
        path = f'{folder}/contribution_plots_2a'
        best_plot_currents(best_individual, path_to_save=path, current_name=current)

        print(f'The max contribution for {current} is: {best_individual}')


def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual


def best_plot_currents(best_ind, path_to_save, current_name):
    vc_protocol = best_ind.protocol

    vc_protocol.plot_voltage_clamp_protocol(is_plotted=False)

    plt.title(current_name)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f"{path_to_save}/{current_name}.svg", format='svg')

    plt.show()


def panel_b():
    path_to_data = f"fig2-data/"

    files = listdir(path_to_data)
    file_name = 'shortened_trial_steps_ramps_200_50_4_-120_60_500_artefact_True_short.pkl'

    for f in files:
        if ('shorten' in f) and ('pkl' in f):
            file_name = f
    
    short_protocol = pickle.load(open(f"exp_data/ga_results/{file_name}", 'rb'))

    print(f'The protocol is {short_protocol.get_voltage_change_endpoints()[-1]} ms')

    short_protocol.plot_voltage_clamp_protocol(is_plotted=False)

    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(f"{path_to_data}/fig2b.svg", format='svg', transparent=True)
    plt.show()


def main():
    panel_a()
    panel_b()


if __name__ == '__main__':
    main()
