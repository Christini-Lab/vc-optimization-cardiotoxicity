import pickle
import matplotlib.pyplot as plt

import mod_kernik as kernik


def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual

def plot_current_conributions():
    trial_conditions = "trial_steps_ramps_Kernik_200_50_4_-120_60"
    currents = ['I_Na', 'I_To', 'I_Kr', 'I_K1', 'I_CaL', 'I_Ks', 'I_F']

    for i, current in enumerate(currents):
        ga_result = pickle.load(open(f'ga_results/{trial_conditions}/ga_results_{current}_artefact_True', 'rb'))
        best_individual = get_high_fitness(ga_result)
        proto = best_individual.protocol
        
        k = kernik.KernikModel(is_exp_artefact=True)
        tr = k.generate_response(proto, is_no_ion_selective=False)

        tr.plot_currents_contribution(current, is_shown=True, title=current,
                saved_to=f'./ga_results/{trial_conditions}/{current}.svg')

def main():
    plot_current_conributions()

if __name__ == '__main__':
    main()
