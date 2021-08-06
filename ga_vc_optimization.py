"""
ADD COMMENT HERE
"""

import copy
import random
from typing import List

from deap import base, creator, tools
import numpy as np
from multiprocessing import Pool
import pickle

import ga_genetic_algorithm_results as genetic_algorithm_results
import mod_protocols as protocols
from mod_kernik import KernikModel


def run_ga(ga_params, toolbox):
    current_results = []
    current = ga_params.config.target_current

    population = toolbox.population(
            ga_params.config.population_size)

    print('\tEvaluating initial population.')

    current_array = [current for _ in range(len(population))]
    eval_input = np.transpose([population, current_array])
    fitnesses = toolbox.map(toolbox.evaluate, eval_input)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = [fit]

    initial_population = []
    for i in range(len(population)):
        initial_population.append(
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=population[i][0].protocol,
                fitness=population[i].fitness.values[0]))

    final_population = [initial_population]
    for generation in range(1, ga_params.config.max_generations):
        print(f'\tGeneration {generation} for {current}')


        selected_offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(i) for i in selected_offspring]

        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.config.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        for i in offspring:
            if random.random() < ga_params.config.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        updated_individuals = [i for i in offspring if not i.fitness.values]

        eval_input = np.transpose([updated_individuals, current_array[0:len(updated_individuals)]])
        fitnesses = toolbox.map(toolbox.evaluate, eval_input)

        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = [fit]

        population = offspring

        intermediate_population = []
        for i in range(len(population)):
            intermediate_population.append(
                genetic_algorithm_results.VCOptimizationIndividual(
                    protocol=population[i][0].protocol,
                    fitness=population[i].fitness.values[0]))

        final_population.append(intermediate_population)
        generate_statistics(population)

    new_current_result = genetic_algorithm_results.\
        GAResultVoltageClampOptimization(VCGA_PARAMS.config,
                                         current=current,
                                         generations=final_population)

    return new_current_result

def _evaluate(eval_input):
    """Evaluates the fitness of an individual.

    Fitness is determined by how well the voltage clamp protocol isolates
    individual ionic currents.
    """
    individual, current = eval_input

    try:
        max_contributions = individual[0].evaluate(
                config=VCGA_PARAMS.config, prestep=5000)
        fitness = max_contributions.loc[max_contributions['Current'] == current][
                'Contribution'].values[0]
    except:
        return 0.0


    return fitness

def _mate(
        i_one: genetic_algorithm_results.VCOptimizationIndividual,
        i_two: genetic_algorithm_results.VCOptimizationIndividual) -> None:
    """Mates two individuals, modifies them in-place."""
    i_one = i_one[0]
    i_two = i_two[0]

    if len(i_one.protocol.steps) != len(i_two.protocol.steps):
        raise ValueError('Individuals do not have the same num of steps.')

    rand_steps = [*range(0, len(i_one.protocol.steps))]
    random.shuffle(rand_steps)

    for i in range(len(i_one.protocol.steps)):
        if random.random() < VCGA_PARAMS.config.gene_swap_probability:
            i_one.protocol.steps[i], i_two.protocol.steps[rand_steps[i]] = (
                i_two.protocol.steps[rand_steps[i]], i_one.protocol.steps[i])

def _mutate(
        individual: genetic_algorithm_results.VCOptimizationIndividual
) -> None:
    """Mutates an individual by choosing a number for norm. distribution."""
    individual = individual[0]
    for i in range(len(individual.protocol.steps)):

        if random.random() < VCGA_PARAMS.config.gene_mutation_probability:
            individual.protocol.steps[i].mutate(VCGA_PARAMS)

def _select(
        population: List[
            genetic_algorithm_results.VCOptimizationIndividual]
) -> List[genetic_algorithm_results.VCOptimizationIndividual]:
    """Selects a list of individuals using tournament selection."""
    new_population = []
    for i in range(len(population)):
        tournament = random.sample(
            population,
            k=VCGA_PARAMS.config.tournament_size)
        best_individual = max(tournament, key=lambda j: j.fitness)
        new_population.append(copy.deepcopy(best_individual))
    return new_population

def _init_individual():
    """Initializes a individual with a randomized protocol."""
    vc = []
    for i in range(VCGA_PARAMS.config.steps_in_protocol):
        which_vc = random.choice(VCGA_PARAMS.config.step_types)
        if which_vc == 'step':
            random_vc = protocols.VoltageClampStep()
        elif which_vc == 'ramp':
            random_vc = protocols.VoltageClampRamp()
        else:
            random_vc = protocols.VoltageClampSinusoid()

        random_vc.set_to_random_step(
            voltage_bounds=VCGA_PARAMS.config.step_voltage_bounds,
            duration_bounds=VCGA_PARAMS.config.step_duration_bounds)

        vc.append(random_vc)

    return genetic_algorithm_results.VCOptimizationIndividual(
        protocol=protocols.VoltageClampProtocol(steps=vc),
        fitness=0)

class VCGAParams():
    def __init__(self, model_name, vco_config):
        """
        Initialize the class
        """
        if model_name == "Kernik":
            self.cell_model = KernikModel
        else:
            self.cell_model = paci_2018.PaciModel

        self.config = vco_config
        self.previous_population = None

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

def start_ga(vco_config):
    global VCGA_PARAMS

    VCGA_PARAMS = VCGAParams(vco_config.model_name, vco_config)

    toolbox = base.Toolbox()
    toolbox.register('init_param', _init_individual)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)
    toolbox.register('evaluate', _evaluate)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=VCGA_PARAMS.config.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    p = Pool()
    toolbox.register("map", p.map)
    #toolbox.register("map", map)

    final_population = run_ga(VCGA_PARAMS, toolbox)

    return final_population


def generate_statistics(population):
    fitness_values = [i.fitness.values for i in population]

    print('\t\tMin fitness: {}'.format(min(fitness_values)))
    print('\t\tMax fitness: {}'.format(max(fitness_values)))
    print('\t\tAverage fitness: {}'.format(np.mean(fitness_values)))
    print('\t\tStandard deviation: {}'.format(np.std(fitness_values)))

