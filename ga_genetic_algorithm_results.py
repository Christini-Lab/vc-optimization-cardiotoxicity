"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

from abc import ABC
import copy
import enum
import math
import random
from typing import Dict, List, Union
from os import listdir, mkdir

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

import ga_configs
import mod_protocols as protocols
import mod_trace as trace
import mod_kernik as kernik


class ExtremeType(enum.Enum):
    LOW = 1
    HIGH = 2


class GeneticAlgorithmResult(ABC):
    """Contains information about a run of a genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
        baseline_trace: The baseline trace of the genetic algorithm run.
        generations: A 2D list of every individual in the genetic algorithm.
    """

    def __init__(self, generations):
        self.config = None
        self.baseline_trace = None
        self.generations = generations

    def get_individual(self, generation, index):
        """Returns the individual at generation and index specified."""
        if generation < 0 or generation >= len(self.generations):
            raise ValueError('Please enter a valid generation.')

        if index < 0 or index >= len(self.generations[generation]):
            raise ValueError('Please enter a valid index.')

        return self.generations[generation][index]

    def get_random_individual(self, generation):
        """Returns a random individual from the specified generation."""
        if len(self.generations) <= generation < 0:
            raise ValueError('Please enter a valid generation.')
        return self.get_individual(
            generation=generation,
            index=random.randint(0, len(self.generations[generation]) - 1))

    def get_high_fitness_individual(self, generation=None):
        """Given a generation, returns the individual with the least error."""
        ind_gen = 0
        if generation is None:
            for gen_num, gen in enumerate(self.all_individuals):
                if gen_num == 0:
                    best_ind = self._get_individual_at_extreme(gen_num,
                            ExtremeType.HIGH)
                else:
                    temp_best = self._get_individual_at_extreme(gen_num,
                            ExtremeType.HIGH)
                    if temp_best.fitness > best_ind.fitness:
                        best_ind = temp_best
                        ind_gen = gen_num
            print(f'Individual is from generation {gen_num}')
            return best_ind
        else:
            return self._get_individual_at_extreme(generation, ExtremeType.HIGH)

    def get_low_fitness_individual(self, generation=None):
        """Given a generation, returns the individual with the most error."""
        if generation is None:
            for gen_num, gen in enumerate(self.all_individuals):
                if gen_num == 0:
                    best_ind = self._get_individual_at_extreme(gen_num,
                            ExtremeType.LOW)
                else:
                    temp_best = self._get_individual_at_extreme(gen_num,
                            ExtremeType.LOW)
                    if temp_best.fitness < best_ind.fitness:
                        best_ind = temp_best
                        ind_gen = gen_num
            print(f'Individual is from generation {gen_num}')
            return best_ind
        else:
            return self._get_individual_at_extreme(generation, ExtremeType.LOW)

    def _get_individual_at_extreme(self,
                                   generation: int,
                                   extreme_type: ExtremeType) -> 'Individual':
        """Retrieves either the best or worst individual given a generation."""
        top_error_individual = self.get_individual(generation, 0)
        for i in range(len(self.generations[generation])):
            individual = self.get_individual(generation, i)
            if (extreme_type == ExtremeType.LOW and
                    individual.fitness < top_error_individual.fitness):
                top_error_individual = individual
            elif (extreme_type == ExtremeType.HIGH and
                    individual.fitness > top_error_individual.fitness):
                top_error_individual = individual
        return top_error_individual

    def generate_heatmap(self):
        """Generates a heatmap showing error of individuals."""
        data = []
        for j in range(len(self.generations[0])):
            row = []
            for i in range(len(self.generations)):
                row.append(self.generations[i][j].fitness)
            data.append(row)
        data = np.array(data)

        # Display log error in colorbar.
        tick_range = range(
            math.floor(math.log10(data.min().min())),
            1 + math.ceil(math.log10(data.max().max())))
        cbar_ticks = [math.pow(10, i) for i in tick_range]
        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())

        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(
            data,
            cmap='viridis',
            xticklabels=2,
            yticklabels=2,
            norm=log_norm,
            cbar_kws={'ticks': cbar_ticks, 'aspect': 15})

        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        plt.xticks(
            [i for i in range(0, self.config.max_generations, 5)],
            [i for i in range(0, self.config.max_generations, 5)])
        plt.yticks(
            [i for i in range(0, self.config.population_size, 5)],
            [i for i in range(0, self.config.population_size, 5)])

        ax.invert_yaxis()
        ax.collections[0].colorbar.set_label('Error')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig('figures/Parameter Tuning Figure/heatmap.svg')

    def plot_error_scatter(self):
        plt.figure(figsize=(10, 5))
        x_data = []
        y_data = []
        for i in range(self.config.max_generations):
            for j in range(self.config.population_size):
                x_data.append(j)
                y_data.append(
                    self.get_individual(generation=i, index=j).fitness)
        plt.scatter(x_data, y_data, alpha=0.3, color='red')


class GAResultVoltageClampOptimization(GeneticAlgorithmResult):
    """Contains information about a run of a parameter tuning genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
    """

    def __init__(self, config: ga_configs.VoltageOptimizationConfig,
            current, generations) -> None:
        super().__init__(generations)
        self.config = config
        self.current = current

    def generate_heatmap(self):
        """Generates a heatmap showing error of individuals."""
        data = []
        for j in range(len(self.generations[0])):
            row = []
            for i in range(len(self.generations)):
                row.append(self.generations[i][j].fitness)
            data.append(row)
        data = np.array(data)

        plt.figure()
        ax = sns.heatmap(
            data,
            cmap='RdBu',
            xticklabels=2,
            yticklabels=2)

        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        ax.invert_yaxis()
        ax.axhline(linewidth=4, color='black')
        ax.axvline(linewidth=4, color='black')
        ax.collections[0].colorbar.set_label('Fitness')
        plt.savefig('figures/Voltage Clamp Figure/Single VC Optimization/'
                    'heatmap.svg')

    def graph_fitness_over_generation(self, with_scatter=False):
        """Graphs the change in error over generations."""
        mean_fitnesses = []
        best_individual_fitnesses = []

        for i in range(len(self.generations)):
            best_individual_fitnesses.append(
                self.get_high_fitness_individual(i).fitness)
            mean_fitnesses.append(
                np.mean([j.fitness for j in self.generations[i]]))

        plt.figure()
        if with_scatter:
            self.plot_error_scatter()
        mean_fitness_line, = plt.plot(
            range(len(self.generations)),
            mean_fitnesses,
            label='Mean Fitness')
        best_individual_fitness_line, = plt.plot(
            range(len(self.generations)),
            best_individual_fitnesses,
            label='Best Individual Fitness')
        plt.xticks(range(len(self.generations)))
        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        plt.legend(handles=[mean_fitness_line, best_individual_fitness_line])
        plt.savefig('figures/Voltage Clamp Figure/Single VC Optimization/'
                    'fitness_over_generation.svg')


def graph_vc_protocol(protocol: protocols.VoltageClampProtocol,
                      title: str) -> None:
    """Graphs a voltage clamp optimization individual."""
    plt.figure()
    i_trace = paci_2018.generate_trace(protocol=protocol)
    if i_trace:
        i_trace.plot_with_currents()
        plt.savefig('figures/Voltage Clamp Figure/Single VC Optimization/'
                    '{}.svg'.format(title))
    else:
        print('Could not generate individual trace for individual: {}.'.format(
            protocol))


def graph_optimized_vc_protocol_full_figure(
        single_current_protocols: Dict[str, protocols.VoltageClampProtocol],
        combined_protocol: protocols.VoltageClampProtocol,
        config: ga_configs.VoltageOptimizationConfig) -> None:
    """Graphs a full figure for a optimized voltage protocol."""
    plt.figure(figsize=(20, 10))
    #i_trace = paci_2018.generate_trace(protocol=combined_protocol)
    i_trace = self.model.generate_trace(protocol=combined_protocol)
    i_trace.plot_with_currents(title='')
    plt.savefig('figures/Voltage Clamp Figure/Full VC Optimization/Combined '
                'trace.svg')

    # Plot single current traces.
    i = 1
    for key in sorted(single_current_protocols.keys()):
        plt.figure(figsize=(10, 5))
        #i_trace = paci_2018.generate_trace(
        #    protocol=single_current_protocols[key])
        i_trace = self.model.generate_trace(
            protocol=single_current_protocols[key])
        i_trace.plot_with_currents(title=r'$I_{{{}}}$'.format(key[2:]))
        i += 1
        plt.savefig(
            'figures/Voltage Clamp Figure/Full VC Optimization/'
            '{} single current trace.svg'.format(key))

    # Plot current contributions for combined trace.
    graph_combined_current_contributions(
        protocol=combined_protocol,
        config=config,
        title='Full VC Optimization/Combined current contributions'
    )

    # Plot single current max contributions.
    graph_single_current_contributions(
        single_current_protocols=single_current_protocols,
        config=config,
        title='Full VC Optimization/Single current contributions')


def graph_single_current_contributions(
        single_current_protocols: Dict[str, protocols.VoltageClampProtocol],
        config: ga_configs.VoltageOptimizationConfig,
        title: str) -> None:
    """Graphs the max current contributions for single currents together."""
    single_current_max_contributions = {}
    for key, value in single_current_protocols.items():
        #i_trace = paci_2018.generate_trace(protocol=value)
        i_trace = self.model.generate_trace(protocol=value)

        max_contributions = i_trace.current_response_info.\
            get_max_current_contributions(
                time=i_trace.t,
                window=config.window,
                step_size=config.step_size)
        single_current_max_contributions[key] = max_contributions[
            max_contributions['Current'] == key]['Contribution'].values[0]

    graph_current_contributions_helper(
        currents=single_current_max_contributions.keys(),
        contributions=single_current_max_contributions.values(),
        target_currents=config.target_currents,
        title=title)


def graph_combined_current_contributions(
        protocol: protocols.VoltageClampProtocol,
        config: ga_configs.VoltageOptimizationConfig,
        title: str) -> None:
    """Graphs the max current contributions for a single protocol."""
    #i_trace = paci_2018.generate_trace(protocol=protocol)
    i_trace = self.model.generate_trace(protocol=protocol)
    max_contributions = i_trace.current_response_info.\
        get_max_current_contributions(
            time=i_trace.t,
            window=config.window,
            step_size=config.step_size)

    graph_current_contributions_helper(
        currents=list(max_contributions['Current']),
        contributions=list(max_contributions['Contribution']),
        target_currents=config.target_currents,
        title=title)


def graph_current_contributions_helper(currents,
                                       contributions,
                                       target_currents,
                                       title):
    plt.figure()
    sns.set(style="white")

    # Sort currents according to alphabetic order.
    zipped_list = sorted(zip(currents, contributions))
    contributions = [
        contrib for curr, contrib in zipped_list if curr in target_currents
    ]
    currents = [curr for curr, _ in zipped_list if curr in target_currents]

    currents = ['$I_{{{}}}$'.format(i[2:]) for i in currents]

    ax = sns.barplot(
        x=currents,
        y=[i * 100 for i in contributions],
        color='gray',
        linewidth=0.75)
    ax.set_ylabel('Percent Contribution')
    ax.set_yticks([i for i in range(0, 120, 20)])
    ax.set_ybound(lower=0, upper=100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('figures/Voltage Clamp Figure/{}.svg'.format(title))


class Individual:
    """Represents an individual in a genetic algorithm population.

    Attributes:
        fitness: The fitness of the individual. This value can either be
            maximized or minimized.
    """

    def __init__(self, fitness):
        self.fitness = fitness


class ParameterTuningIndividual(Individual):
    """Represents an individual in a parameter tuning genetic algorithm.

    Attributes:
        parameters: An individuals parameters, ordered according to labels
            found in the config object the individual is associated with.
    """

    def __init__(self, parameters: List[float], fitness: float) -> None:
        super().__init__(fitness=fitness)
        self.parameters = parameters

    def __str__(self):
        return ', '.join([str(i) for i in self.parameters])

    def __repr__(self):
        return ', '.join([str(i) for i in self.parameters])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.fitness == other.fitness and
                    self.parameters == other.parameters)
        else:
            return False


class VCOptimizationIndividual(Individual):
    """Represents an individual in voltage clamp optimization genetic algorithm.

    Attributes:
        protocol: The protocol associated with an individual.
    """

    def __init__(self,
                 protocol: protocols.VoltageClampProtocol,
                 fitness: float=0.0,
                 model=kernik) -> None:
        super().__init__(fitness=fitness)
        self.protocol = protocol

    def __str__(self):
        return str(self.fitness)

    def __repr__(self):
        return str(self.fitness)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.protocol == other.protocol and
                    self.fitness == other.fitness)
        else:
            return False

    def __lt__(self, other):
        return self.fitness < other.fitness

    def evaluate(self, config: ga_configs.VoltageOptimizationConfig,
            prestep=5000) -> int:
        """Evaluates the fitness of the individual."""
        #try:
        if config.model_name == 'Paci':
            i_trace = get_model_response(
                    paci_2018.PaciModel(is_exp_artefact=config.with_artefact), self.protocol, prestep=prestep)
            scale = 1000
        else:
            i_trace = get_model_response(
                    kernik.KernikModel(is_exp_artefact=config.with_artefact), self.protocol, prestep=prestep)
            scale = 1

        #except:
        #    print('failed')
        #    return 0
        #i_trace = kernik.KernikModel().generate_response(protocol=self.protocol)


        #if not i_trace:
        #    return 0

        max_contributions = i_trace.current_response_info.\
            get_max_current_contributions(
                time=i_trace.t,
                window=config.window/scale,
                step_size=config.step_size/scale)

        return max_contributions


def get_model_response(model, protocol, prestep):
    """
    Parameters
    ----------
    model : CellModel
        This can be a Kernik, Paci, or OR model instance
    protocol : VoltageClampProtocol
        This can be any VoltageClampProtocol

    Returns
    -------
    trace : Trace
        Trace object with the current and voltage data during the protocol

    Accepts a model object, applies  a -80mV holding prestep, and then 
    applies the protocol. The function returns a trace object with the 
    recording during the input protocol.
    """
    if isinstance(model, kernik.KernikModel):
        if prestep == 5000:
            model.y_ss = [-8.00000000e+01,  3.21216155e-01,  4.91020485e-05,  
                          7.17831342e+00, 1.04739792e+02,  0.00000000e+00,  
                          2.08676499e-04,  9.98304915e-01, 1.00650102e+00,  
                          2.54947318e-04,  5.00272640e-01,  4.88514544e-02, 
                          8.37710905e-01,  8.37682940e-01,  1.72812888e-02,  
                          1.12139759e-01, 9.89533019e-01,  1.79477762e-04,  
                          1.29720330e-04,  9.63309509e-01, 5.37483590e-02,  
                          3.60848821e-05,  6.34831828e-04]
            if model.is_exp_artefact:
                y_ss = model.y_ss
                model.y_ss = model.y_initial
                model.y_ss[0:23] = y_ss
        else:
            prestep_protocol = protocols.VoltageClampProtocol(
                [protocols.VoltageClampStep(voltage=-80.0,
                                            duration=prestep)])
            model.generate_response(prestep_protocol, is_no_ion_selective=False)
            model.y_ss = model.y[:, -1]
    else:
        if prestep == 5000:
            model.y_ss = [-8.25343151e-02,  8.11127086e-02,  1.62883570e-05, 0.00000000e+00, 2.77952737e-05,  9.99999993e-01, 9.99997815e-01,  9.99029678e-01, 3.30417586e-06, 4.72698779e-01,  1.96776956e-02,  9.28349600e-01, 9.27816541e-01,  6.47972131e-02,  6.69227157e-01, 9.06520741e-01, 3.71681543e-03,  9.20330726e+00, 5.31745508e-04,  3.36764418e-01, 2.02812194e-02, 7.93275445e-03,  9.92246026e-01,  0.00000000e+00, 1.00000000e-01,  1.00000000e-01, -2.68570533e-02, -8.00000000e-02]

            if not model.is_exp_artefact:
                model.y_ss = model.y_ss[0:24]

        else:
            prestep_protocol = protocols.VoltageClampProtocol( [protocols.VoltageClampStep(voltage=-80.0, duration=prestep)])

            model.generate_response(prestep_protocol, is_no_ion_selective=False)

            model.y_ss = model.y[:, -1]


    response_trace = model.generate_response(protocol, is_no_ion_selective=False)

    return response_trace
