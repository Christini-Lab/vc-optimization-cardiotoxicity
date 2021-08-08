from cell_models import kernik, protocols, paci_2018
import pickle
import matplotlib.pyplot as plt

from deap import base, creator, tools
import mod_protocols as protocols
import mod_kernik as kernik


def make_shortened_results(trial_conditions, only_end, holding_step=2000, with_artefact=False, model_name='Kernik'):
    currents = ['I_Kr', 'I_CaL', 'I_Na', 'I_To', 'I_K1', 'I_F', 'I_Ks']
    folder = f"ga_results/{trial_conditions}"

    original_protocols = {}
    shortened_protocols = {}

    for current in currents:
        ga_result = pickle.load(open(f'{folder}/ga_results_{current}_artefact_{with_artefact}', 'rb'))
        best_individual = get_high_fitness(ga_result)
        shortened_protocols[current] = shorten_protocol(
            best_individual, window=10, current_name=current,
            only_end=only_end, model_name=model_name)
        original_protocols[current] = best_individual.protocol

    new_long_protocol = get_long_protocol(shortened_protocols, holding_step)

    scale = 1


    shortened_max_currents = get_max_currents(
        new_long_protocol, prestep=2000, window=10, model_name=model_name,
        scale=scale)

    print(f"The shortened_max_currents are {shortened_max_currents}")
    shortened_max_currents.to_csv(
            f"{folder}/shortened_{trial_conditions}_holding_{holding_step}_{with_artefact}.csv")

    pickle.dump(new_long_protocol, open(f"{folder}/shortened_{trial_conditions}_{holding_step}_artefact_{with_artefact}_short.pkl", 'wb'))


def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual


def shorten_protocol(best_individual, current_name, only_end, model_name, window=10):
    vc_protocol = best_individual.protocol
    length_of_protocol = vc_protocol.get_voltage_change_endpoints()[-1]

    scale = 1

    max_currents = get_max_currents(vc_protocol, model_name=model_name, prestep=2000, window=window, scale=scale)

    start_time = float(max_currents[max_currents["Current"] ==
                                    current_name]["Time Start"])
    shortened_protocol = get_protocol_without_end(
        vc_protocol, start_time, window, extra_time=100, scale=scale)

    if not only_end:
        max_contribution = max_currents[max_currents["Current"] ==
                                    current_name]["Contribution"]
        if current_name == "I_Kr":
            accepted_threshold = .99
        else: 
            accepted_threshold = .95
        shortened_protocol = shorten_protocol_start(
            shortened_protocol, start_time, window, max_contribution,
            current_name, acceptable_change=accepted_threshold, scale=scale,
            model_name=model_name)

    print(
        f'Protocol length of {current_name} decreased from {length_of_protocol} to {shortened_protocol.get_voltage_change_endpoints()[-1]}.')
    return shortened_protocol


def get_max_currents(vc_protocol, prestep, window, model_name, scale=1):
    if model_name == 'Paci':
        baseline_paci= paci_2018.PaciModel(is_exp_artefact=True)
        i_trace = get_trace(baseline_paci, vc_protocol, prestep=prestep)
    else:
        baseline_kernik = kernik.KernikModel(is_exp_artefact=True)
        i_trace = get_trace(baseline_kernik, vc_protocol, prestep=prestep)


    max_currents = i_trace.current_response_info.get_max_current_contributions(
            i_trace.t, window=window/scale, step_size=5/scale)

    return max_currents


def get_trace(model, protocol, prestep=2000):
    prestep_protocol = protocols.VoltageClampProtocol(
        [protocols.VoltageClampStep(voltage=-80.0,
                                    duration=prestep)])

    model.generate_response(prestep_protocol, is_no_ion_selective=False)

    model.y_ss = model.y[:, -1]

    response_trace = model.generate_response(protocol, is_no_ion_selective=False)

    return response_trace


def get_protocol_without_end(protocol, start_time, window, extra_time, scale=1):
    window = window/scale
    extra_time = extra_time/scale

    vc_segment_endpoints = protocol.get_voltage_change_endpoints()

    cutoff_time = scale * (start_time + extra_time)

    if cutoff_time >  vc_segment_endpoints[-1]:
        return protocol

    is_found = False
    i = 0

    while not is_found:
        if cutoff_time < vc_segment_endpoints[i]:
            max_segment_idx = i
            is_found = True

        i += 1

    new_duration = (cutoff_time - vc_segment_endpoints[max_segment_idx-1])

    if isinstance(protocol.steps[max_segment_idx], protocols.VoltageClampRamp):
        new_start_voltage = protocol.steps[max_segment_idx].voltage_start
        new_final_voltage = protocol.get_voltage_at_time(cutoff_time)
        new_segment = protocols.VoltageClampRamp(
            new_start_voltage, new_final_voltage, new_duration)
    else:
        new_start_voltage = protocol.steps[max_segment_idx].voltage
        new_segment = protocols.VoltageClampStep(
            new_start_voltage, new_duration)

     
    new_protocol = protocols.VoltageClampProtocol(
            protocol.steps[0:max_segment_idx] + [new_segment])

    return new_protocol


def shorten_protocol_start(protocol, start_time, window, max_contribution,
        current_name, acceptable_change, model_name, removal_time_step=200,
        scale=1):
    min_acceptable_current = max_contribution * acceptable_change

    while (max_contribution > min_acceptable_current).values[0]:
        if protocol.get_voltage_change_endpoints()[-1] <= removal_time_step:
            return protocol

        print(max_contribution)
        last_protocol = protocol
        
        protocol = remove_start_of_protocol(
            protocol, removal_time_step=removal_time_step)

        max_currents = get_max_currents(
            protocol, prestep=2000, window=10, scale=scale, model_name=model_name)
        max_contribution = max_currents[max_currents["Current"] ==
                                    current_name]["Contribution"]

    return last_protocol


def remove_start_of_protocol(protocol, removal_time_step):
    vc_segment_endpoints = protocol.get_voltage_change_endpoints()

    is_found = False
    i = 0

    while not is_found:
        if removal_time_step < vc_segment_endpoints[i]:
            max_segment_idx = i
            is_found = True

        i += 1

    new_start_voltage = protocol.get_voltage_at_time(removal_time_step)
    new_duration = (vc_segment_endpoints[max_segment_idx] - removal_time_step)

    #TODO make separate function and call from remove_end..()
    if isinstance(protocol.steps[max_segment_idx], protocols.VoltageClampRamp):
        new_final_voltage = protocol.steps[max_segment_idx].voltage_end
        new_segment = protocols.VoltageClampRamp(
            new_start_voltage, new_final_voltage, new_duration)
    else:
        new_start_voltage = protocol.steps[max_segment_idx].voltage
        new_segment = protocols.VoltageClampStep(
            new_start_voltage, new_duration)

    new_protocol = protocols.VoltageClampProtocol(
            [new_segment] + protocol.steps[(max_segment_idx + 1):])

    return new_protocol


def get_long_protocol(individual_dictionary, holding_step=2000):
    all_steps = []
    holding_step = protocols.VoltageClampStep(-80, holding_step)
    for current, protocol in individual_dictionary.items():
        all_steps.append(holding_step)
        all_steps += protocol.steps

    long_protocol = protocols.VoltageClampProtocol(all_steps)

    return long_protocol


def main():
    trial_conditions = 'trial_steps_ramps_Kernik_200_50_4_-120_60'
    make_shortened_results(trial_conditions, only_end=False, holding_step=500, with_artefact=True, model_name='Kernik')

if __name__ == '__main__':
    main()
