import pickle
from os import listdir
import matplotlib.pyplot as plt
import mod_protocols as protocols
import mod_kernik as kernik


def plot_shorten_with_vcmd(trial_conditions, model_name, is_vc_only=False):
    path_to_data = f"ga_results/{trial_conditions}"

    files = listdir(path_to_data)

    for f in files:
        if ('shorten' in f) and ('pkl' in f):
            file_name = f
    
    short_protocol = pickle.load(open(f"{path_to_data}/{file_name}", 'rb'))

    print(f'The protocol is {short_protocol.get_voltage_change_endpoints()[-1]} ms')

    if is_vc_only:
        short_protocol.plot_voltage_clamp_protocol()
        return

    max_currents = get_max_currents(short_protocol, 500, 2, model_name)
    max_currents.to_csv(f"{path_to_data}/{file_name}_max_currents.csv")

    print(max_currents)

    i_trace = get_trace(kernik.KernikModel(is_exp_artefact=True), short_protocol)

    i_trace.plot_with_individual_currents(currents=['I_Kr', 'I_CaL'], with_artefacts=True)


def get_max_currents(vc_protocol, prestep, window, model_name, step_size=None):
    if step_size is None:
        step_size = window

    model = kernik.KernikModel(is_exp_artefact=True)
    scale = 1

    i_trace = get_trace(model, vc_protocol, prestep=prestep)

    max_currents = i_trace.current_response_info.get_max_current_contributions( i_trace.t, window=window/scale, step_size=step_size/scale)

    return max_currents


def get_trace(model, protocol, prestep=2000, with_artefact=False):
    prestep_protocol = protocols.VoltageClampProtocol(
        [protocols.VoltageClampStep(voltage=-80.0,
                                    duration=prestep)])

    model.generate_response(prestep_protocol, is_no_ion_selective=False)

    model.y_ss = model.y[:, -1]

    response_trace = model.generate_response(protocol, is_no_ion_selective=False)

    return response_trace


def main():
    trial_conditions = "trial_steps_ramps_Kernik_200_50_4_-120_60"
    model_name = 'Kernik'

    plot_shorten_with_vcmd(trial_conditions, model_name, is_vc_only=True)


if __name__ == '__main__':
    main()
