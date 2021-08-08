import pickle
import mod_protocols as mod_protocols
from cell_models import protocols

for current in ['I_Kr', 'I_Ks', 'I_F', 'I_K1', 'I_Na', 'I_To', 'I_CaL']:
    ga_result = pickle.load(open(f'ga_results/trial_steps_ramps_200_50_4_-120_60/ga_results_{current}_artefact_True', 'rb'))
    #ga_result = './/ga_results_I_Kr_artefact_True'
    for gen in ga_result.generations:
        for ind in gen:
            new_proto = []
            for st in ind.protocol.steps:
                if isinstance(st, protocols.VoltageClampRamp):
                    new_seg = mod_protocols.VoltageClampRamp(st.voltage_start, st.voltage_end, st.duration)
                else:
                    new_seg = mod_protocols.VoltageClampStep(voltage=st.voltage, duration=st.duration)

                
                new_proto.append(new_seg)

            ind.protocol = mod_protocols.VoltageClampProtocol(new_proto)
   
    ga_result = pickle.dump(ga_result, open(f'ga_results/trial_steps_ramps_200_50_4_-120_60/ga_results_{current}_artefact_True', 'wb'))
