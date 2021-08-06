from math import log, sqrt 
from typing import List
from math import log, exp

import numpy as np
from scipy import integrate

from mod_cell_model import CellModel
from mod_current_models import KernikCurrents, Ishi
from mod_model_initial import kernik_model_initial
import mod_trace as trace



class KernikModel(CellModel):
    """An implementation of the Kernik model by Kernik et al.

    Attributes:
        updated_parameters: A dict containing all parameters that are being
            tuned.
        
    """
    cm_farad = 60

    # Constants
    t_kelvin = 310.0
    r_joule_per_mole_kelvin = 8.314472
    f_coulomb_per_mmole = 96.4853415

    Ko = 5.4 # millimolar (in model_parameters)
    Cao = 2.0  # millimolar (in model_parameters
    Nao = 137  # millimolar (in model_parameters)

    # half-saturation constant millimolar (in i_NaK)
    Km_Na = 40

    def __init__(self, updated_parameters=None,
                 no_ion_selective_dict=None,
                 default_time_unit='ms',
                 default_voltage_unit='mV',
                 concentration_indices={'Ca_SR': 1, 'Cai': 2,
                                        'Nai': 3, 'Ki': 4},
                 is_exp_artefact=False,
                 ki_millimolar=None,
                 nai_millimolar=None,
                 updated_kinetics=None,
                 model_kinetics_type='Baseline',
                 model_conductances_type='Baseline',
                 exp_artefact_params=None
                 ):
        
        model_parameters_obj = KernikModelParameters()
        if updated_kinetics is None:
            self.kinetics = model_parameters_obj.return_kinetics(
                    model_kinetics_type)
        else:
            self.kinetics = updated_kinetics

        self.conductances = model_parameters_obj.return_conductances(
                model_conductances_type)

        self.kernik_currents = KernikCurrents(self.Ko, self.Cao, self.Nao,
                                              self.t_kelvin,
                                              self.f_coulomb_per_mmole,
                                              self.r_joule_per_mole_kelvin,
                                              model_kinetics=self.kinetics,
                                              model_conductances=self.conductances)

        default_parameters = {
            'G_K1': 1,
            'G_K1_Ishi': 0,
            'G_Kr': 1,
            'G_Ks': 1,
            'G_to': 1,
            'P_CaL': 1,
            'G_CaT': 1,
            'G_Na': 1,
            'G_F': 1,
            'K_NaCa': 1,
            'P_NaK': 1,
            'VmaxUp': 1,
            'V_leak': 1,
            'ks': 1,
            'G_b_Na': 1,
            'G_b_Ca': 1,
            'G_PCa': 1,
            'G_seal_leak': 1,
            'V_off': 1,
            'R_access': 1
        }

        if model_conductances_type == 'Random':
            updated_parameters = self.get_random_conductances(default_parameters)

        y_initial = kernik_model_initial()
        
        self.ki_millimolar = ki_millimolar
        self.nai_millimolar = nai_millimolar

        super().__init__(concentration_indices,
                         y_initial, default_parameters,
                         updated_parameters,
                         no_ion_selective_dict,
                         default_time_unit,
                         default_voltage_unit,
                         is_exp_artefact=is_exp_artefact,
                         exp_artefact_params=exp_artefact_params)


    def get_random_conductances(self, default_parameters, cond_range=10):
        updated_parameters = {}

        for k, val in default_parameters.items():
            if val == 0:
                updated_parameters[k] = val
            elif k in ['G_seal_leak', 'V_off']:
                updated_parameters[k] = val
            else:
                updated_parameters[k] = 10**(np.random.uniform(np.log10(1/cond_range), np.log10(cond_range)))

        return updated_parameters
                

    def action_potential_diff_eq(self, t, y):
        """
        differential equations for Kernik iPSC-CM model
        solved by ODE15s in main_ipsc.m

        # State variable definitions:
        # 0: Vm (millivolt)

        # Ionic Flux: ---------------------------------------------------------
        # 1: Ca_SR (millimolar)
        # 2: Cai (millimolar)
        # 3: Nai (millimolar)
        # 4: Ki (millimolar)

        # Current Gating (dimensionless):--------------------------------------
        # 5: y1    (I_K1 Ishihara)
        # 6: d     (activation in i_CaL)
        # 7: f1    (inactivation in i_CaL)
        # 8: fCa   (calcium-dependent inactivation in i_CaL)
        # 9: Xr1   (activation in i_Kr)
        # 10: Xr2  (inactivation in i_Kr
        # 11: Xs   (activation in i_Ks)
        # 12: h    (inactivation in i_Na)
        # 13: j    (slow inactivation in i_Na)
        # 14: m    (activation in i_Na)
        # 15: Xf   (inactivation in i_f)
        # 16: s    (inactivation in i_to)
        # 17: r    (activation in i_to)
        # 18: dCaT (activation in i_CaT)
        # 19: fCaT (inactivation in i_CaT)
        # 20: R (in Irel)
        # 21: O (in Irel)
        # 22: I (in Irel)
        """

        if self.is_exp_artefact:
            d_y = np.zeros(28)
        else:
            d_y = np.zeros(23)

        if abs(y[0]) > 400:
            return d_y
        # --------------------------------------------------------------------
        # Reversal Potentials:
        if self.nai_millimolar is not None:
            y[3] = self.nai_millimolar

        try:
            E_Ca = (0.5 * self.r_joule_per_mole_kelvin * self.t_kelvin / 
                    self.f_coulomb_per_mmole * log(self.Cao / y[2]))  # millivolt
        except ValueError:
            print(f'Intracellular Calcium calcium negative at time {t}')
            y[2] = 4.88E-5
            E_Ca = 0.5 * self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mmole * log(self.Cao / y[2])  # millivolt
        E_Na = self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mmole * log(self.Nao / y[3])  # millivolt

        if self.ki_millimolar:
            y[4] = self.ki_millimolar
        E_K = self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mmole * log(self.Ko / y[4])  # millivolt

        # --------------------------------------------------------------------
        # Currents:
        i_K1 = self.kernik_currents.i_K1(y[0], E_K, self.default_parameters['G_K1'])
        i_K1_ishi, d_y[5] = Ishi.I_K1(y[0], E_K, y[5], self.Ko, self.default_parameters['G_K1_Ishi'])

        d_y[9], d_y[10], i_Kr = self.kernik_currents.i_Kr(
                y[0], E_K, y[9], y[10], self.default_parameters['G_Kr'])

        d_y[11], i_Ks = self.kernik_currents.i_Ks(
                y[0], E_K, y[11], self.default_parameters['G_Ks'])

        d_y[16], d_y[17], i_to = self.kernik_currents.i_to(
                y[0], E_K, y[16], y[17], self.default_parameters['G_to'])

        d_y[6], d_y[7], d_y[8], i_CaL, i_CaL_Ca, i_CaL_Na, i_CaL_K = \
                self.kernik_currents.i_CaL(
                y[0], y[6], y[7], y[8], y[2], y[3], y[4], 
                self.default_parameters['P_CaL'])

        d_y[18], d_y[19], i_CaT = self.kernik_currents.i_CaT(y[0], E_Ca, y[18], y[19],
                self.default_parameters['G_CaT'])

        d_y[12], d_y[13], d_y[14], i_Na = self.kernik_currents.i_Na(y[0], E_Na,
                y[12], y[13], y[14], self.default_parameters['G_Na'])


        d_y[15], i_f, i_fNa, i_fK = self.kernik_currents.i_f(y[0], E_K, E_Na, y[15], 
                self.default_parameters['G_F'])

        i_NaCa = self.kernik_currents.i_NaCa(y[0], y[2], y[3], self.default_parameters['K_NaCa'])

        i_NaK = self.kernik_currents.i_NaK(y[0], y[3], self.default_parameters['P_NaK'], self.Km_Na)
        
        i_up = self.kernik_currents.i_up(y[2], self.default_parameters['VmaxUp'])

        i_leak = self.kernik_currents.i_leak(y[1], y[2], self.default_parameters['V_leak'])

        d_y[20], d_y[21], d_y[22], i_rel = self.kernik_currents.i_rel(y[1], y[2], 
                y[20], y[21], y[22], self.default_parameters['ks'])

        i_b_Na = self.kernik_currents.i_b_Na(y[0], E_Na,
                self.default_parameters['G_b_Na'])

        i_b_Ca = self.kernik_currents.i_b_Ca(y[0], E_Ca, 
                self.default_parameters['G_b_Ca'])

        i_PCa = self.kernik_currents.i_PCa(y[2], self.default_parameters['G_PCa'])


        # --------------------------------------------------------------------
        # Concentration Changes:
        d_y[1] = self.kernik_currents.Ca_SR_conc(y[1], i_up, i_rel, i_leak)

        d_y[2] = self.kernik_currents.Cai_conc(y[2], i_leak, i_up, i_rel, 
                                         i_CaL_Ca, i_CaT, i_b_Ca,
                                         i_PCa, i_NaCa, self.cm_farad)

        d_y[3] = self.kernik_currents.Nai_conc(i_Na, i_b_Na, i_fNa, i_NaK, i_NaCa, 
                                         i_CaL_Na, self.cm_farad, t)

        d_y[4] = self.kernik_currents.Ki_conc(i_K1, i_to, i_Kr, i_Ks, i_fK,
                                        i_NaK, i_CaL_K, self.cm_farad)
        #d_y[4] = -d_y[3]

        # --------------------------------------------------------------------
        # Handling i_no_ion
        i_no_ion = 0
        if self.is_no_ion_selective:
            current_dictionary = {
                'I_K1':    i_K1,
                'I_To':    i_to,
                'I_Kr':    i_Kr,
                'I_Ks':    i_Ks,
                'I_CaL':   i_CaL_Ca,
                'I_NaK':   i_NaK,
                'I_Na':    i_Na,
                'I_NaCa':  i_NaCa,
                'I_pCa':   i_PCa,
                'I_F':     i_f,
                'I_bNa':   i_b_Na,
                'I_bCa':   i_b_Ca,
                'I_CaT':   i_CaT,
                'I_up':    i_up,
                'I_leak':  i_leak
            }
            
            conductance_dict = {'I_K1': 'G_K1', 'I_To': 'G_to', 'I_Kr': 'G_Kr',
                                'I_Ks': 'G_Ks', 'I_CaL': 'P_CaL',
                                'I_NaK': 'P_NaK', 'I_Na': 'G_Na',
                                'I_NaCa': 'K_NaCa', 'I_pCa': 'G_PCa',
                                'I_F': 'G_F', 'I_bNa': 'G_b_Na',
                                'I_bCa': 'G_b_Ca', 'I_CaT': 'G_CaT',
                                'I_up': 'VmaxUp', 'I_leak': 'V_leak'}

            for curr_name, scale in self.no_ion_selective.items():
                if curr_name == 'I_K1_Ishi':
                    i_no_ion += scale * Ishi.I_K1(y[0], E_K, y[5], self.Ko, 1)[0]
                else:
                    cond_name = conductance_dict[curr_name]
                    i_no_ion += (scale * current_dictionary[curr_name] / 
                                        self.default_parameters[cond_name])

        # -------------------------------------------------------------------
        # Experimental Artefact
        if self.is_exp_artefact:
            ##############Involved##########################
            i_ion = self.exp_artefacts.c_m*((i_K1+i_to+i_Kr+ i_Ks+i_CaL+i_CaT+i_NaK+i_Na+i_NaCa + i_PCa+i_f+i_b_Na+i_b_Ca + i_K1_ishi + i_no_ion) - self.i_stimulation)
            g_leak = self.exp_artefacts.g_leak
            e_leak = self.exp_artefacts.e_leak
            c_m = self.exp_artefacts.c_m
            c_m_star = self.exp_artefacts.c_m_star
            r_access = self.exp_artefacts.r_access
            v_off = self.exp_artefacts.v_off
            tau_clamp = self.exp_artefacts.tau_clamp
            comp_rs = self.exp_artefacts.comp_rs
            comp_predrs = self.exp_artefacts.comp_predrs
            r_access_star = self.exp_artefacts.r_access_star
            tau_sum = self.exp_artefacts.tau_sum
            c_p = self.exp_artefacts.c_p
            c_p_star = self.exp_artefacts.c_p_star
            tau_z = self.exp_artefacts.tau_z

            # y[23] : v_p
            # y[24] : v_clamp
            # y[25] : I_out 
            # y[26] : v_cmd
            # y[27] : v_est

            v_m = y[0]
            v_p = y[23]
            v_clamp = y[24]
            i_out = y[25]
            v_cmd = y[26]
            v_est = y[27]

            i_seal_leak = g_leak * (v_m - e_leak)

            #REMOVE to get thesis version
            v_p = v_cmd + r_access_star * comp_rs * (i_ion + i_seal_leak)

            dvm_dt = (1/r_access/c_m) * (v_p + v_off - y[0]) - (
                    i_ion + i_seal_leak) / c_m 

            #dvp_dt = (v_clamp - v_p) / tau_clamp

            #if comp_predrs < .05:
            #    dvest_dt = 0
            #else:
            #    dvest_dt = (v_cmd - v_est) / ((1 - comp_predrs) *
            #            r_access_star * c_m_star / comp_predrs)

            #vcmd_prime = v_cmd + ((comp_rs * r_access_star * i_out) +
            #        (comp_predrs * r_access_star * c_m_star * dvest_dt))

            #dvclamp_dt = (vcmd_prime - v_clamp) / tau_sum

            #i_cp = c_p * dvp_dt - c_p_star * dvclamp_dt
            #
            #if r_access_star < 1E-6:
            #    i_cm = c_m_star * dvclamp_dt
            #else:
            #    i_cm = c_m_star * dvest_dt

            #i_in = (v_p - v_m + v_off) / r_access #+ i_cp - i_cm

            #di_out_dt = (i_in - i_out) / tau_z

            d_y[0] = dvm_dt
            #d_y[23] = dvp_dt
            #d_y[24] = dvclamp_dt
            #d_y[25] = di_out_dt
            #d_y[27] = dvest_dt

            i_ion = i_ion / self.exp_artefacts.c_m
            i_seal_leak = i_seal_leak / self.exp_artefacts.c_m
            #i_out = i_out / self.exp_artefacts.c_m
            #REMOVE TO GET THESIS VERSION
            i_out = i_ion + i_seal_leak
            #i_cm = i_cm / self.exp_artefacts.c_m
            #i_cp = i_cp / self.exp_artefacts.c_m

            ################################################
            ################################################

            if self.current_response_info:
                current_timestep = [
                    trace.Current(name='I_K1', value=i_K1),
                    trace.Current(name='I_K1_Ishi', value=i_K1_ishi),
                    trace.Current(name='I_To', value=i_to),
                    trace.Current(name='I_Kr', value=i_Kr),
                    trace.Current(name='I_Ks', value=i_Ks),
                    trace.Current(name='I_CaL', value=i_CaL_Ca),
                    trace.Current(name='I_NaK', value=i_NaK),
                    trace.Current(name='I_Na', value=i_Na),
                    trace.Current(name='I_NaCa', value=i_NaCa),
                    trace.Current(name='I_pCa', value=i_PCa),
                    trace.Current(name='I_F', value=i_f),
                    trace.Current(name='I_bNa', value=i_b_Na),
                    trace.Current(name='I_bCa', value=i_b_Ca),
                    trace.Current(name='I_CaT', value=i_CaT),
                    trace.Current(name='I_up', value=i_up),
                    trace.Current(name='I_leak', value=i_leak),
                    trace.Current(name='I_ion', value=i_ion),
                    trace.Current(name='I_seal_leak', value=i_seal_leak),
                    #trace.Current(name='I_Cm', value=i_cm),
                    #trace.Current(name='I_Cp', value=i_cp),
                    #trace.Current(name='I_in', value=i_in),
                    trace.Current(name='I_out', value=i_out),
                    trace.Current(name='I_no_ion', value=i_no_ion),
                ]
                self.current_response_info.currents.append(current_timestep)

        # --------------------------------------------------------------------
        # Calculate change in Voltage and Save currents
        else:
            d_y[0] = -(i_K1+i_to+i_Kr+i_Ks+i_CaL+i_CaT+i_NaK+i_Na+i_NaCa +
                       i_PCa+i_f+i_b_Na+i_b_Ca + i_K1_ishi + i_no_ion) + self.i_stimulation

            if self.current_response_info:
                current_timestep = [
                    trace.Current(name='I_K1', value=i_K1),
                    trace.Current(name='I_K1_Ishi', value=i_K1_ishi),
                    trace.Current(name='I_To', value=i_to),
                    trace.Current(name='I_Kr', value=i_Kr),
                    trace.Current(name='I_Ks', value=i_Ks),
                    trace.Current(name='I_CaL', value=i_CaL_Ca),
                    trace.Current(name='I_NaK', value=i_NaK),
                    trace.Current(name='I_Na', value=i_Na),
                    trace.Current(name='I_NaCa', value=i_NaCa),
                    trace.Current(name='I_pCa', value=i_PCa),
                    trace.Current(name='I_F', value=i_f),
                    trace.Current(name='I_bNa', value=i_b_Na),
                    trace.Current(name='I_bCa', value=i_b_Ca),
                    trace.Current(name='I_CaT', value=i_CaT),
                    trace.Current(name='I_up', value=i_up),
                    trace.Current(name='I_leak', value=i_leak),
                    trace.Current(name='I_no_ion', value=i_no_ion),
                    trace.Current(name='I_stim', value=self.i_stimulation)
                ]
                self.current_response_info.currents.append(current_timestep)

        return d_y


class KernikModelParameters():

    def __init__(self):
        """
        This class will prepare the kinetics and conductance values for
        a given Kernik model.
        Parameters
        ----------
            kinetics – numpy 2d array
                Each row corresponds to one kinetic parameter in the Kernik
                model. The columns are:
                Baseline model, Average model, STD, Min, Max
                For reasons I do not know, the averages are usually, but
                not always, equal to the Baseline model.
            conductances – numpy 2d array
                Each row corresponds to one conductance parameter in the Kernik
                model. The columns are:
                Baseline model, Average model, STD, Min, Max
                For reasons I do not know, the averages are usually, but
                not always, equal to the Baseline model.
                The conductances are in the following order:
                gk1 gkr gks gto gcal gcat gna gf
        """
        self.conductances = np.array([
            [0.133785778, 0.167232222, 0.1539573, 0.02287708, 0.37448125],
            [0.218025, 0.218025, 0.12307354, 0.1173, 0.38556],
            [0.0077, 0.0077, 4.10E-03, 0.003, 0.0105],
            [0.117833333, 0.11783333, 0.08164915, 0.025, 0.1785],
            [0.308027691, 0.30802769, 0.14947741, 0.17779487	, 0.48404881],
            [0.185, 0.185, 0, 0.185, 0.185],
            [9.720613409, 7.77649073, 5.64526927, 3.314905, 14.1230769],
            [0.0435, 0.0725, 0.03889087, 0.045, 0.1]])

        self.kinetics = np.array([
            [0.477994972, 0.477994972, 0.78032536, 0.086091386, 1.643258199],
            [27.24275588, 27.24275588, 46.6925068, 1, 96.93907235],
            [4.925023318, 4.925023318, 4.82603947, 1.543602925, 9.5526684],
            [8.7222376, 5.814825067, 11.4633013, 1, 23.00964732],
            [56.6361975, 56.6361975, 36.8545007, 5.89018309, 53.98242312],
            [0.005748852, 0.005748852, 0.00473475, 0.002413274, 0.012618316],
            [13.62349264, 13.62349264, 3.08586237, 9.982309016, 16.41208169],
            [0.047630571, 0.047630571, 0.04826739, 0.010493187, 0.116829523],
            [-7.06808743, -7.06808743, 1.09408185, -7.837301799, -5.503841634],
            [0.012456641, 0.012456641, 0, 0.012456641, 0.012456641],
            [-25.99445816, -25.99445816, 0, -25.99445816, -25.99445816],
            [37.34263315, 37.34263315, 0, 37.34263315, 37.34263315],
            [22.09196424, 22.09196424, 0, 22.09196424, 22.09196424],
            [50, 50, 0, 50, 50],
            [0, 0, 0, 0, 0],
            [0.001165584, 0.00116558, 1.97E-04, 0.000938402, 0.001279985],
            [66726.83868, 66726.8387, 5.76E+04, 180.516028, 180.516028],
            [0.280458908, 0.28045891, 6.53E-02, 0.226791034, 0.35315329],
            [-18.86697157, -18.866972, 3.81E+00, -22.09064507, -14.66331138],
            [4.74E-06, 4.7412E-06, 8.21E-06, 1.42234E-05, 1.42234E-05],
            [0.055361418, 0.05536142, 0.00650597, 0.047848966, 0.059117682],
            [11.68420234, 11.6842023, 3.01474182, 9.943629418, 15.16532634],
            [3.98918108, 3.98918108, 1.05712329, 3.378844251, 5.209841907],
            [-11.0471393, -11.047139, 3.12062648, -14.65052838, -9.245436067],
            [0.000344231, 0.00034423, 0.00016743, 0.000156649, 0.000478564],
            [-17.63447229, -17.634472, 5.50967004, -21.83632615, -11.39654378],
            [186.7605369, 186.760537, 95.4355319, 76.78365895, 247.8116054],
            [8.180933873, 8.18093387, 1.5661887, 7.245459851, 9.989048928],
            [0.696758421, 0.69675842, 0.28471474, 0.367998153, 0.861139357],
            [11.22445772, 11.2244577, 3.59589485, 9.143422652, 15.37663552],
            [12.96629419, 11.6559002, 3.97928545, 7.724718063, 15.56079568],
            [7.079145965, 6.90656573, 0.94025736, 5.861186262, 7.874815006],
            [0.044909416, 0.05568978, 0.02496024, 0.032552968, 0.088030875],
            [-6.909880369, -6.7488536, 0.88944641, -7.668072635, -5.756217955],
            [0.00051259, 0.00079554, 0.00057811, 0.000345496, 0.001644383],
            [-49.50571203, -57.74437, 17.0315447, -82.46034376, -43.41688999],
            [1931.211224, 1652.96016, 1278.99495, 729.0226806, 3483.755573],
            [5.7300275, 5.38558719, 0.91724962, 4.352266246, 6.184908264],
            [1.658246947, 1.67772716, 0.05215074, 1.612282651, 1.736167781],
            [100.4625592, 99.7825129, 3.44090843, 97.14763413, 104.7168375],
            [108.0458464, 456.733242, 604.009439, 99.17442902, 1154.108035],
            [13.10701573, 11.3992167, 2.97074072, 7.983618551, 13.38191118],
            [0.002326914, 0.00162761, 0.00196027, 0.000229015, 0.003868206],
            [-7.91772629, -7.4266635, 0.90062426, -8.213861508, -6.444537922],
            [0.003626599, 0.0036266, 0.00265009, 0.000960603, 0.0062605],
            [-19.83935886, -19.839359, 2.34170813, -21.62564536, -17.18823882],
            [9663.294977, 9890.78299, 6950.91978, 4042.292718, 17575.45426],
            [7.395503565, 8.10022296, 1.19231566, 6.724140533, 8.825873829],
            [0.000512257, 0.00051226, 0.00020327, 0.000278371, 0.000646264],
            [-66.5837555, -66.583756, 6.10674426, -70.75895034, -59.57496549],
            [0.03197758, 0.03197758, 0.00474363, 0.028876838, 0.037438327],
            [0.167331503, 0.1673315, 0.02829329, 0.149629048, 0.199962502],
            [0.951088725, 0.95108872, 0.32200385, 0.741717293, 1.321874393],
            [5.79E-07, 9.7833E-07, 9.413E-07, 3.1273E-07, 1.64393E-06],
            [-14.58971217, -14.474487, 0.27158838, -14.66652896, -14.282445],
            [20086.65024, 20461.5753, 1060.44821, 19711.72518, 21211.42542],
            [10.20235285, 9.66252506, 1.52686356, 8.582869478, 10.74218063],
            [23.94529135, 23.9452913, 33.8200742, 1, 47.85969513]])

    def return_conductances(self, cond_type='Baseline'):
        """
        The order of conductances in self.conductances is:
            gk1 gkr gks gto gcal gcat gna gf
        """
        if cond_type == 'Baseline':
            cond = self.conductances[:, 0]
        elif cond_type == 'Average':
            cond = self.conductances[:, 1]
        elif cond_type == 'Random':
            cond = self.conductances[:, 0]
        elif cond_type == 'Ma':
            cond = self.get_ma_conductances()

        return dict(zip(['G_K1', 'G_Kr', 'G_Ks', 'G_To', 'G_CaL',
            'G_CaT', 'G_Na', 'G_F'], cond))
                
    def return_kinetics(self, kinetics_type='Baseline'):
        """
        Return values kinetics values for each current
        """
        if kinetics_type == 'Baseline':
            kinetics = self.kinetics[:,0]
        elif kinetics_type == 'Average':
            kinetics = self.kinetics[:, 1]
        elif kinetics_type == 'Random':
            kinetics = self.get_random_kinetics()
        elif kinetics_type == 'Ma':
            kinetics = self.get_ma_kinetics()
            

        return kinetics

    def get_random_kinetics(self):
        rand_kinetics = np.zeros(58)
        average_kinetics = self.kinetics[:, 1]
        min_kinetics = self.kinetics[:, 3]
        max_kinetics = self.kinetics[:, 4]

        for i, k in enumerate(average_kinetics):
            if (min_kinetics[i] == max_kinetics[i]):
                new_val = k
            elif average_kinetics[i] > 0:
                curr_min = min_kinetics[i] * .95 / k
                curr_max = max_kinetics[i] * 1.05 / k
            elif average_kinetics[i] < 0:
                curr_min = max_kinetics[i] * .95 / k
                curr_max = min_kinetics[i] * 1.05 / k

            new_val = k * 10**(np.random.uniform(np.log10(curr_min),
                        np.log10(curr_max)))
                

            rand_kinetics[i] = new_val

        return rand_kinetics

    def get_ma_kinetics(self):
        return [0.086091386, 96.93907235, 9.5526684, 23.00964732,
                5.89018309, 0.005142489, 9.982309016, 0.019198781,
                -5.503841634, 0.012456641, -25.99445816, 37.34263315,
                22.09196424, 50, 0, 0.000938402, 180.516028,
                0.35315329, -14.66331138, 1.42E-05, 0.059117682,
                9.943651267, 3.378844251, -9.245453457, 0.00039748,
                -21.83632615, 247.8116054, 7.245459851, 0.861137753,
                15.37663552, 7.724718063, 6.38882503, 0.088030875,
                -6.265773109, 0.001644383, -82.46034376, 818.2069825,
                4.352266246, 1.736167781, 97.7423739, 99.17442902,
                12.83212028, 0.003868206, -8.213861508, 0.0062605,
                -21.62564536, 17575.45426, 6.724140533, 0.000646264,
                -69.41735068, 0.037438327, 0.199962502, 1.321874393,
                1.64E-06, -14.282445, 21211.42542, 8.582869478,
                0.030887565]

    def get_ma_conductances(self):
        return [0.084112429, 0.1173, 0.003, 0.025, 0.190164645,
                0.185, 14.12307692, 0.045]
