from math import log, sqrt, floor
from typing import List
import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema

#from cell_models import protocols, trace
#from cell_models.current_models import ExperimentalArtefactsThesis
#from cell_models.protocols import VoltageClampProtocol
import mod_protocols as protocols
import mod_trace as trace
from mod_current_models import ExperimentalArtefactsThesis


class CellModel:
    """An implementation a general cell model
    Attributes:
        default_parameters: A dict containing tunable parameters
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """
    def __init__(self, concentration_indices, y_initial=[], 
                 default_parameters=None, updated_parameters=None,
                 no_ion_selective_dict=None, default_time_unit='s',
                 default_voltage_unit='V', default_voltage_position=0,
                 y_ss=None, is_exp_artefact=False, exp_artefact_params=None):
        self.y_initial = y_initial
        self.default_parameters = default_parameters
        self.no_ion_selective = {}
        self.is_no_ion_selective = False
        self.default_voltage_position = default_voltage_position
        self.y_ss = y_ss
        self.concentration_indices = concentration_indices
        self.i_stimulation = 0
        self.is_exp_artefact = is_exp_artefact
        
        if updated_parameters:
            self.default_parameters.update(updated_parameters)
        if no_ion_selective_dict:
            self.no_ion_selective = no_ion_selective_dict
            self.is_no_ion_selective = True
        if default_time_unit == 's':
            self.time_conversion = 1.0
            self.default_unit = 'standard'
        else:
            self.time_conversion = 1000.0
            self.default_unit = 'milli'
        if default_voltage_unit == 'V':
            self.voltage_conversion = 1
        else:
            self.voltage_conversion = 1000
        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.current_response_info = None
        self.full_y = []
        self.exp_artefacts = ExperimentalArtefactsThesis()

        if exp_artefact_params is not None:
            for k, v in exp_artefact_params.items():
                setattr(self.exp_artefacts, k, v)

        self.exp_artefacts.g_leak *= default_parameters['G_seal_leak']
        v_off_shift = np.log10(default_parameters['V_off']) * 2
        v_off = -2.8 + v_off_shift  #mV
        self.exp_artefacts.v_off += v_off_shift
        default_parameters['R_access'] = 1 + .5 * (
                default_parameters['R_access']-1)
        self.exp_artefacts.r_access *= default_parameters['R_access']
        v_cmd_initial = -80 #mV
        if is_exp_artefact:
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
            # With experimental artefact --------------------------------------
            # 23: Vp (millivolt)
            # 24: Vclamp (millivolt)
            # 25: Iout (nA)
            # 26: Vcmd (millivolt)
            # 27: Vest (millivolt)
            """
            if default_voltage_unit == 'V':
                conversion = 1000
            else:
                conversion = 1
            self.y_initial = np.append(self.y_initial, 0)
            self.y_initial = np.append(self.y_initial, 0)
            self.y_initial = np.append(self.y_initial, 0)
            self.y_initial = np.append(self.y_initial, 0)
            self.cmd_index = len(self.y_initial) - 1
            v_est = v_cmd_initial/conversion
            self.y_initial = np.append(self.y_initial, 0)

    @property
    def no_ion_selective(self):
        return self.__no_ion_selective

    @no_ion_selective.setter
    def no_ion_selective(self, no_ion_selective):
        self.__no_ion_selective = no_ion_selective

    def calc_currents(self, exp_target=None):
        self.current_response_info = trace.CurrentResponseInfo()

        if exp_target is not None:
            i_stim = [exp_target.get_current_at_time(t) for t in self.t * 1000 / self.time_conversion]

        if len(self.y) < 200:
            list(map(self.action_potential_diff_eq, self.t, self.y.transpose()))
        else:
            list(map(self.action_potential_diff_eq, self.t, self.y))

        if exp_target is not None:
            for i, current_timestep in enumerate(self.
                    current_response_info.currents):
                for c in current_timestep:
                    if c.name == 'I_stim':
                        c.value = i_stim[i]

    def generate_response(self, protocol, is_no_ion_selective):
        """Returns a trace based on the specified target objective.
        Args:
            protocol: A Protocol Object or a TargetObjective Object.
        Returns:
            A Trace object representing the change in membrane potential over
            time.
        """
        # Reset instance variables when model is run again.
        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.full_y = []
        self.current_response_info = None

        self.is_no_ion_selective = is_no_ion_selective

        #from cell_models.ga.target_objective import TargetObjective

        if isinstance(protocol, protocols.SpontaneousProtocol):
            return self.generate_spontaneous_response(protocol)
        elif isinstance(protocol, protocols.IrregularPacingProtocol):
            return self.generate_irregular_pacing_response(protocol)
        elif isinstance(protocol, protocols.VoltageClampProtocol):
            return self.generate_VC_protocol_response(protocol)
        elif isinstance(protocol, protocols.PacedProtocol):
            return self.generate_pacing_response(protocol)
        elif isinstance(protocol, protocols.AperiodicPacingProtocol):
            return self.generate_aperiodic_pacing_response(protocol)
        #This means, the input is a Target Objective
        elif isinstance(protocol, TargetObjective):
            #This is if the input Target Objective is a protocol
            if protocol.target_protocol is not None:
                if protocol.g_ishi is not None:
                    self.no_ion_selective = {'I_K1_Ishi': protocol.g_ishi}
                    is_no_ion_selective = True

                return self.generate_response(protocol.target_protocol,
                        is_no_ion_selective)
            #These are if the input Target Objective is exp data
            elif protocol.protocol_type == "Voltage Clamp":
                return self.generate_exp_voltage_clamp(protocol)
            elif protocol.protocol_type == "Dynamic Clamp":
                return self.generate_exp_current_clamp(protocol)

    def find_steady_state(self, ss_type=None, from_peak=False, tol=1E-3,
            max_iters=140):
        """
        Finds the steady state conditions for a spontaneous or stimulated
        (in the case of OR) AP
        """
        if self.y_ss is not None:
            return

        if (ss_type is None):
            protocol = protocols.VoltageClampProtocol(
                [protocols.VoltageClampStep(voltage=-80.0, duration=10000)])

        concentration_indices = list(self.concentration_indices.values())

        is_err = True
        i = 0
        y_values = []

        import time
        outer_time = time.time()

        while is_err:
            init_t = time.time()

            tr = self.generate_response(protocol, is_no_ion_selective=False)

            if isinstance(protocol, protocols.VoltageClampProtocol):
                y_val = self.y[:, -1]
            else:
                y_val = self.get_last_min_max(from_peak)
            self.y_initial = self.y[:, -1]
            self.t = []
            y_values.append(y_val)
            y_percent = []

            if len(y_values) > 2:
                y_percent = np.abs((y_values[i][concentration_indices] -
                                    y_values[i - 1][concentration_indices]) / (
                    y_values[i][concentration_indices]))
                is_below_tol = (y_percent < tol)
                is_err = not is_below_tol.all()

            if i >= max_iters:
                #print("Did not reach steady state. Setting y_ss to last iter.")
                self.y_ss = y_val
                return

            i = i + 1

            if i > 10:
                print(
                    f'Iteration {i}; {time.time() - init_t} seconds; {y_percent}')

        self.y_ss = y_values[-1]
        print(f'Total Time: {time.time() - outer_time}')
        return [tr, i]

    def get_last_min_max(self, from_peak):
        if from_peak:
            inds = argrelextrema(self.y_voltage, np.less)
            last_peak_time = self.t[inds[0][-2]]
            ss_time = last_peak_time - .04*self.time_conversion
            y_val_idx = np.abs(self.t - ss_time).argmin()
        else:
            inds = argrelextrema(self.y_voltage, np.less)
            y_val_idx = inds[0][-2]
        try:
            y_val = self.y[:,y_val_idx]
        except:
            y_val = self.y[y_val_idx,:]

        return y_val

    def generate_spontaneous_function(self):
        def spontaneous(t, y):
            return self.action_potential_diff_eq(t, y)

        return spontaneous

    def generate_spontaneous_response(self, protocol):
        """
        Args:
            protocol: An object of a specified protocol.
        Returns:
            A single action potential trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        solution = integrate.solve_ivp(
            self.generate_spontaneous_function(),
            [0, protocol.duration * self.time_conversion * 1e-3],
            y_init,
            method='BDF',
            max_step=1e-3*self.time_conversion)

        self.t = solution.t
        self.y = solution.y.transpose()
        self.y_initial = self.y[-1]
        self.y_voltage = solution.y[self.default_voltage_position,:]

        self.calc_currents()


        return trace.Trace(protocol,
                           self.default_parameters,
                           self.t,
                           self.y_voltage,
                           current_response_info=self.current_response_info,
                           default_unit=self.default_unit)

    def generate_irregular_pacing_response(self, protocol):
        """
        Args:
            protocol: An irregular pacing protocol 
        Returns:
            A irregular pacing trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        pacing_info = trace.IrregularPacingInfo()

        try:
            solution = integrate.solve_ivp(self.generate_irregular_pacing_function(
                protocol, pacing_info), [0, protocol.duration],
                                y_init,
                                method='BDF',
                                max_step=1e-3*self.time_conversion)

            self.t = solution.t
            self.y = solution.y
            self.y_initial = self.y[-1]
            self.y_voltage = solution.y[self.default_voltage_position,:]


            self.calc_currents()

        except ValueError:
            return None
        return trace.Trace(protocol, self.default_parameter, self.t,
                self.y_voltage, pacing_info=pacing_info,
                default_unit=self.default_unit)

    def generate_irregular_pacing_function(self, protocol, pacing_info):
        offset_times = protocol.make_offset_generator()

        def irregular_pacing(t, y):
            d_y = self.action_potential_diff_eq(t, y)

            if pacing_info.detect_peak(self.t, y[0], self.d_y_voltage):
                pacing_info.peaks.append(t)
                voltage_diff = abs(pacing_info.AVG_AP_START_VOLTAGE - y[0])
                pacing_info.apd_90_end_voltage = y[0] - voltage_diff * 0.9

            if pacing_info.detect_apd_90(y[0]):
                try:
                    pacing_info.add_apd_90(t)
                    pacing_info.stimulations.append(t + next(offset_times))
                except StopIteration:
                    pass

            if pacing_info.should_stimulate(t):
                i_stimulation = protocol.STIM_AMPLITUDE_AMPS / self.cm_farad
            else:
                i_stimulation = 0.0

            d_y[0] += i_stimulation
            return d_y

        return irregular_pacing

    def generate_VC_protocol_response(self, protocol):
        """
        Args:
            protocol: A voltage clamp protocol
        Returns:
            A Trace object for a voltage clamp protocol
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        self.current_response_info = trace.CurrentResponseInfo(
            protocol=protocol)

        solution = integrate.solve_ivp(
            self.generate_voltage_clamp_function(protocol),
            [0, protocol.get_voltage_change_endpoints()[-1] /
                1E3 * self.time_conversion],
            y_init,
            method='BDF',
            max_step=1E-3*self.time_conversion,
            atol=1E-2, rtol=1E-4)

        self.t = solution.t
        self.y = solution.y

        command_voltages = [protocol.get_voltage_at_time(t *
            1E3 / self.time_conversion) / 1E3 * self.time_conversion
            for t in self.t]
        self.command_voltages = command_voltages

        if self.is_exp_artefact:
            self.y_voltages = self.y[0, :]
        else:
            self.y_voltages = command_voltages

        self.calc_currents()

        return trace.Trace(protocol,
                           self.default_parameters,
                           self.t,
                           command_voltages=self.command_voltages,
                           y=self.y_voltages,
                           current_response_info=self.current_response_info,
                           default_unit=self.default_unit)

    def generate_voltage_clamp_function(self, protocol):
        def voltage_clamp(t, y):
            if self.is_exp_artefact:
                try:
                    y[self.cmd_index] = protocol.get_voltage_at_time(t * 1e3 / self.time_conversion)
                # Breaks if Vcmd = 0
                    if y[self.cmd_index] == 0:
                        y[self.cmd_index] = .1
                except:
                    y[self.cmd_index] = 2000

                y[self.cmd_index] /= (1E3 / self.time_conversion)
            else:
                try:
                    y[self.default_voltage_position] = protocol.get_voltage_at_time(t * 1E3 / self.time_conversion)
                    #Can't handle Vcmd = 0
                    if y[self.default_voltage_position] == 0: 
                        y[self.default_voltage_position] = .1
                except:
                    y[self.default_voltage_position] = 2000

                y[self.default_voltage_position] /= (1E3 / self.time_conversion)

            return self.action_potential_diff_eq(t, y)

        return voltage_clamp

    def generate_pacing_response(self, protocol):
        """
        Args:
            protocol: A pacing protocol
        Returns:
            A pacing trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial
        pacing_info = trace.IrregularPacingInfo()
        solution = integrate.solve_ivp(self.generate_pacing_function(
            protocol), [0, protocol.stim_end * self.time_conversion * 1e-3],
                            y_init,
                            method='LSODA',
                            max_step=8e-4*self.time_conversion)
        self.t = solution.t
        self.y = solution.y
        self.y_initial = self.y[:,-1]
        self.y_voltage = solution.y[self.default_voltage_position,:]
        self.calc_currents()
        return trace.Trace(protocol,
                           self.default_parameters,
                self.t, self.y_voltage, pacing_info=pacing_info,
                current_response_info=self.current_response_info,
                default_unit=self.default_unit)

    def generate_pacing_function(self, protocol):
        stim_amplitude = protocol.stim_amplitude * 1E-3 * self.time_conversion
        stim_start = protocol.stim_start * 1E-3 * self.time_conversion
        stim_duration = protocol.stim_duration * 1E-3 * self.time_conversion
        stim_end = protocol.stim_end * 1E-3 * self.time_conversion
        i_stim_period = self.time_conversion / protocol.pace

        if self.time_conversion == 1:
            denom = 1E9
        else:
            denom = 1
            
        def pacing(t, y):
            self.i_stimulation = (stim_amplitude if t - stim_start -\
                i_stim_period*floor((t - stim_start)/i_stim_period) <=\
                stim_duration and t <= stim_end and t >= stim_start else\
                0) / self.cm_farad / denom
            d_y = self.action_potential_diff_eq(t, y)

            return d_y
        return pacing

    def generate_aperiodic_pacing_response(self, protocol):
        """
        Args:
            protocol: A pacing protocol
        Returns:
            A pacing trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial
        solution = integrate.solve_ivp(self.generate_aperiodic_pacing_function(
            protocol), [0, protocol.duration / 1E3 * self.time_conversion],
                            y_init,
                            method='BDF',
                            max_step=1e-3*self.time_conversion)
        self.t = solution.t
        self.y = solution.y
        self.y_initial = self.y[:,-1]
        self.y_voltage = solution.y[self.default_voltage_position,:]
        self.calc_currents()
        return trace.Trace(protocol, self.default_parameters, self.t,
                self.y_voltage, current_response_info=self.current_response_info,
                default_unit=self.default_unit)

    def generate_aperiodic_pacing_function(self, protocol):
        def pacing(t, y):
            for t_start in protocol.stim_starts:
                t_start = t_start / 1000 * self.time_conversion
                t_end = t_start + (protocol.stim_duration /
                        1000 * self.time_conversion)
                if (t > t_start) and (t < t_end):
                    self.i_stimulation = protocol.stim_amplitude
                    break 
                else:
                    self.i_stimulation = 0
            d_y = self.action_potential_diff_eq(t, y)
            return d_y
        return pacing

    def generate_exp_voltage_clamp(self, exp_target):
        """
        Args:
            protocol: A voltage clamp protocol
        Returns:
            A Trace object for a voltage clamp protocol
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial
        self.current_response_info = trace.CurrentResponseInfo(
            protocol=exp_target)
        solution = integrate.solve_ivp(
            self.generate_exp_voltage_clamp_function(exp_target),
            [0, floor(exp_target.time.max()) /
                1E3 * self.time_conversion],
            y_init,
            method='BDF',
            max_step=1e-3*self.time_conversion,
            atol=1E-2, rtol=1E-4)

        self.t = solution.t
        self.y = solution.y
        command_voltages = [exp_target.get_voltage_at_time(t *
            1E3 / self.time_conversion) / 1E3 * self.time_conversion
            for t in self.t]
        self.command_voltages = command_voltages
        if self.is_exp_artefact:
            self.y_voltages = self.y[0, :]
        else:
            self.y_voltages = command_voltages
        self.calc_currents()
        #import matplotlib.pyplot as plt
        #plt.plot(self.t, self.command_voltages)
        #plt.plot(self.t, self.y_voltages)
        #plt.show()
        return trace.Trace(exp_target, self.default_parameters, self.t,
                           command_voltages=self.command_voltages,
                           y=self.y_voltages,
                           current_response_info=self.current_response_info,
                           default_unit=self.default_unit)

    def generate_exp_voltage_clamp_function(self, exp_target):
        def voltage_clamp(t, y):
            if self.is_exp_artefact:
                try:
                    y[26] = exp_target.get_voltage_at_time(t * 1e3 / self.time_conversion)
                    if y[self.cmd_index] == 0:
                        y[self.cmd_index] = .1
                except:
                    y[26] = 20000
                y[26] /= (1E3 / self.time_conversion)
            else:
                try:
                    y[self.default_voltage_position] = exp_target.get_voltage_at_time(t * 1E3 / self.time_conversion)

                    if y[self.default_voltage_position] == 0: 
                        y[self.default_voltage_position] = .1
                except:
                    y[self.default_voltage_position] = 2000
            y[self.default_voltage_position] /= (1E3 / self.time_conversion)
            return self.action_potential_diff_eq(t, y)
        return voltage_clamp
    
    def generate_exp_current_clamp(self, exp_target):
        """
        Args:
            protocol: A voltage clamp protocol
        Returns:
            A Trace object for a voltage clamp protocol
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial
        self.current_response_info = trace.CurrentResponseInfo(
            protocol=exp_target)
        solution = integrate.solve_ivp(
            self.generate_exp_dynamic_clamp_function(exp_target),
            [0, floor(exp_target.time.max()) /
                1E3 * self.time_conversion],
            y_init,
            method='BDF',
            max_step=1e-3*self.time_conversion)
        self.t = solution.t
        self.y = solution.y
        self.y_voltages = self.y[0, :]
        self.calc_currents(exp_target)
        voltages_offset_added = (self.y_voltages +
                self.exp_artefacts['v_off'] / 1000 * self.time_conversion)
        return trace.Trace(exp_target, self.t,
                           y=voltages_offset_added,
                           current_response_info=self.current_response_info,
                           voltages_with_offset=self.y_voltages,
                           default_unit=self.default_unit)

    def generate_exp_dynamic_clamp_function(self, exp_target):
        def dynamic_clamp(t, y):
            r_access = self.exp_artefacts['r_access']
            r_seal = 1 / self.exp_artefacts['g_leak']
            i_access_proportion = r_seal / (r_seal + r_access)
            self.i_stimulation = (-exp_target.get_current_at_time(t * 1000 /
                    self.time_conversion) *
                        i_access_proportion)
            d_y = self.action_potential_diff_eq(t, y)
            return d_y
        return dynamic_clamp

