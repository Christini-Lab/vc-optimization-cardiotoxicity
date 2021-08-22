from math import log, sqrt
from typing import List
from cell_models.cell_model import CellModel
from cell_models.current_models import Ishi


import numpy as np
from scipy import integrate

from mod_cell_model import CellModel
import mod_protocols as protocols
import mod_trace as trace



class PaciModel(CellModel):
    """An implementation of the Paci2018 model by Paci et al.

    Attributes:
        default_parameters: A dict containing tunable parameters along with
            their default values as specified in Paci et al.
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    # Parameters from optimizer.
    vmax_up_millimolar_per_second = 0.5113
    g_irel_max_millimolar_per_second = 62.5434
    ry_ra_1_micromolar = 0.05354
    ry_ra_2_micromolar = 0.0488
    ry_rahalf_micromolar = 0.02427
    ry_rohalf_micromolar = 0.01042
    ry_rchalf_micromolar = 0.00144
    k_na_ca_a_per_f = 3917.0463
    p_na_k_a_per_f = 2.6351
    k_up_millimolar = 3.1928e-4
    v_leak_per_second = 4.7279e-4
    alpha_in_i_na_ca = 2.5371

    # half-saturation constant millimolar (in i_NaK)
    Km_Na = 40.0

    # Constants (in model parameters).
    f_coulomb_per_mole = 96485.3415
    r_joule_per_mole_kelvin = 8.314472
    t_kelvin = 310.0

    # Cell geometry
    v_sr_micrometer_cube = 583.73
    vc_micrometer_cube = 8800.0
    cm_farad = 9.87109e-11

    # Extracellular concentrations
    nao_millimolar = 151.0
    ko_millimolar = 5.4
    cao_millimolar = 1.8

    #ki_millimolar = 100.0

    # Other variables
    t_drug_application = 10000
    i_na_f_red_med = 1
    i_ca_l_red_med = 1
    i_kr_red_med = 1
    i_ks_red_med = 1

    y_names = [
        'Vm', 'Ca_SR', 'Cai', 'g', 'd', 'f1', 'f2', 'fCa', 'Xr1', 'Xr2', 'Xs',
        'h', 'j', 'm', 'Xf', 'q', 'r', 'Nai', 'm_L', 'h_L', 'RyRa', 'RyRo',
        'RyRc', 'y1'
    ]
    y_initial_zero = [
        -0.070, 0.32, 0.0002, 0, 0, 1, 1, 1, 0, 1, 0, 0.75, 0.75, 0, 0.1, 1, 0,
        9.2, 0, 0.75, 0.3, 0.9, 0.1, 0.000000000000000000e+00
    ]

    default_conductances={
            'G_Na': 3671.2302,
            'G_CaL': 8.635702e-5,
            'G_F': 30.10312,
            'G_Ks': 2.041,
            'G_Kr': 29.8667,
            'G_K1': 28.1492,
            'G_pCa': 0.4125,
            'G_bNa': 0.95,
            'G_bCa': 0.727272,
            'G_NaL': 17.25,
            'K_NaCa': 3917.0463
            }

    def __init__(self, updated_parameters=None,
                 no_ion_selective_dict=None,
                 is_exp_artefact=False,
                 default_time_unit='s',
                 default_voltage_unit='V',
                 concentration_indices={'Ca_SR': 1,
                                        'Cai': 2,
                                        'Nai': 17},
                 ki_millimolar=130,
                 nai_millimolar=None,
                 exp_artefact_params=None):
        """Creates a Paci Model individual

        Leave `updated_parameters` argument empty if generating baseline trace
        with default parameter values.

        Args:
            updated_parameters: Dict of parameters with values to be updated.
            no_ion_selective: Dict of no ion selective scaling factors.
                Baseline is equivalent to all no_ion_selective parameters
                equal to zero
        """
        self.ki_millimolar = ki_millimolar
        self.nai_millimolar = nai_millimolar

        default_parameters = {
            'G_Na': 1,
            'G_CaL': 1,
            'G_F': 1,
            'G_Ks': 1,
            'G_Kr': 1,
            'G_K1': 1,
            'G_To': 1,
            'G_pCa': 1,
            'G_bNa': 1,
            'G_bCa': 1,
            'G_NaL': 1,
            'K_NaCa': 1,
            'P_NaK': 1,
            'G_seal_leak': 1,
            'V_off': 1,
            'R_access': 1
            }

        y_initial = [
            -0.0749228904740065, 0.0936532528714175, 3.79675694306440e-05, 0,
            8.25220533963093e-05, 0.741143500777858, 0.999983958619179,
            0.997742015033076, 0.266113517200784, 0.434907203275640,
            0.0314334976383401, 0.745356534740988, 0.0760523580322096,
            0.0995891726023512, 0.0249102482276486, 0.841714924246004,
            0.00558005376429710, 8.64821066193476, 0.00225383437957339,
            0.0811507312565017, 0.0387066722172937, 0.0260449185736275,
            0.0785849084330126, 0.000000000000000000e+00
        ]

        super().__init__(concentration_indices,
                         y_initial, default_parameters,
                         updated_parameters,
                         no_ion_selective_dict,
                         default_time_unit,
                         default_voltage_unit,
                         is_exp_artefact=is_exp_artefact,
                         exp_artefact_params=exp_artefact_params)

    def action_potential_diff_eq(self, t, y):
        if self.is_exp_artefact:
            d_y = np.zeros(29)
        else:
            d_y = np.zeros(24)

        # Nernst potential
        if self.nai_millimolar is not None:
            y[17] = self.nai_millimolar
        try:
            e_na = self.r_joule_per_mole_kelvin *  self.t_kelvin / self.f_coulomb_per_mole * log( self.nao_millimolar / y[17])
        except ValueError:
            print(f'Intracellular Sodium negative at time {t}')

            y[17] = 8.6
            e_na = self.r_joule_per_mole_kelvin *  self.t_kelvin / self.f_coulomb_per_mole * log( self.nao_millimolar / y[17])


        try:
            e_ca = 0.5 * self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mole * log(
                self.cao_millimolar / y[2])
        except:
            print(f'Intracellular Calcium negative at time {t}')
            y[2] = 1.3E-4
            e_ca = 0.5 * self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mole * log(
                self.cao_millimolar / y[2])

        e_k = self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mole * log(
            self.ko_millimolar / self.ki_millimolar)
        pk_na = 0.03
        e_ks = self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mole * log(
            (self.ko_millimolar + pk_na * self.nao_millimolar) /
            (self.ki_millimolar + pk_na * y[17]))

        # iNa
        i_na = ((t < self.t_drug_application) * 1 + (
                    t >= self.t_drug_application)
                * self.i_na_f_red_med) * self.default_parameters['G_Na'] * y[
                   13] ** 3.0 * y[11] * y[12] * \
               (y[0] - e_na) * self.default_conductances['G_Na']

        h_inf = 1.0 / sqrt(1.0 + np.exp((y[0] * 1000.0 + 72.1) / 5.7))
        alpha_h = 0.057 * np.exp(-(y[0] * 1000.0 + 80.0) / 6.8)
        beta_h = 2.7 * np.exp(
            0.079 * y[0] * 1000.0) + 3.1 * 10.0**5.0 * np.exp(
                0.3485 * y[0] * 1000.0)
        if y[0] < -0.0385:
            tau_h = 1.5 / ((alpha_h + beta_h) * 1000.0)
        else:
            tau_h = 1.5 * 1.6947 / 1000.0

        d_y[11] = (h_inf - y[11]) / tau_h

        j_inf = 1.0 / sqrt(1.0 + np.exp((y[0] * 1000.0 + 72.1) / 5.7))
        if y[0] < -0.04:
            alpha_j = (-25428.0 * np.exp(
                0.2444 * y[0] * 1000.0) - 6.948 * 10.0 ** -6.0
                       * np.exp(-0.04391 * y[0] * 1000.0)) * (
                              y[0] * 1000.0 + 37.78) \
                      / (1.0 + np.exp(0.311 * (y[0] * 1000.0 + 79.23)))
        else:
            alpha_j = 0.0

        if y[0] < -0.04:
            beta_j = ((0.02424 * np.exp(-0.01052 * y[0] * 1000) /
                       (1 + np.exp(-0.1378 * (y[0] * 1000 + 40.14)))))
        else:
            beta_j = ((0.6 * np.exp(0.057 * y[0] * 1000) /
                       (1 + np.exp(-0.1 * (y[0] * 1000 + 32)))))

        tau_j = 7.0 / ((alpha_j + beta_j) * 1000.0)
        d_y[12] = (j_inf - y[12]) / tau_j

        m_inf = 1.0 / (1.0 + np.exp(
            (-y[0] * 1000.0 - 34.1) / 5.9))**(1.0 / 3.0)
        alpha_m = 1.0 / (1.0 + np.exp((-y[0] * 1000.0 - 60.0) / 5.0))
        beta_m = 0.1 / (1.0 + np.exp(
            (y[0] * 1000.0 + 35.0) / 5.0)) + 0.1 / (1.0 + np.exp(
                (y[0] * 1000.0 - 50.0) / 200.0))
        tau_m = 1.0 * alpha_m * beta_m / 1000.0
        d_y[13] = (m_inf - y[13]) / tau_m

        # i NaL
        my_coef_tau_m = 1
        tau_i_na_l_ms = 200
        vh_h_late = 87.61
        i_na_l = self.default_parameters['G_NaL'] * y[18]**3 *\
            y[19] * (y[0] - e_na) * self.default_conductances['G_NaL']

        m_inf_l = 1 / (1 + np.exp(-(y[0] * 1000 + 42.85) / 5.264))
        alpha_m_l = 1 / (1 + np.exp((-60 - y[0] * 1000) / 5))
        beta_m_l = 0.1 / (1 + np.exp(
            (y[0] * 1000 + 35) / 5)) + 0.1 / (1 + np.exp(
                (y[0] * 1000 - 50) / 200))
        tau_m_l = 1 / 1000 * my_coef_tau_m * alpha_m_l * beta_m_l
        d_y[18] = (m_inf_l - y[18]) / tau_m_l

        h_inf_l = 1 / (1 + np.exp((y[0] * 1000 + vh_h_late) / (7.488)))
        tau_h_l = 1 / 1000 * tau_i_na_l_ms
        d_y[19] = (h_inf_l - y[19]) / tau_h_l

        # i f
        e_f_volt = -0.017
        i_f = self.default_parameters['G_F'] * y[14] * \
            (y[0] - e_f_volt) * self.default_conductances['G_F']
        i_f_na = 0.42 * self.default_parameters['G_F'] * y[14] * (
            y[0] - e_na) * self.default_conductances['G_F']

        xf_infinity = 1.0 / (1.0 + np.exp((y[0] * 1000.0 + 77.85) / 5.0))
        tau_xf = 1900.0 / (1.0 + np.exp(
            (y[0] * 1000.0 + 15.0) / 10.0)) / 1000.0
        d_y[14] = (xf_infinity - y[14]) / tau_xf

        # i CaL
        i_ca_l = ((t < self.t_drug_application) * 1 + (
            t >= self.t_drug_application) *
            self.i_ca_l_red_med) * self.default_conductances['G_CaL']\
            * self.default_parameters['G_CaL'] * 4.0 * y[0] \
            * self.f_coulomb_per_mole ** 2.0 / (
            self.r_joule_per_mole_kelvin * self.t_kelvin) * \
            (y[2] * np.exp(2.0 * y[0] * self.f_coulomb_per_mole / (
                self.r_joule_per_mole_kelvin * self.t_kelvin)) - 0.341 *
             self.cao_millimolar) / (
            np.exp(2.0 * y[0] * self.f_coulomb_per_mole / (
                                     self.r_joule_per_mole_kelvin * self.t_kelvin)) - 1.0) * \
                 y[4] * y[5] * y[
                     6] * y[7]

        d_infinity = 1.0 / (1.0 + np.exp(-(y[0] * 1000.0 + 9.1) / 7.0))
        alpha_d = 0.25 + 1.4 / (1.0 + np.exp((-y[0] * 1000.0 - 35.0) / 13.0))
        beta_d = 1.4 / (1.0 + np.exp((y[0] * 1000.0 + 5.0) / 5.0))
        gamma_d = 1.0 / (1.0 + np.exp((-y[0] * 1000.0 + 50.0) / 20.0))
        tau_d = (alpha_d * beta_d + gamma_d) * 1.0 / 1000.0
        d_y[4] = (d_infinity - y[4]) / tau_d

        f1_inf = 1.0 / (1.0 + np.exp((y[0] * 1000.0 + 26.0) / 3.0))
        if f1_inf - y[5] > 0.0:
            const_f1 = 1.0 + 1433.0 * (y[2] - 50.0 * 1.0e-6)
        else:
            const_f1 = 1.0

        tau_f1 = (20.0 + 1102.5 * np.exp(-(
            (y[0] * 1000.0 + 27.0)**2.0 / 15.0)**2.0) + 200.0 / (1.0 + np.exp(
                (13.0 - y[0] * 1000.0) / 10.0)) + 180.0 / (1.0 + np.exp(
                    (30.0 + y[0] * 1000.0) / 10.0))) * const_f1 / 1000.0
        d_y[5] = (f1_inf - y[5]) / tau_f1

        f2_inf = 0.33 + 0.67 / (1.0 + np.exp((y[0] * 1000.0 + 32.0) / 4.0))
        const_f2 = 1.0
        tau_f2 = (600.0 * np.exp(-(y[0] * 1000.0 + 25.0)**2.0 / 170.0) + 31.0 /
                  (1.0 + np.exp(
                      (25.0 - y[0] * 1000.0) / 10.0)) + 16.0 / (1.0 + np.exp(
                          (30.0 + y[0] * 1000.0) / 10.0))) * const_f2 / 1000.0
        d_y[6] = (f2_inf - y[6]) / tau_f2

        alpha_f_ca = 1.0 / (1.0 + (y[2] / 0.0006)**8.0)
        beta_f_ca = 0.1 / (1.0 + np.exp((y[2] - 0.0009) / 0.0001))
        gamma_f_ca = 0.3 / (1.0 + np.exp((y[2] - 0.00075) / 0.0008))
        f_ca_inf = (alpha_f_ca + beta_f_ca + gamma_f_ca) / 1.3156
        if y[0] > -0.06 and f_ca_inf > y[7]:
            const_f_ca = 0.0
        else:
            const_f_ca = 1.0

        tau_f_ca = 0.002  # second (in i_CaL_fCa_gate)
        d_y[7] = const_f_ca * (f_ca_inf - y[7]) / tau_f_ca

        # i to
        g_to_s_per_f = 29.9038
        i_to = self.default_parameters['G_To'] * g_to_s_per_f * (y[0] - e_k) * y[15] * y[16]

        q_inf = 1.0 / (1.0 + np.exp((y[0] * 1000.0 + 53.0) / 13.0))
        tau_q = (6.06 + 39.102 /
                 (0.57 * np.exp(-0.08 * (y[0] * 1000.0 + 44.0)) +
                  0.065 * np.exp(0.1 * (y[0] * 1000.0 + 45.93)))) / 1000.0
        d_y[15] = (q_inf - y[15]) / tau_q

        r_inf = 1.0 / (1.0 + np.exp(-(y[0] * 1000.0 - 22.3) / 18.75))
        tau_r = (2.75352 + 14.40516 /
                 (1.037 * np.exp(0.09 * (y[0] * 1000.0 + 30.61)) +
                  0.369 * np.exp(-0.12 * (y[0] * 1000.0 + 23.84)))) / 1000.0
        d_y[16] = (r_inf - y[16]) / tau_r

        # i Ks
        i_ks = ((t < self.t_drug_application) * 1 + (
                    t >= self.t_drug_application) *
                self.i_ks_red_med) * self.default_parameters['G_Ks'] * (
                           y[0] - e_ks) * y[10] ** 2.0 * \
               (1.0 + 0.6 / (1.0 + (3.8 * 0.00001 / y[2]) ** 1.4)) \
               * self.default_conductances['G_Ks']

        xs_infinity = 1.0 / (1.0 + np.exp((-y[0] * 1000.0 - 20.0) / 16.0))
        alpha_xs = 1100.0 / sqrt(1.0 + np.exp((-10.0 - y[0] * 1000.0) / 6.0))
        beta_xs = 1.0 / (1.0 + np.exp((-60.0 + y[0] * 1000.0) / 20.0))
        tau_xs = 1.0 * alpha_xs * beta_xs / 1000.0
        d_y[10] = (xs_infinity - y[10]) / tau_xs

        # i Kr
        l0 = 0.025
        q = 2.3  # dimensionless (in i_Kr_Xr1_gate)
        i_kr = ((t < self.t_drug_application) * 1 + (
                t >= self.t_drug_application) * self.i_kr_red_med) * \
            self.default_parameters['G_Kr'] *\
            self.default_conductances['G_Kr'] * (
            y[0] - e_k) * y[8] * y[9] * sqrt(
            self.ko_millimolar / 5.4)

        v_half = 1000.0 * (-self.r_joule_per_mole_kelvin * self.t_kelvin /
                           (self.f_coulomb_per_mole * q) * log(
                               (1.0 + self.cao_millimolar / 2.6)**4.0 /
                               (l0 *
                                (1.0 + self.cao_millimolar / 0.58)**4.0)) -
                           0.019)

        xr1_inf = 1.0 / (1.0 + np.exp((v_half - y[0] * 1000.0) / 4.9))
        alpha_xr1 = 450.0 / (1.0 + np.exp((-45.0 - y[0] * 1000.0) / 10.0))
        beta_xr1 = 6.0 / (1.0 + np.exp((30.0 + y[0] * 1000.0) / 11.5))
        tau_xr1 = 1.0 * alpha_xr1 * beta_xr1 / 1000.0
        d_y[8] = (xr1_inf - y[8]) / tau_xr1

        xr2_infinity = 1.0 / (1.0 + np.exp((y[0] * 1000.0 + 88.0) / 50.0))
        alpha_xr2 = 3.0 / (1.0 + np.exp((-60.0 - y[0] * 1000.0) / 20.0))
        beta_xr2 = 1.12 / (1.0 + np.exp((-60.0 + y[0] * 1000.0) / 20.0))
        tau_xr2 = 1.0 * alpha_xr2 * beta_xr2 / 1000.0
        d_y[9] = (xr2_infinity - y[9]) / tau_xr2

        # i K1
        alpha_k1 = 3.91 / (1.0 +
                           np.exp(0.5942 *
                                  (y[0] * 1000.0 - e_k * 1000.0 - 200.0)))
        beta_k1 = (-1.509 * np.exp(0.0002 *
                                   (y[0] * 1000.0 - e_k * 1000.0 + 100.0)) +
                   np.exp(0.5886 * (y[0] * 1000.0 - e_k * 1000.0 - 10.0))) / (
                       1.0 + np.exp(0.4547 * (y[0] * 1000.0 - e_k * 1000.0)))
        xk1_inf = alpha_k1 / (alpha_k1 + beta_k1)
        i_k1 = self.default_parameters['G_K1'] * xk1_inf * (y[0] - e_k) * sqrt(
            self.ko_millimolar / 5.4) * self.default_conductances['G_K1']

        # i NaCa
        km_ca_millimolar = 1.38
        km_nai_millimolar = 87.5
        ksat = 0.1
        gamma = 0.35
        k_na_ca1_a_per_f = self.default_parameters['K_NaCa'] * \
            self.default_conductances['K_NaCa']
        i_na_ca = (k_na_ca1_a_per_f *
                   (np.exp(gamma * y[0] * self.f_coulomb_per_mole /
                           (self.r_joule_per_mole_kelvin * self.t_kelvin)) *
                    y[17]**3.0 * self.cao_millimolar - np.exp(
                        (gamma - 1.0) * y[0] * self.f_coulomb_per_mole /
                        (self.r_joule_per_mole_kelvin * self.t_kelvin)) *
                    self.nao_millimolar**3.0 * y[2] * self.alpha_in_i_na_ca) /
                   ((km_nai_millimolar**3.0 + self.nao_millimolar**3.0) *
                    (km_ca_millimolar + self.cao_millimolar) *
                    (1.0 + ksat * np.exp(
                        (gamma - 1.0) * y[0] * self.f_coulomb_per_mole /
                        (self.r_joule_per_mole_kelvin * self.t_kelvin)))))

        # i NaK
        km_k_millimolar = 1.0
        km_na_millimolar = self.Km_Na
        p_na_k1 = self.p_na_k_a_per_f * self.default_parameters['P_NaK']
        i_na_k = (
            p_na_k1 * self.ko_millimolar /
            (self.ko_millimolar + km_k_millimolar) * y[17] /
            (y[17] + km_na_millimolar) /
            (1.0 +
             0.1245 * np.exp(-0.1 * y[0] * self.f_coulomb_per_mole /
                             (self.r_joule_per_mole_kelvin * self.t_kelvin)) +
             0.0353 * np.exp(-y[0] * self.f_coulomb_per_mole /
                             (self.r_joule_per_mole_kelvin * self.t_kelvin))))

        # i pCa
        kp_ca_millimolar = 0.0005
        i_p_ca = self.default_parameters['G_pCa'] * y[2] / (y[2] +
                    kp_ca_millimolar) * self.default_conductances['G_pCa']

        # Background currents
        i_b_na = self.default_parameters['G_bNa'] * \
            (y[0] - e_na) * self.default_conductances['G_bNa']

        i_b_ca = self.default_parameters['G_bCa'] * \
            (y[0] - e_ca) * self.default_conductances['G_bCa']

        # Sarcoplasmic reticulum
        i_up = self.vmax_up_millimolar_per_second / (
            1.0 + self.k_up_millimolar**2.0 / y[2]**2.0)

        i_leak = (y[1] - y[2]) * self.v_leak_per_second

        d_y[3] = 0

        # RyR
        ry_rsr_cass = (1 - 1 / (1 + np.exp((y[1] - 0.3) / 0.1)))
        i_rel = self.g_irel_max_millimolar_per_second * ry_rsr_cass * y[
            21] * y[22] * (y[1] - y[2])

        ry_rainfss = self.ry_ra_1_micromolar - self.ry_ra_2_micromolar / (
            1 + np.exp((1000 * y[2] - self.ry_rahalf_micromolar) / 0.0082))
        ry_rtauadapt = 1
        d_y[20] = (ry_rainfss - y[20]) / ry_rtauadapt

        ry_roinfss = (1 - 1 / (1 + np.exp(
            (1000 * y[2] - (y[20] + self.ry_rohalf_micromolar)) / 0.003)))
        if ry_roinfss >= y[21]:
            ry_rtauact = 18.75e-3
        else:
            ry_rtauact = 0.1 * 18.75e-3

        d_y[21] = (ry_roinfss - y[21]) / ry_rtauact

        ry_rcinfss = (1 / (1 + np.exp(
            (1000 * y[2] - (y[20] + self.ry_rchalf_micromolar)) / 0.001)))
        if ry_rcinfss >= y[22]:
            ry_rtauinact = 2 * 87.5e-3
        else:
            ry_rtauinact = 87.5e-3

        d_y[22] = (ry_rcinfss - y[22]) / ry_rtauinact

        # Ca2+ buffering
        buf_c_millimolar = 0.25
        buf_sr_millimolar = 10.0
        kbuf_c_millimolar = 0.001
        kbuf_sr_millimolar = 0.3
        cai_bufc = 1.0 / (1.0 + buf_c_millimolar * kbuf_c_millimolar /
                          (y[2] + kbuf_c_millimolar)**2.0)
        ca_sr_buf_sr = 1.0 / (1.0 + buf_sr_millimolar * kbuf_sr_millimolar /
                              (y[1] + kbuf_sr_millimolar)**2.0)

        # Ionic concentrations
        # Nai
        d_y[17] = -self.cm_farad * (
            i_na + i_na_l + i_b_na + 3.0 * i_na_k + 3.0 * i_na_ca + i_f_na) / (
                self.f_coulomb_per_mole * self.vc_micrometer_cube * 1.0e-18)

        # cai
        d_y[2] = cai_bufc * (
            i_leak - i_up + i_rel -
            (i_ca_l + i_b_ca + i_p_ca - 2.0 * i_na_ca) * self.cm_farad /
            (2.0 * self.vc_micrometer_cube * self.f_coulomb_per_mole * 1.0e-18)
        )
        # CaSR
        d_y[1] = ca_sr_buf_sr * self.vc_micrometer_cube / self.v_sr_micrometer_cube * (
            i_up - (i_rel + i_leak))


        i_K1_ishi, d_y[23] = Ishi.I_K1(y[0], e_k, y[23], self.ko_millimolar, 
                0)


        # No Ion Selective
        i_no_ion = 0
        if self.is_no_ion_selective:
            current_dictionary = {
                'I_K1': i_k1,
                'I_To': i_to,
                'I_Kr': i_kr,
                'I_Ks': i_ks,
                'I_CaL': i_ca_l,
                'I_NaK': i_na_k,
                'I_Na': i_na,
                'I_NaL': i_na_l,
                'I_NaCa': i_na_ca,
                'I_pCa': i_p_ca,
                'I_F': i_f,
                'I_bNa': i_b_na,
                'I_bCa': i_b_ca
            }


            conductance_dict = {'I_K1': 'G_K1', 'I_To': 'G_To', 'I_Kr': 'G_Kr',
                                'I_Ks': 'G_Ks', 'I_CaL': 'G_CaL',
                                'I_NaK': 'P_NaK', 'I_Na': 'G_Na',
                                'I_NaL': 'G_NaL', 'I_NaCa': 'K_NaCa',
                                'I_pCa': 'G_pCa', 'I_F': 'G_F',
                                'I_bNa': 'G_bNa', 'I_bCa': 'G_bCa'}

            for curr_name, scale in self.no_ion_selective.items():
                if curr_name == 'I_K1_Ishi':
                    i_no_ion += scale * Ishi.I_K1(y[0] * 1000, e_k * 1000, y[23],
                                                      self.ko_millimolar, 1)[0]
                else:
                    cond_name = conductance_dict[curr_name]
                    i_no_ion += (scale * current_dictionary[curr_name] /
                                        self.default_parameters[cond_name])


        if self.is_exp_artefact:
            i_ion = self.exp_artefacts.c_m*((i_k1 + i_to + i_kr + i_ks +
                i_ca_l + i_na_k + i_na + i_na_l + i_na_ca + i_p_ca +
                i_f + i_b_na + i_b_ca + i_no_ion) - self.i_stimulation)

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

            # y[24] : v_p
            # y[25] : v_clamp
            # y[26] : I_out 
            # y[27] : v_cmd
            # y[28] : v_est

            v_m = y[0]
            v_p = y[24]
            v_clamp = y[25]
            i_out = y[26]
            v_cmd = y[27]
            v_est = y[28]

            i_seal_leak = g_leak * (v_m*1000 - e_leak)

            v_p = 1000*v_cmd + r_access_star * comp_rs * (i_ion + i_seal_leak)

            dvm_dt = (1/r_access/c_m) * (v_p + v_off - v_m*1000) - (
                    i_ion + i_seal_leak) / c_m 

            #dvp_dt = 1/tau_clamp * (v_clamp*1000 - v_p*1000)

            #if comp_predrs < 0.05:
            #    dvest_dt = 0
            #else:
            #    dvest_dt = (v_cmd*1000 - v_est*1000) / ((1 - comp_predrs) *
            #            r_access_star * c_m_star / comp_predrs)

            #vcmd_prime = v_cmd*1000 + ((comp_rs * r_access_star * i_out) +
            #        (comp_predrs * r_access_star * c_m_star * dvest_dt))

            #dvclamp_dt = (vcmd_prime - v_clamp*1000) / tau_sum

            #i_cp = c_p * dvp_dt - c_p_star * dvclamp_dt

            #if r_access_star < 1E-6:
            #    i_cm = c_m_star * dvclamp_dt
            #else:
            #    i_cm = c_m_star * dvest_dt 

            #i_in = (v_p*1000 - v_m*1000 + v_off) / r_access + i_cp - i_cm 

            #di_out_dt = 1000 * (i_in - i_out) / tau_z

            #i_out = y[26]
            d_y[0] = dvm_dt
            #d_y[24] = dvp_dt
            #d_y[25] = dvclamp_dt
            #d_y[26] = di_out_dt
            #d_y[28] = dvest_dt

            i_ion = i_ion / self.exp_artefacts.c_m
            i_seal_leak = i_seal_leak / self.exp_artefacts.c_m
            #i_out = i_out / self.exp_artefacts.c_m
            i_out = i_ion + i_seal_leak
            #i_in = i_in / self.exp_artefacts.c_m

            if self.current_response_info:
                current_timestep = [
                    trace.Current(name='I_K1', value=i_k1),
                    trace.Current(name='I_To', value=i_to),
                    trace.Current(name='I_Kr', value=i_kr),
                    trace.Current(name='I_Ks', value=i_ks),
                    trace.Current(name='I_CaL', value=i_ca_l),
                    trace.Current(name='I_NaK', value=i_na_k),
                    trace.Current(name='I_Na', value=i_na),
                    trace.Current(name='I_NaL', value=i_na_l),
                    trace.Current(name='I_NaCa', value=i_na_ca),
                    trace.Current(name='I_pCa', value=i_p_ca),
                    trace.Current(name='I_F', value=i_f),
                    trace.Current(name='I_bNa', value=i_b_na),
                    trace.Current(name='I_bCa', value=i_b_ca),
                    trace.Current(name='I_ion', value=i_ion),
                    trace.Current(name='I_seal_leak', value=i_seal_leak),
                    trace.Current(name='I_out', value=i_out),
                    #trace.Current(name='I_in', value=i_in),
                    trace.Current(name='I_no_ion', value=i_no_ion),
                ]


        else:
            d_y[0] = -(i_k1 + i_to + i_kr + i_ks + i_ca_l + i_na_k + i_na + i_na_l
                       + i_na_ca + i_p_ca + i_f + i_b_na + i_b_ca + i_no_ion) + self.i_stimulation

            if self.current_response_info:
                current_timestep = [
                    trace.Current(name='I_K1', value=i_k1),
                    trace.Current(name='I_To', value=i_to),
                    trace.Current(name='I_Kr', value=i_kr),
                    trace.Current(name='I_Ks', value=i_ks),
                    trace.Current(name='I_CaL', value=i_ca_l),
                    trace.Current(name='I_NaK', value=i_na_k),
                    trace.Current(name='I_Na', value=i_na),
                    trace.Current(name='I_NaL', value=i_na_l),
                    trace.Current(name='I_NaCa', value=i_na_ca),
                    trace.Current(name='I_pCa', value=i_p_ca),
                    trace.Current(name='I_F', value=i_f),
                    trace.Current(name='I_bNa', value=i_b_na),
                    trace.Current(name='I_bCa', value=i_b_ca),
                    trace.Current(name='I_no_ion', value=i_no_ion)
                ]
        
        if self.current_response_info:
            self.current_response_info.currents.append(current_timestep)

        return d_y

def generate_trace(protocol, tunable_parameters=None, params=None):
    """Generates a trace.

    Leave `params` argument empty if generating baseline trace with
    default parameter values.

    Args:
        tunable_parameters: List of tunable parameters.
        protocol: A protocol object used to generate the trace.
        params: A set of parameter values (where order must match with ordered
            labels in `tunable_parameters`).

    Returns:
        A Trace object.
    """
    new_params = dict()
    if params and tunable_parameters:
        for i in range(len(params)):
            new_params[tunable_parameters[i].name] = params[i]

    return PaciModel(updated_parameters=new_params).generate_response(protocol, is_no_ion_selective=False)
