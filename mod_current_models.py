import numpy as np
from math import log, sqrt, exp


class KernikCurrents():
    """An implementation of all Kernik currents by Kernik et al.

    Attributes:
        default_parameters: A dict containing tunable parameters along with
            their default values as specified in Kernik et al.
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    Cm = 60
    V_tot = 3960
    Vc_tenT = 16404
    VSR_tenT = 1094
    V_tot_tenT = Vc_tenT + VSR_tenT
    Vc = V_tot * (Vc_tenT / V_tot_tenT)
    V_SR = V_tot * (VSR_tenT / V_tot_tenT)

    def __init__(self, Ko, Cao, Nao, t_kelvin=310.0,
                 f_coulomb_per_mmole=96.4853415,
                 r_joule_per_mole_kelvin=8.314472,
                 model_kinetics=None,
                 model_conductances=None):
        self.Ko = Ko # millimolar (in model_parameters)
        self.Cao = Cao  # millimolar (in model_parameters
        self.Nao = Nao  # millimolar (in model_parameters)

        self.t_kelvin = t_kelvin  
        self.r_joule_per_mole_kelvin = r_joule_per_mole_kelvin  
        self.f_coulomb_per_mmole = f_coulomb_per_mmole 

        if model_kinetics is None:
            model_parameters = KernikModelParameters()
            model_kinetics = model_parameters.return_kinetics()

        self.x_K1 = model_kinetics[0:5]
        self.x_KR = model_kinetics[5:15]
        self.x_IKS = model_kinetics[15:20]
        self.xTO = model_kinetics[20:30]
        self.x_cal = model_kinetics[30:40]
        self.x_NA = model_kinetics[40:53]
        self.x_F = model_kinetics[53:]

        if model_conductances is None:
            model_parameters = KernikModelParameters()
            self.model_conductances = model_parameters.return_conductances()
        self.model_conductances = model_conductances
        
    def i_K1(self, v_m, E_K, g_K1):
        xK11 = self.x_K1[0]
        xK12 = self.x_K1[1]
        xK13 = self.x_K1[2]
        xK14 = self.x_K1[3]
        xK15 = self.x_K1[4]

        alpha_xK1 = xK11*exp((v_m+xK13)/xK12)
        beta_xK1 = exp((v_m+xK15)/xK14)
        XK1_inf = alpha_xK1/(alpha_xK1+beta_xK1)

        # Current:
        g_K1 = self.model_conductances['G_K1'] * g_K1 
        return g_K1*XK1_inf*(v_m-E_K)*sqrt(self.Ko/5.4)

    def i_Kr(self, v_m, E_K, Xr1, Xr2, g_Kr):
        # define parameters from x_KR
        Xr1_1 = self.x_KR[0]
        Xr1_2 = self.x_KR[1]
        Xr1_5 = self.x_KR[2]
        Xr1_6 = self.x_KR[3]
        Xr2_1 = self.x_KR[4]
        Xr2_2 = self.x_KR[5]
        Xr2_5 = self.x_KR[6]
        Xr2_6 = self.x_KR[7]

        # parameter-dependent values:
        Xr1_3 = Xr1_5*Xr1_1
        Xr2_3 = Xr2_5*Xr2_1
        Xr1_4 = 1/((1/Xr1_2)+(1/Xr1_6))
        Xr2_4 = 1/((1/Xr2_2)+(1/Xr2_6))

        # 10: Xr1 (dimensionless) (activation in i_Kr_Xr1)
        alpha_Xr1 = Xr1_1*exp((v_m)/Xr1_2)
        beta_Xr1 = Xr1_3*exp((v_m)/Xr1_4)
        Xr1_inf = alpha_Xr1/(alpha_Xr1 + beta_Xr1)
        tau_Xr1 = ((1./(alpha_Xr1 + beta_Xr1))+self.x_KR[8])
        d_Xr1 = (Xr1_inf-Xr1)/tau_Xr1

        # 11: Xr2 (dimensionless) (inactivation in i_Kr_Xr2)
        alpha_Xr2 = Xr2_1*exp((v_m)/Xr2_2)
        beta_Xr2 = Xr2_3*exp((v_m)/Xr2_4)
        Xr2_inf = alpha_Xr2/(alpha_Xr2+beta_Xr2)
        tau_Xr2 = ((1./(alpha_Xr2+beta_Xr2))+self.x_KR[9])
        d_Xr2 = (Xr2_inf-Xr2)/tau_Xr2

        # Current:
        g_Kr = self.model_conductances['G_Kr'] * g_Kr  # nS_per_pF (in i_Kr)
        i_Kr = g_Kr*(v_m-E_K)*Xr1*Xr2*sqrt(self.Ko/5.4)
        return [d_Xr1, d_Xr2, i_Kr]

    def i_Ks(self, v_m, E_K, Xs, g_Ks):
        ks1 = self.x_IKS[0]
        ks2 = self.x_IKS[1]
        ks5 = self.x_IKS[2]
        ks6 = self.x_IKS[3]
        tauks_const = self.x_IKS[4]

        # parameter-dependent values:
        ks3 = ks5*ks1
        ks4 = 1/((1/ks2)+(1/ks6))

        # 12: Xs (dimensionless) (activation in i_Ks)
        alpha_Xs = ks1*exp((v_m)/ks2)
        beta_Xs = ks3*exp((v_m)/ks4)
        Xs_inf = alpha_Xs/(alpha_Xs+beta_Xs)
        tau_Xs = (1./(alpha_Xs+beta_Xs)) + tauks_const
        d_Xs = (Xs_inf-Xs)/tau_Xs

        # Current:
        g_Ks = self.model_conductances['G_Ks']*g_Ks  # nS_per_pF (in i_Ks)
        i_Ks = g_Ks*(v_m-E_K)*(Xs**2)

        return [d_Xs, i_Ks]

    def i_to(self, v_m, E_K, s, r, g_to):
        # Transient outward current (Ito): define parameters from xTO
        r1 = self.xTO[0]
        r2 = self.xTO[1]
        r5 = self.xTO[2]
        r6 = self.xTO[3]
        s1 = self.xTO[4]
        s2 = self.xTO[5]
        s5 = self.xTO[6]
        s6 = self.xTO[7]
        tau_r_const = self.xTO[8]
        tau_s_const = self.xTO[9]

        # parameter-dependent values:
        r3 = r5*r1
        r4 = 1/((1/r2)+(1/r6))
        s3 = s5*s1
        s4 = 1/((1/s2)+(1/s6))

        # 17: s (dimensionless) (inactivation in i_to)
        alpha_s = s1*exp((v_m)/s2)
        beta_s = s3*exp((v_m)/s4)
        s_inf = alpha_s/(alpha_s+beta_s)
        tau_s = ((1./(alpha_s+beta_s))+tau_s_const)
        d_s = (s_inf-s)/tau_s

        # 18: r (dimensionless) (activation in i_to)
        alpha_r = r1*exp((v_m)/r2)
        beta_r = r3*exp((v_m)/r4)
        r_inf = alpha_r/(alpha_r + beta_r)
        tau_r = (1./(alpha_r + beta_r))+tau_r_const
        d_r = (r_inf-r)/tau_r

        # Current:
        g_to = self.model_conductances['G_To']*g_to  # nS_per_pF (in i_to)
        i_to = g_to*(v_m-E_K)*s*r
        return [d_s, d_r, i_to]

    def i_CaL(self, v_m, d, f, fCa, Cai, Nai, Ki, p_CaL):
        d1 = self.x_cal[0]
        d2 = self.x_cal[1]
        d5 = self.x_cal[2]
        d6 = self.x_cal[3]
        f1 = self.x_cal[4]
        f2 = self.x_cal[5]
        f5 = self.x_cal[6]
        f6 = self.x_cal[7]
        taud_const = self.x_cal[8]
        tauf_const = self.x_cal[9]

        # parameter-dependent values:
        d3 = d5*d1
        d4 = 1/((1/d2)+(1/d6))
        f3 = f5*f1
        f4 = 1/((1/f2)+(1/f6))

        # 7: d (dimensionless) (activation in i_CaL)
        alpha_d = d1*exp(((v_m))/d2)
        beta_d = d3*exp(((v_m))/d4)
        d_inf = alpha_d/(alpha_d + beta_d)
        tau_d = ((1/(alpha_d + beta_d))+taud_const)
        d_d = (d_inf-d)/tau_d

        # 8: f (dimensionless) (inactivation  i_CaL)
        alpha_f = f1*exp(((v_m))/f2)
        beta_f = f3*exp(((v_m))/f4)
        f_inf = alpha_f/(alpha_f+beta_f)
        tau_f = ((1./(alpha_f+beta_f)) + tauf_const)
        d_f = (f_inf-f)/tau_f

        # 9: fCa (dimensionless) (calcium-dependent inactivation in i_CaL)
        # from Ten tusscher 2004
        scale_Ical_Fca_Cadep = 1.2
        alpha_fCa = 1.0/(1.0+((scale_Ical_Fca_Cadep*Cai)/.000325) ** 8.0)

        try:
            beta_fCa = 0.1/(1.0+exp((scale_Ical_Fca_Cadep*Cai-.0005)/0.0001))
        except OverflowError:
            beta_fCa_exp = (scale_Ical_Fca_Cadep*Cai-.0005)/0.0001

            if beta_fCa_exp > 50:
                beta_fCa = 0
            else:
                beta_fCa = 0.1

        gamma_fCa = .2/(1.0+exp((scale_Ical_Fca_Cadep*Cai-0.00075)/0.0008))

        fCa_inf = ((alpha_fCa+beta_fCa+gamma_fCa+.23)/(1.46))
        tau_fCa = 2  # ms
        if ((fCa_inf > fCa) and (v_m > -60)):
            k_fca = 0
        else:
            k_fca = 1

        d_fCa = k_fca*(fCa_inf-fCa)/tau_fCa

        # Current
        p_CaL = self.model_conductances['G_CaL']*p_CaL  # nS_per_pF (in i_CaL)
        p_CaL_shannonCa = 5.4e-4
        p_CaL_shannonNa = 1.5e-8
        p_CaL_shannonK = 2.7e-7
        p_CaL_shannonTot = p_CaL_shannonCa + p_CaL_shannonNa + p_CaL_shannonK
        p_CaL_shannonCap = p_CaL_shannonCa/p_CaL_shannonTot
        p_CaL_shannonNap = p_CaL_shannonNa/p_CaL_shannonTot
        p_CaL_shannonKp = p_CaL_shannonK/p_CaL_shannonTot

        p_CaL_Ca = p_CaL_shannonCap*p_CaL
        p_CaL_Na = p_CaL_shannonNap*p_CaL
        p_CaL_K = p_CaL_shannonKp*p_CaL

        ibarca = p_CaL_Ca*4.0*v_m*self.f_coulomb_per_mmole ** 2.0/(self.r_joule_per_mole_kelvin*self.t_kelvin) * (.341*Cai*exp(
            2.0*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-0.341*self.Cao)/(exp(2.0*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-1.0)
        i_CaL_Ca = ibarca * d*f*fCa

        ibarna = p_CaL_Na * \
            v_m*self.f_coulomb_per_mmole ** 2.0/(self.r_joule_per_mole_kelvin*self.t_kelvin) * (.75*Nai*exp(v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin)) -
                                  0.75*self.Nao)/(exp(v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-1.0)
        i_CaL_Na = ibarna * d*f*fCa

        ibark = p_CaL_K*v_m*self.f_coulomb_per_mmole ** 2.0/(self.r_joule_per_mole_kelvin*self.t_kelvin) * (.75*Ki *
                                              exp(v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-0.75*self.Ko)/(exp(
                                                  v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-1.0)
        i_CaL_K = ibark * d*f*fCa

        i_CaL = i_CaL_Ca+i_CaL_Na+i_CaL_K

        return [d_d, d_f, d_fCa, i_CaL, i_CaL_Ca, i_CaL_Na, i_CaL_K]

    def i_CaT(self, v_m, E_Ca, dCaT, fCaT, g_CaT):
        # 19: dCaT (activation in i_CaT)
        dcat_inf = 1./(1+exp(-((v_m) + 26.3)/6))
        tau_dcat = 1./(1.068*exp(((v_m)+26.3)/30) + 1.068*exp(-((v_m)+26.3)/30))
        d_dCaT = (dcat_inf-dCaT)/tau_dcat

        # 20: fCaT (inactivation in i_CaT)
        fcat_inf = 1./(1+exp(((v_m) + 61.7)/5.6))
        tau_fcat = 1./(.0153*exp(-((v_m)+61.7)/83.3) + 0.015*exp(
            ((v_m)+61.7)/15.38))
        d_fCaT = (fcat_inf-fCaT)/tau_fcat

        g_CaT = self.model_conductances['G_CaT']*g_CaT # nS_per_pF (in i_CaT)
        i_CaT = g_CaT*(v_m-E_Ca)*dCaT*fCaT

        return [d_dCaT, d_fCaT, i_CaT]

    def i_Na(self, v_m, E_Na, h, j, m, g_Na):
        # Sodium Current (INa):
        # define parameters from x_Na
        m1 = self.x_NA[0]
        m2 = self.x_NA[1]
        m5 = self.x_NA[2]
        m6 = self.x_NA[3]
        h1 = self.x_NA[4]
        h2 = self.x_NA[5]
        h5 = self.x_NA[6]
        h6 = self.x_NA[7]
        j1 = self.x_NA[8]
        j2 = self.x_NA[9]
        tau_m_const = self.x_NA[10]
        tau_h_const = self.x_NA[11]
        tau_j_const = self.x_NA[12]

        # parameter-dependent values:
        m3 = m5*m1
        m4 = 1/((1/m2)+(1/m6))
        h3 = h5*h1
        h4 = 1/((1/h2)+(1/h6))
        j5 = h5
        j6 = h6
        j3 = j5*j1
        j4 = 1/((1/j2)+(1/j6))

        # 13: h (dimensionless) (inactivation in i_Na)
        alpha_h = h1*exp((v_m)/h2)
        beta_h = h3*exp((v_m)/h4)
        h_inf = (alpha_h/(alpha_h+beta_h))
        tau_h = ((1./(alpha_h+beta_h))+tau_h_const)
        d_h = (h_inf-h)/tau_h

        # 14: j (dimensionless) (slow inactivation in i_Na)
        alpha_j = j1*exp((v_m)/j2)
        beta_j = j3*exp((v_m)/j4)
        j_inf = (alpha_j/(alpha_j+beta_j))
        tau_j = ((1./(alpha_j+beta_j))+tau_j_const)
        d_j = (j_inf-j)/tau_j

        # 15: m (dimensionless) (activation in i_Na)
        alpha_m = m1*exp((v_m)/m2)
        beta_m = m3*exp((v_m)/m4)
        m_inf = alpha_m/(alpha_m+beta_m)
        tau_m = ((1./(alpha_m+beta_m))+tau_m_const)
        d_m = (m_inf-m)/tau_m

        # Current:
        g_Na = self.model_conductances['G_Na']*g_Na
        # nS_per_pF (in i_Na)
        i_Na = g_Na*m ** 3.0*h*j*(v_m-E_Na)

        return [d_h, d_j, d_m, i_Na]

    def i_f(self, v_m, E_K, E_Na, Xf, g_f):
        # Funny/HCN current (If):
        # define parameters from x_F
        xF1 = self.x_F[0]
        xF2 = self.x_F[1]
        xF5 = self.x_F[2]
        xF6 = self.x_F[3]
        xF_const = self.x_F[4]

        # parameter-dependent values:
        xF3 = xF5*xF1
        xF4 = 1/((1/xF2)+(1/xF6))

        # 16: Xf (dimensionless) (inactivation in i_f)
        alpha_Xf = xF1*exp((v_m)/xF2)
        beta_Xf = xF3*exp((v_m)/xF4)
        Xf_inf = alpha_Xf/(alpha_Xf+beta_Xf)
        tau_Xf = ((1./(alpha_Xf+beta_Xf))+xF_const)
        d_Xf = (Xf_inf-Xf)/tau_Xf

        # Current:
        g_f = self.model_conductances['G_F']*g_f
        # nS_per_pF (in i_f)
        NatoK_ratio = .491  # Verkerk et al. 2013
        Na_frac = NatoK_ratio/(NatoK_ratio+1)
        i_fNa = Na_frac*g_f*Xf*(v_m-E_Na)
        i_fK = (1-Na_frac)*g_f*Xf*(v_m-E_K)
        i_f = i_fNa+i_fK

        return [d_Xf, i_f, i_fNa, i_fK]

    def i_NaCa(self, v_m, Cai, Nai, k_NaCa):
        # Na+/Ca2+ Exchanger current (INaCa):
        # Ten Tusscher formulation
        KmCa = 1.38    # Cai half-saturation constant millimolar (in i_NaCa)
        KmNai = 87.5    # Nai half-saturation constnat millimolar (in i_NaCa)
        Ksat = 0.1    # saturation factor dimensionless (in i_NaCa)
        gamma = 0.35*2    # voltage dependence parameter dimensionless (in i_NaCa)
        # factor to enhance outward nature of inaca dimensionless (in i_NaCa)
        alpha = 2.5*1.1
        # maximal inaca pA_per_pF (in i_NaCa)
        kNaCa = 1000*1.1*k_NaCa
        i_NaCa = kNaCa*((exp(gamma*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))*(Nai ** 3.0)*self.Cao)-(exp(
            (gamma-1.0)*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))*(
            self.Nao ** 3.0)*Cai*alpha))/(((KmNai ** 3.0)+(self.Nao ** 3.0))*(KmCa+self.Cao)*(
                1.0+Ksat*exp((gamma-1.0)*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))))

        return i_NaCa

    def i_NaK(self, v_m, Nai, p_NaK, Km_Na=40.0):
        Km_K = 1.0    # Ko half-saturation constant millimolar (in i_NaK)
        # maxiaml nak pA_per_pF (in i_NaK)
        PNaK = 1.362*1.818*p_NaK
        i_NaK = PNaK*((self.Ko*Nai)/((self.Ko+Km_K)*(Nai+Km_Na)*(1.0 + 0.1245*exp(
            -0.1*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))+0.0353*exp(-v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin)))))

        return i_NaK

    def i_up(self, Cai, v_max_up):
        # SR Uptake/SERCA (J_up):
        # Ten Tusscher formulation
        Kup = 0.00025*0.702    # millimolar (in calcium_dynamics)
        # millimolar_per_milisecond (in calcium_dynamics)
        VmaxUp = 0.000425 * 0.26 * v_max_up
        i_up = VmaxUp/(1.0+Kup ** 2.0/Cai ** 2.0)

        return i_up

    def i_leak(self, Ca_SR, Cai, V_leak):
        # SR Leak (J_leak):
        # Ten Tusscher formulation
        V_leak = V_leak*0.00008*0.02
        i_leak = (Ca_SR-Cai)*V_leak

        return i_leak

    def i_rel(self, Ca_SR, Cai, R, O, I, ks):
        ks = 12.5*ks  # [1/ms]
        koCa = 56320*11.43025              # [mM**-2 1/ms]
        kiCa = 54*0.3425                   # [1/mM/ms]
        kom = 1.5*0.1429                   # [1/ms]
        kim = 0.001*0.5571                 # [1/ms]
        ec50SR = 0.45
        MaxSR = 15
        MinSR = 1

        kCaSR = MaxSR - (MaxSR-MinSR)/(1+(ec50SR/Ca_SR)**2.5)
        koSRCa = koCa/kCaSR
        kiSRCa = kiCa*kCaSR
        RI = 1-R-O-I

        d_R = (kim*RI-kiSRCa*Cai*R) - (koSRCa*Cai**2*R-kom*O)
        d_O = (koSRCa*Cai**2*R-kom*O) - (kiSRCa*Cai*O-kim*I)
        d_I = (kiSRCa*Cai*O-kim*I) - (kom*I-koSRCa*Cai**2*RI)

        i_rel = ks*O*(Ca_SR-Cai)*(self.V_SR/self.Vc)

        return [d_R, d_O, d_I, i_rel]

    def i_b_Na(self, v_m, E_Na, g_b_Na):
        g_b_Na = .00029*1.5*g_b_Na    # nS_per_pF (in i_b_Na)
        i_b_Na = g_b_Na*(v_m-E_Na)

        return i_b_Na

    def i_b_Ca(self, v_m, E_Ca, g_b_Ca):
        g_b_Ca = .000592*0.62*g_b_Ca    # nS_per_pF (in i_b_Ca)
        i_b_Ca = g_b_Ca*(v_m-E_Ca)

        return i_b_Ca

    def i_PCa(self, Cai, g_PCa): # SL Pump
        g_PCa = 0.025*10.5*g_PCa    # pA_per_pF (in i_PCa)
        KPCa = 0.0005    # millimolar (in i_PCa)
        i_PCa = g_PCa*Cai/(Cai+KPCa)

        return i_PCa

    def Ca_SR_conc(self, Ca_SR, i_up, i_rel, i_leak):
        # 2: CaSR (millimolar)
        # rapid equilibrium approximation equations --
        # not as formulated in ten Tusscher 2004 text
        Buf_SR = 10.0*1.2  # millimolar (in calcium_dynamics)
        Kbuf_SR = 0.3  # millimolar (in calcium_dynamics)
        Ca_SR_bufSR = 1/(1.0+Buf_SR*Kbuf_SR/(Ca_SR+Kbuf_SR)**2.0)

        d_Ca_SR = Ca_SR_bufSR*self.Vc/self.V_SR*(i_up-(i_rel+i_leak))

        return d_Ca_SR

    def Cai_conc(self, Cai, i_leak, i_up, i_rel, i_CaL_Ca, i_CaT, 
                 i_b_Ca, i_PCa, i_NaCa, Cm):
        # 3: Cai (millimolar)
        # rapid equilibrium approximation equations --
        # not as formulated in ten Tusscher 2004 text
        Buf_C = .06  # millimolar (in calcium_dynamics)
        Kbuf_C = .0006  # millimolar (in calcium_dynamics)
        Cai_bufc = 1/(1.0+Buf_C*Kbuf_C/(Cai+Kbuf_C)**2.0)


        d_Cai = (Cai_bufc)*(i_leak-i_up+i_rel - 
                (i_CaL_Ca+i_CaT+i_b_Ca+i_PCa-2*i_NaCa)*Cm/(2.0*self.Vc*self.f_coulomb_per_mmole))

        return d_Cai

    def Nai_conc(self, i_Na, i_b_Na, i_fNa, i_NaK, i_NaCa, i_CaL_Na, Cm, t):
        # 4: Nai (millimolar) (in sodium_dynamics)
        d_Nai = -Cm*(i_Na+i_b_Na+i_fNa+3.0*i_NaK+3.0*i_NaCa + 
                    i_CaL_Na)/(self.f_coulomb_per_mmole*self.Vc)

        return d_Nai

    def Ki_conc(self, i_K1, i_to, i_Kr, i_Ks, i_fK, i_NaK, i_CaL_K, Cm):
        d_Ki = -Cm*(i_K1+i_to+i_Kr+i_Ks+i_fK - 2.*i_NaK + 
                    i_CaL_K)/(self.f_coulomb_per_mmole*self.Vc)

        return d_Ki


class Ishi():
    Mg_in = 1
    SPM_in = 0.005
    phi = 0.9

    def __init__(self):
        pass

    @classmethod
    def I_K1(cls, V, E_K, y1, K_out, g_K1):
        IK1_alpha = (0.17*exp(-0.07*((V-E_K) + 8*cls.Mg_in)))/(1+0.01*exp(0.12*(V-E_K)+8*cls.Mg_in))
        IK1_beta = (cls.SPM_in*280*exp(0.15*(V-E_K)+8*cls.Mg_in))/(1+0.01*exp(0.13*(V-E_K)+8*cls.Mg_in));
        Kd_spm_l = 0.04*exp(-(V-E_K)/9.1);
        Kd_mg = 0.45*exp(-(V-E_K)/20);
        fo = 1/(1 + (cls.Mg_in/Kd_mg));
        y2 = 1/(1 + cls.SPM_in/Kd_spm_l);
        
        d_y1 = (IK1_alpha*(1-y1) - IK1_beta*fo**3*y1);

        gK1 = 2.5*(K_out/5.4)**.4 * g_K1 
        I_K1 = gK1*(V-E_K)*(cls.phi*fo*y1 + (1-cls.phi)*y2);

        return [I_K1, y1]


class ExperimentalArtefactsThesis():
    """
    Experimental artefacts from Lei 2020
    For a cell model that includes experimental artefacts, you need to track
    three additional differential parameters: 

    The undetermined variables are: v_off, g_leak, e_leak
    Given the simplified model in section 4c,
    you can make assumptions that allow you to reduce the undetermined
    variables to only:
        v_off_dagger – mostly from liquid-junction potential
        g_leak_dagger
        e_leak_dagger (should be zero)
    """
    def __init__(self, g_leak=1, v_off=-2.8, e_leak=0, r_pipette=2E-3,
                 comp_rs=.8, r_access_star=20E-3,
                 c_m_star=60, tau_clamp=.8E-3, c_p_star=4, tau_z=7.5E-3,
                 tau_sum=40E-3, comp_predrs=None):
        """
        Parameters:
            Experimental measures:
                r_pipette – series resistance of the pipette
                c_m – capacitance of the membrane
            Clamp settings
                alpha – requested proportion of series resistance compensation
        """
        self.g_leak = g_leak
        self.e_leak = e_leak
        self.v_off = v_off
        self.c_p = c_p_star * .95
        self.c_p_star = c_p_star
        self.r_pipette = r_pipette
        self.c_m = c_m_star * .95
        self.r_access = r_access_star * .95
        self.comp_rs = comp_rs # Rs compensation
        self.r_access_star = r_access_star
        self.c_m_star = c_m_star
        self.tau_clamp = tau_clamp
        self.tau_z = tau_z
        self.tau_sum = tau_sum

        if comp_predrs is None:
            self.comp_predrs = comp_rs # Rs prediction
