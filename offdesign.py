import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

from hx.heat_exchanger import HeatExchanger


R_conv_h = []
R_conv_c = []
R_cond = []


class OffDesignHX(HeatExchanger):
    def __init__(self, hfld, cfld, geom, flow={}, backend='HEOS', **kwargs):
        super().__init__(hfld, cfld, geom, flow, backend, **kwargs)
        self.const_P = False

    def solve(self, flow=None, show_profiles=False, const_P=False):
        self.const_P = const_P

        if flow is not None:
            self.T_hi = flow.get('T_hi', 0)
            self.T_ci = flow.get('T_ci', 0)
            self.P_hi = flow.get('P_hi', 0)
            self.P_ci = flow.get('P_ci', 0)
            self.m_dot_h = flow.get('m_dot_h', 0)
            self.m_dot_c = flow.get('m_dot_c', 0)

        self.cfld.update(CP.PT_INPUTS, self.P_ci, self.T_ci)
        i_ci = self.cfld.hmass()
        self.hfld.update(CP.PT_INPUTS, self.P_hi, self.T_hi)
        i_hi = self.hfld.hmass()

        C_min = min(self.m_dot_c * self.cfld.cpmass(), self.m_dot_h * self.hfld.cpmass())

        self.m_dot_c_chan = self.m_dot_c / self._N  # per-channel mass flow rate
        self.m_dot_h_chan = self.m_dot_h / self._N  # Note that CV has half-channels!

        if self.const_P:
            # Bracketing enthalpies based on specified inlet conditions
            self.cfld.update(CP.PT_INPUTS, self.P_ci, self.T_hi)
            iu = self.cfld.hmass()
            self.cfld.update(CP.PT_INPUTS, self.P_ci, self.T_ci)
            il = self.cfld.hmass()

            # Everything is solved for a single channel using the shooting method, then scaled to N channels.
            root_res = root_scalar(f=self._opt_obj,
                                   bracket=(il, iu),
                                   method='brentq',
                                   args=(i_ci, i_hi, self.P_ci, self.P_hi))
            if not root_res.converged:
                raise ValueError('root finding did not converge')
            i_co = root_res.root
            P_co = self.P_ci
        else:
            # Cold outlet enthalpy and pressure initial guesses
            self.cfld.update(CP.PT_INPUTS, self.P_ci, self.T_hi)
            i_co0 = self.cfld.hmass()
            P_co0 = self.P_ci

            # Everything is solved for a single channel using the shooting method, then scaled to N channels.
            opt_res = minimize(fun=self._opt_obj,
                               x0=np.array([i_co0, P_co0]),
                               method='Nelder-Mead',
                               # method='BFGS',
                               tol=1e-6,
                               args=(i_ci, i_hi, self.P_ci, self.P_hi))
            i_co, P_co = opt_res.x

            if not opt_res.success:
                raise ValueError('Solution failed to converge. {}'.format(opt_res.fun))

        x = np.linspace(0, self._L, 1000)
        self.calc_profiles(i_hi, i_co, self.P_hi, P_co, x)
        x *= 100

        if show_profiles and not self.const_P:  # Whether or not to calculate and show the temperature profiles of the heat exchanger
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            ch = 'red'
            cw = 'green'
            cc = 'black'

            T_w = [self.avg_wall_temp(self.i_h[i], self.i_c[i], self.P_h[i], self.P_c[i]) - 273.15 for i in range(len(x))]

            ax1.plot(x, self.T_h[::-1], color=ch, label=r'$T_h$')
            ax1.plot(x, T_w[::-1], color=cw, label=r'$T_w$')
            ax1.plot(x, self.T_c[::-1], color=cc, label=r'$T_c$')
            ax1.set_ylabel('T (C)')
            ax1.legend()

            ax2.plot(x, (self.P_h - self.P_h[0])[::-1], color=ch, label=r'$\Delta P_h$')
            ax2.plot(x, (self.P_c - self.P_c[-1])[::-1], color=cc, label=r'$\Delta P_c$')
            ax2.set_ylabel('Pressure Change (Pa)')
            ax2.set_xlabel('Position (cm)')
            ax2.legend()

            # plt.savefig('./figs/{}.png'.format(self.channel_type), dpi=150)
            # plt.clf()
            plt.show()
        elif show_profiles and self.const_P:
            fig, ax1 = plt.subplots()
            ch = 'red'
            cc = 'blue'

            ax1.plot(x, self.T_h[::-1], color=ch)
            ax1.plot(x, self.T_c[::-1], color=cc)
            ax1.set_ylabel('T [C]')

            # plt.savefig('./figs/{}.png'.format(self.channel_type), dpi=150)
            # plt.clf()
            plt.show()

        Q_dot = self.m_dot_c * (i_co - i_ci)
        self.Q_dot = Q_dot
        self.Q_dot_chan = self.Q_dot / self._N

        Q_dot_max_c = self.m_dot_c * (CP.PropsSI('H', 'P', self.P_ci, 'T', self.T_hi, 'CO2') - CP.PropsSI('H', 'P', self.P_ci, 'T', self.T_ci, 'CO2'))
        Q_dot_max_h = self.m_dot_h * (CP.PropsSI('H', 'P', self.P_hi, 'T', self.T_hi, 'CO2') - CP.PropsSI('H', 'P', self.P_hi, 'T', self.T_ci, 'CO2'))
        Q_dot_max = min(Q_dot_max_c, Q_dot_max_h)

        # if self.channel_type == 'straight' or self.channel_type == 'zigzag':
        #     A_c = self._p * (2 * self._t + self._b + self._y) - (self._a * self._b + self._x * self._y)  # include 'fin' walls
        # else:  # geometry is s-fins or airfoils
        #     A_c = self._p * 2 * self._t  # don't include fins

        A_c1 = self._p * (2 * self._t + self._b + self._y) - (self._a * self._b + self._x * self._y)
        A_c2 = self._p * 2 * self._t
        A_c_eff = 0.5 * (A_c1 + A_c2)
        # A_c_eff = A_c2

        R_ac = self._L / (max(self.k_w(self.T_h)) * A_c_eff)
        C_C = np.mean(self.m_dot_c_chan * CP.PropsSI('CPMASS', 'P', self.P_c, 'H', self.i_c, 'CO2').ravel())
        C_H = np.mean(self.m_dot_h_chan * CP.PropsSI('CPMASS', 'P', self.P_h, 'H', self.i_h, 'CO2').ravel())
        C_min = min([C_C, C_H])
        lam = 1 / (R_ac * C_min)

        # self.effectiveness = Q_dot / Q_dot_max

        self.effectiveness = Q_dot / Q_dot_max - lam
        self.Q_dot = self.effectiveness * Q_dot_max

        # Hydraulic diameters
        D_h = 2 * self._a * self._b / (self._a + self._b)
        D_c = 2 * self._x * self._y / (self._x + self._y)

        # Reynolds number and Prandtl number
        Re_h = 4 * self.m_dot_h_chan / (np.pi * D_h * CP.PropsSI('VISCOSITY', 'H', self.i_h, 'P', self.P_h,'CO2').mean())
        Re_c = 4 * self.m_dot_c_chan / (np.pi * D_c * CP.PropsSI('VISCOSITY', 'H', self.i_c, 'P', self.P_c,'CO2').mean())

        # R_conv_h, R_conv_c, R_cond = self._thermal_resistances2(self.T_h, self.T_c, self.P_h, self.P_c)
        # R_tot = R_conv_h + R_conv_c + R_cond
        # print(np.mean(R_conv_h / R_tot), np.mean(R_conv_c / R_tot), np.mean(R_cond / R_tot))

        return self.Q_dot

    def _opt_obj(self, x, i_ci, i_hi, P_ci, P_hi):
        # Solve IVP with i_hi (known), i_co (guess), P_hi (known), P_co (guess) as initial conditions
        if self.const_P:
            i_co = x
            P_co = P_ci
        else:
            i_co, P_co = x

        ivp_res = solve_ivp(fun=self._ode_obj,
                            t_span=(0, self._L),
                            y0=np.array([i_hi, i_co, P_hi, P_co]),
                            method='RK45',
                            max_step=self._L/20)
        i_ho, i_ci_ivp, P_ho, P_ci_ivp = ivp_res.y.T[-1]

        # Update objective function value (sum of squared relative error) based on accuracy of i_ci and P_ci
        if self.const_P:
            cost = i_ci_ivp - i_ci
        else:
            cost = (((i_ci_ivp - i_ci) / i_ci) ** 2 + ((P_ci_ivp - P_ci) / P_ci) ** 2) ** 0.5

        return cost

    def _ode_obj(self, t, x):
        i_H, i_C, P_H, P_C = x

        # Get properties we'll need
        try:
            self.hfld.update(CP.HmassP_INPUTS, i_H, P_H)
            self.cfld.update(CP.HmassP_INPUTS, i_C, P_C)
        except ValueError as ve:
            return np.array([0., 0., 0., 0.])

        T_h = self.hfld.T()
        mu_h = self.hfld.viscosity()
        rho_h = self.hfld.rhomass()
        k_h = self.hfld.conductivity()
        Pr_h = self.hfld.Prandtl()

        T_c = self.cfld.T()
        mu_c = self.cfld.viscosity()
        rho_c = self.cfld.rhomass()
        k_c = self.cfld.conductivity()
        Pr_c = self.cfld.Prandtl()

        # Hydraulic diameter of each channel
        D_h = 2 * self._a * self._b / (self._a + self._b)
        D_c = 2 * self._x * self._y / (self._x + self._y)

        # Reynolds number and Prandtl number
        Re_h = 4 * self.m_dot_h_chan / (np.pi * D_h * mu_h)
        Re_c = 4 * self.m_dot_c_chan / (np.pi * D_c * mu_c)

        # Calculate heat transfer rate and pressure drop rate
        dqdx_tube = self._heat_transfer_rate(Re_h, Re_c, Pr_h, Pr_c, k_h, k_c, D_h, D_c, T_h, T_c)
        di_h = -dqdx_tube / (0.5 * self.m_dot_h_chan)
        di_c = -dqdx_tube / (0.5 * self.m_dot_c_chan)

        if self.const_P:
            dP_h, dP_c = 0, 0
        else:
            dP_h, dP_c = self._pressure_drop_rate(Re_h, Re_c, D_h, D_c, rho_h, rho_c)

        return np.array([di_h, di_c, dP_h, dP_c])

    def _pressure_drop_rate(self, Re_h, Re_c, D_h, D_c, rho_h, rho_c):
        # Calculate Fanning factor from correlation
        f_h = self.f(Re_h, e=self._eps, D=D_h)
        f_c = self.f(Re_c, e=self._eps, D=D_c)

        # Pressure loss rate from Fanning's equation (see also Darcy-Weisbach equation)
        dP_h = -2 * f_h / D_h * rho_h * (self.m_dot_h_chan / (rho_h * self._a * self._b)) ** 2
        dP_c = 2 * f_c / D_c * rho_c * (self.m_dot_c_chan / (rho_c * self._x * self._y)) ** 2

        return dP_h, dP_c

    def _heat_transfer_rate(self, Re_h, Re_c, Pr_h, Pr_c, k_h, k_c, D_h, D_c, T_h, T_c):
        """
        Solves for the convection coefficients of a heat exchanger.
        """
        # Heat transfer coefficient from Nusselt number correlations
        h_h = self.Nu(Re_h, Pr_h) * k_h / D_h
        h_c = self.Nu(Re_c, Pr_c) * k_c / D_c

        # Hot side convective thermal resistance
        m_h = (2 * h_h / (self.k_w(T_h) * (self._p - self._a))) ** 0.5
        L_h = self._b / 2
        A_tot_h = self._a + self._b
        A_fin_h = self._b
        etaf_h = np.tanh(m_h * L_h) / (m_h * L_h)
        eta0_h = 1 - A_fin_h / A_tot_h * (1 - etaf_h)
        R_conv_h_prime = 1 / (eta0_h * h_h * A_tot_h)

        # Cold side convective thermal resistance
        m_c = (2 * h_c / (self.k_w(T_h) * (self._p - self._x))) ** 0.5
        L_c = self._y / 2
        A_tot_c = self._x + self._y
        A_fin_c = self._y
        etaf_c = np.tanh(m_c * L_c) / (m_c * L_c)
        eta0_c = 1 - A_fin_c / A_tot_c * (1 - etaf_c)
        R_conv_c_prime = 1 / (eta0_c * h_c * A_tot_c)

        # Conductive resistance
        R_cond_prime = self._t / (self.k_w(T_h) * 0.5 * (self._x + self._a))  # uses average channel width

        # Total thermal resistance
        R_tot_prime = R_conv_h_prime + R_cond_prime + R_conv_c_prime

        dqdx = (T_h - T_c) / R_tot_prime

        return dqdx

    def _thermal_resistances(self, i_H, i_C, P_H, P_C):
        # Get properties we'll need
        try:
            self.hfld.update(CP.HmassP_INPUTS, i_H, P_H)
            self.cfld.update(CP.HmassP_INPUTS, i_C, P_C)
        except ValueError as ve:
            return np.array([0., 0., 0., 0.])

        T_h = self.hfld.T()
        mu_h = self.hfld.viscosity()
        rho_h = self.hfld.rhomass()
        k_h = self.hfld.conductivity()
        Pr_h = self.hfld.Prandtl()

        T_c = self.cfld.T()
        mu_c = self.cfld.viscosity()
        rho_c = self.cfld.rhomass()
        k_c = self.cfld.conductivity()
        Pr_c = self.cfld.Prandtl()

        # Hydraulic diameter of each channel
        D_h = 2 * self._a * self._b / (self._a + self._b)
        D_c = 2 * self._x * self._y / (self._x + self._y)

        # Reynolds number and Prandtl number
        Re_h = 4 * self.m_dot_h_chan / (np.pi * D_h * mu_h)
        Re_c = 4 * self.m_dot_c_chan / (np.pi * D_c * mu_c)

        # Heat transfer coefficient from Nusselt number correlations
        h_h = self.Nu(Re_h, Pr_h) * k_h / D_h
        h_c = self.Nu(Re_c, Pr_c) * k_c / D_c

        # Hot side convective thermal resistance
        m_h = (2 * h_h / (self.k_w(T_h) * (self._p - self._a))) ** 0.5
        L_h = self._b / 2
        A_tot_h = self._a + self._b
        A_fin_h = self._b
        etaf_h = np.tanh(m_h * L_h) / (m_h * L_h)
        eta0_h = 1 - A_fin_h / A_tot_h * (1 - etaf_h)
        R_conv_h_prime = 1 / (eta0_h * h_h * A_tot_h)

        # Cold side convective thermal resistance
        m_c = (2 * h_c / (self.k_w(T_h) * (self._p - self._x))) ** 0.5
        L_c = self._y / 2
        A_tot_c = self._x + self._y
        A_fin_c = self._y
        etaf_c = np.tanh(m_c * L_c) / (m_c * L_c)
        eta0_c = 1 - A_fin_c / A_tot_c * (1 - etaf_c)
        R_conv_c_prime = 1 / (eta0_c * h_c * A_tot_c)

        # Conductive resistance
        R_cond_prime = self._t / (self.k_w(T_h) * 0.5 * (self._x + self._a))  # uses average channel width

        return R_conv_h_prime, R_conv_c_prime, R_cond_prime

    def avg_wall_temp(self, i_H, i_C, P_H, P_C):
        # Get properties we'll need
        try:
            self.hfld.update(CP.HmassP_INPUTS, i_H, P_H)
            self.cfld.update(CP.HmassP_INPUTS, i_C, P_C)
        except ValueError as ve:
            return np.array([0., 0., 0., 0.])

        T_h = self.hfld.T()
        mu_h = self.hfld.viscosity()
        rho_h = self.hfld.rhomass()
        k_h = self.hfld.conductivity()
        Pr_h = self.hfld.Prandtl()

        T_c = self.cfld.T()
        mu_c = self.cfld.viscosity()
        rho_c = self.cfld.rhomass()
        k_c = self.cfld.conductivity()
        Pr_c = self.cfld.Prandtl()

        # Hydraulic diameter of each channel
        D_h = 2 * self._a * self._b / (self._a + self._b)
        D_c = 2 * self._x * self._y / (self._x + self._y)

        # Reynolds number and Prandtl number
        Re_h = 4 * self.m_dot_h_chan / (np.pi * D_h * mu_h)
        Re_c = 4 * self.m_dot_c_chan / (np.pi * D_c * mu_c)

        # Heat transfer coefficient from Nusselt number correlations
        h_h = self.Nu(Re_h, Pr_h) * k_h / D_h
        h_c = self.Nu(Re_c, Pr_c) * k_c / D_c

        # Hot side convective thermal resistance
        m_h = (2 * h_h / (self.k_w(T_h) * (self._p - self._a))) ** 0.5
        L_h = self._b / 2
        A_tot_h = self._a + self._b
        A_fin_h = self._b
        etaf_h = np.tanh(m_h * L_h) / (m_h * L_h)
        eta0_h = 1 - A_fin_h / A_tot_h * (1 - etaf_h)
        R_conv_h_prime = 1 / (eta0_h * h_h * A_tot_h)

        # Cold side convective thermal resistance
        m_c = (2 * h_c / (self.k_w(T_h) * (self._p - self._x))) ** 0.5
        L_c = self._y / 2
        A_tot_c = self._x + self._y
        A_fin_c = self._y
        etaf_c = np.tanh(m_c * L_c) / (m_c * L_c)
        eta0_c = 1 - A_fin_c / A_tot_c * (1 - etaf_c)
        R_conv_c_prime = 1 / (eta0_c * h_c * A_tot_c)

        # Conductive resistance
        R_cond_prime = self._t / (self.k_w(T_h) * 0.5 * (self._x + self._a))  # uses average channel width

        dqdx = (T_h - T_c) / (R_conv_c_prime + R_conv_h_prime + R_cond_prime)

        T_wh = T_h - dqdx * R_conv_h_prime
        T_wc = T_c + dqdx * R_conv_c_prime

        return 0.5 * (T_wh + T_wc)

    def heat_flux(self, i_H, i_C, P_H, P_C):
        try:
            self.hfld.update(CP.HmassP_INPUTS, i_H, P_H)
            self.cfld.update(CP.HmassP_INPUTS, i_C, P_C)
        except ValueError as ve:
            return np.array([0., 0., 0., 0.])

        T_h = self.hfld.T()
        mu_h = self.hfld.viscosity()
        rho_h = self.hfld.rhomass()
        k_h = self.hfld.conductivity()
        Pr_h = self.hfld.Prandtl()

        T_c = self.cfld.T()
        mu_c = self.cfld.viscosity()
        rho_c = self.cfld.rhomass()
        k_c = self.cfld.conductivity()
        Pr_c = self.cfld.Prandtl()

        # Hydraulic diameter of each channel
        D_h = 2 * self._a * self._b / (self._a + self._b)
        D_c = 2 * self._x * self._y / (self._x + self._y)

        # Reynolds number and Prandtl number
        Re_h = 4 * self.m_dot_h_chan / (np.pi * D_h * mu_h)
        Re_c = 4 * self.m_dot_c_chan / (np.pi * D_c * mu_c)

        # Heat transfer coefficient from Nusselt number correlations
        h_h = self.Nu(Re_h, Pr_h) * k_h / D_h
        h_c = self.Nu(Re_c, Pr_c) * k_c / D_c

        # Hot side convective thermal resistance
        m_h = (2 * h_h / (self.k_w(T_h) * (self._p - self._a))) ** 0.5
        L_h = self._b / 2
        A_tot_h = self._a + self._b  # actually is a half-perimeter
        A_fin_h = self._b  # two half-sized fin lengths combined
        etaf_h = np.tanh(m_h * L_h) / (m_h * L_h)
        eta0_h = 1 - A_fin_h / A_tot_h * (1 - etaf_h)
        R_conv_h_prime = 1 / (eta0_h * h_h * A_tot_h)

        # Cold side convective thermal resistance
        m_c = (2 * h_c / (self.k_w(T_h) * (self._p - self._x))) ** 0.5
        L_c = self._y / 2
        A_tot_c = self._x + self._y
        A_fin_c = self._y
        etaf_c = np.tanh(m_c * L_c) / (m_c * L_c)
        eta0_c = 1 - A_fin_c / A_tot_c * (1 - etaf_c)
        R_conv_c_prime = 1 / (eta0_c * h_c * A_tot_c)

        # Conductive resistance
        R_cond_prime = self._t / (self.k_w(T_h) * 0.5 * (self._x + self._a))  # uses average channel width

        # Total thermal resistance
        R_tot_prime = R_conv_h_prime + R_cond_prime + R_conv_c_prime

        dqdx = (T_h - T_c) / R_tot_prime

        return dqdx

    def calc_profiles(self, i_hi, i_co, P_hi, P_co, x=None):
        ivp_res = solve_ivp(fun=self._ode_obj,
                            t_span=(0, self._L),
                            y0=np.array([i_hi, i_co, P_hi, P_co]),
                            method='RK45',
                            max_step=self._L / 20,
                            dense_output=True)
        oderesult = ivp_res.sol

        if x is None:
            x = np.linspace(0, self._L, 100)

        i_h, i_c, P_h, P_c = oderesult(x)
        T_h = CP.PropsSI('T', 'H', i_h, 'P', P_h, self.hfld.fluid_names()[0]) - 273.15
        T_c = CP.PropsSI('T', 'H', i_c, 'P', P_c, self.cfld.fluid_names()[0]) - 273.15

        # R_conv_h = np.zeros(x.shape)
        # R_conv_c = np.zeros(x.shape)
        # R_cond = np.zeros(x.shape)
        #
        # for i in range(len(x)):
        #     R_conv_h[i], R_conv_c[i], R_cond[i] = self._thermal_resistances(i_h[i], i_c[i], P_h[i], P_c[i])
        #
        # R_tot = R_conv_h + R_conv_c + R_cond

        self.T_h = T_h
        self.T_c = T_c
        self.P_h = P_h
        self.P_c = P_c
        self.i_h = i_h
        self.i_c = i_c

        # plt.plot(x * 100, R_conv_h, label='R_conv_h')
        # plt.plot(x * 100, R_conv_c, label='R_conv_c')
        # plt.plot(x * 100, R_cond, label='R_cond')
        # plt.xlabel('x (cm)')
        # plt.ylabel('Thermal Resistance (K/W*m)')
        # plt.legend()
        # plt.savefig('./figs/thermal_resistances/tr_abs.png', dpi=150)

    def _thermal_resistances2(self, T_h, T_c, P_h, P_c):
        properties = ['VISCOSITY', 'D', 'CONDUCTIVITY', 'PRANDTL']
        mu_h, rho_h, k_h, Pr_h = CP.PropsSI(properties, 'P', P_h, 'T', T_h, 'CO2').T
        mu_c, rho_c, k_c, Pr_c = CP.PropsSI(properties, 'P', P_c, 'T', T_c, 'CO2').T

        # Hydraulic diameter of each channel
        D_h = 2 * self._a * self._b / (self._a + self._b)
        D_c = 2 * self._x * self._y / (self._x + self._y)

        # Reynolds number and Prandtl number
        Re_h = 4 * self.m_dot_h_chan / (np.pi * D_h * mu_h)
        Re_c = 4 * self.m_dot_c_chan / (np.pi * D_c * mu_c)

        # Heat transfer coefficient from Nusselt number correlations
        h_h = self.Nu(Re_h, Pr_h) * k_h / D_h
        h_c = self.Nu(Re_c, Pr_c) * k_c / D_c

        # Hot side convective thermal resistance
        m_h = (2 * h_h / (self.k_w(T_h) * (self._p - self._a))) ** 0.5
        L_h = self._b / 2
        A_tot_h = self._a + self._b
        A_fin_h = self._b
        etaf_h = np.tanh(m_h * L_h) / (m_h * L_h)
        eta0_h = 1 - A_fin_h / A_tot_h * (1 - etaf_h)
        R_conv_h_prime = 1 / (eta0_h * h_h * A_tot_h)

        # Cold side convective thermal resistance
        m_c = (2 * h_c / (self.k_w(T_h) * (self._p - self._x))) ** 0.5
        L_c = self._y / 2
        A_tot_c = self._x + self._y
        A_fin_c = self._y
        etaf_c = np.tanh(m_c * L_c) / (m_c * L_c)
        eta0_c = 1 - A_fin_c / A_tot_c * (1 - etaf_c)
        R_conv_c_prime = 1 / (eta0_c * h_c * A_tot_c)

        # Conductive resistance
        R_cond_prime = self._t / (self.k_w(T_h) * 0.5 * (self._x + self._a))  # uses average channel width

        return R_conv_h_prime, R_conv_c_prime, R_cond_prime
