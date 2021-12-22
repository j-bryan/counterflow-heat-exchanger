from abc import abstractmethod
import numpy as np
from scipy.interpolate import interp1d
import CoolProp.CoolProp as CP


class HeatExchanger:
    def __init__(self, hfld, cfld, geom, flow, backend='BICUBIC&HEOS', **kwargs):
        if isinstance(hfld, CP.AbstractState):
            self.hfld = hfld
        else:
            self.hfld = CP.AbstractState(backend, hfld)
        if isinstance(cfld, CP.AbstractState):
            self.cfld = cfld
        else:
            self.cfld = CP.AbstractState(backend, cfld)

        self.geom = geom
        self.flow = flow

        self.Nu = None
        self.f = None

        use_const_kw= kwargs.get('use_const_kw', False)

        self.channel_type = geom.get('channel type')
        self._p = geom.get('p', 0)
        self._t = geom.get('t', 0)
        self._a = geom.get('a', 0)
        self._b = geom.get('b', 0)
        self._x = geom.get('x', 0)
        self._y = geom.get('y', 0)
        self._N = round(geom.get('N', 1))
        self._L = geom.get('L')
        self._eps = geom.get('roughness', 0)
        self.wall_mat = geom.get('wall material')
        self.k_w = self._k_w_generator(use_const=use_const_kw)

        self.T_hi = flow.get('T_hi', 0)
        self.T_ci = flow.get('T_ci', 0)
        self.P_hi = flow.get('P_hi', 0)
        self.P_ci = flow.get('P_ci', 0)
        self.m_dot_h = flow.get('m_dot_h', 0)
        self.m_dot_c = flow.get('m_dot_c', 0)
        self.m_dot_h_chan = self.m_dot_h / self._N
        self.m_dot_c_chan = self.m_dot_c / self._N
        self.Q_dot = flow.get('Q_dot', 0)
        self.Q_dot_chan = flow.get('Q_dot', 0) / self._N

        self.effectiveness = 0

        self.T_h = None
        self.T_c = None
        self.P_h = None
        self.P_c = None
        self.i_h = None
        self.i_c = None

    @abstractmethod
    def solve(self):
        raise NotImplementedError

    def _set_correlations(self, chantype):
        NU_CORRELATIONS = {
            'straight': lambda Re, Pr: 4.089 * (Re < 2300) + 0.023 * Re ** 0.8 * Pr ** 0.4 * (Re >= 2300),
            's-fins': lambda Re, Pr: 0.1740 * Re ** 0.593 * Pr ** 0.430,
            'zigzag': lambda Re, Pr: (4.089 + 0.00365 * Re * Pr ** 0.58) * (
                        Re < 2300) + 0.1696 * Re ** 0.629 * Pr ** 0.317 * (Re >= 2300),
            'airfoil': lambda Re, Pr: (3.7 + 0.0013 * Re ** 1.12 * Pr ** 0.38) * (Re < 2300) + (
                    0.027 * Re ** 0.78 * Pr ** 0.4) * (Re >= 2300)
        }

        F_CORRELATIONS = {
            'straight': lambda Re, e, D: 15.78 / Re * (Re < 2300) + self._alashkar(Re, e, D) * (Re >= 2300),
            # 's-fins': lambda Re, e, D: np.nan * (Re < 2300) + (0.4545 * Re ** -0.34) * (Re >= 2300),
            's-fins': lambda Re, e, D: 0.4545 * Re ** -0.34,
            'zigzag': lambda Re, e, D: (15.78 + 0.004868 * Re ** 0.8416) / Re * (Re < 2300) + (
                        0.1942 * Re ** -0.091) * (Re >= 2300),
            'airfoil': lambda Re, e, D: (9.31 + 0.028 * Re ** 0.86) / Re
        }
        self.Nu = NU_CORRELATIONS.get(chantype)
        self.f = F_CORRELATIONS.get(chantype)

    @staticmethod
    def _alashkar(Re, e, D):
        """ Alashkar's (2012) explicit approximation of the Colebrook equation. """
        A = e / D / 3.7065
        B = 2.5226 / Re
        return 1.325474505 * np.log(
            A - 0.8686068432 * B * np.log(A - 0.8784893582 * B * np.log(A + (1.664368035 * B) ** 0.8373492157))) ** (-2)

    @staticmethod
    def _k_w_generator(use_const=False):
        # Linear interpolation of thermal conductivity data for 
        T_inconel = np.array([23., 100, 200, 300, 400, 500, 600]) + 273.15
        k_inconel = np.array([9.8, 11.4, 13.4, 15.5, 17.6, 19.6, 21.3])
        T_ss = np.array([100., 500]) + 273.15
        k_ss = np.array([16.2, 21.4])

        f_inconel = interp1d(T_inconel, k_inconel, kind='linear', bounds_error=False, fill_value='extrapolate')
        f_ss = interp1d(T_ss, k_ss, kind='linear', bounds_error=False, fill_value='extrapolate')

        if use_const:
            def k_w(T, T_switch=673.15):
                if isinstance(T, np.ndarray):
                    return 16.27 * np.ones(T.shape)
                    # return 18.75 * np.ones(T.shape)
                else:
                    return 16.27
                    # return 18.75
        else:
            def k_w(T, T_switch=673.15):
                return f_ss(T) * (T < T_switch) + f_inconel(T) * (T >= T_switch)
            
        return k_w

    @property
    def channel_type(self):
        return self._channel_type

    @channel_type.setter
    def channel_type(self, val):
        self._channel_type = val
        self._set_correlations(val)

    @property
    def wall_mat(self):
        return self._wall_mat

    @wall_mat.setter
    def wall_mat(self, val):
        self._wall_mat = val

    def set_geom(self, geom):
        self.channel_type = geom.get('channel type')
        self._p = geom.get('p')
        self._t = geom.get('t')
        self._a = geom.get('a')
        self._b = geom.get('b')
        self._x = geom.get('x')
        self._y = geom.get('y')
        self._N = round(geom.get('N'))
        self._L = geom.get('L')
        self._eps = geom.get('roughness', 0)
        self.wall_mat = geom.get('wall material')
        self.m_dot_c_chan = self.m_dot_c / self._N
        self.m_dot_h_chan = self.m_dot_h / self._N

    def get_geom(self):
        geom = {
            'p': self._p,
            't': self._t,
            'a': self._a,
            'b': self._b,
            'x': self._x,
            'y': self._y,
            'N': self._N,
            'L': self._L,
            'roughness': self._eps,
            'wall material': self._wall_mat,
            'channel type': self._channel_type,
        }
        return geom

    def set_flow(self, flow):
        self.T_hi = flow.get('T_hi')
        self.T_ci = flow.get('T_ci')
        self.P_hi = flow.get('P_hi')
        self.P_ci = flow.get('P_ci')
        self.m_dot_h = flow.get('m_dot_h')
        self.m_dot_h_chan = self.m_dot_h / self._N
        self.m_dot_c = flow.get('m_dot_c')
        self.m_dot_c_chan = self.m_dot_c / self._N
        self.Q_dot = flow.get('Q_dot')

    def get_flow(self):
        flow = {
            'T_hi': self.T_hi,
            'T_ci': self.T_ci,
            'P_hi': self.P_hi,
            'P_ci': self.P_ci,
            'm_dot_h': self.m_dot_h,
            'm_dot_c': self.m_dot_c,
            'Q_dot': self.Q_dot,
        }
        return flow
