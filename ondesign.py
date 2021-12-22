import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, root, root_scalar, differential_evolution, LinearConstraint, NonlinearConstraint
import CoolProp.CoolProp as CP

from hx.heat_exchanger import HeatExchanger
from hx.offdesign import OffDesignHX
import multiprocessing as mp


class OnDesignHX(HeatExchanger):
    def __init__(self, hfld, cfld, flow, geom={}, backend='HEOS'):
        super().__init__(hfld, cfld, geom, flow, backend)
        self._solve_vars = []
        self.designed_hx = None

    def solve(self, solve_vars=None, geom=None, bounds=None, constraints=None, const_P=True, **kwargs):
        # Make sure that a variable has been selected to solve for.
        if geom is not None:
            self.geom = geom
            self.channel_type = geom.get('channel type', 'straight')  # use straight channels as default
            self._p = geom.get('p')
            self._t = geom.get('t')
            self._a = geom.get('a')
            self._b = geom.get('b')
            self._x = geom.get('x')
            self._y = geom.get('y')
            self._N = geom.get('N')
            self._L = geom.get('L')
            self.wall_mat = geom.get('wall material')

        if solve_vars is not None:
            if not isinstance(solve_vars, list):
                self._solve_vars = [solve_vars]
            else:
                self._solve_vars = solve_vars
        else:
            raise ValueError
        # else:
        #     for v in ['p', 'e', 'a', 'b', 'x', 'y', 'N', 'L']:
        #         if geom.get(v) is None:
        #             self._solve_vars.append(v)

        if len(self._solve_vars) == 1:
            # Univariate case. There is a unique solution for this, but there is not guarantee a solution exists!
            soln = self._solve_univar(bounds, const_P, **kwargs)
        elif len(self._solve_vars) > 1:
            # Multivariate case. This is phrased as a multivariate global optimization, solved using differential evolution
            soln = self._solve_multivar(bounds, constraints, const_P, **kwargs)
        else:
            # No variable was specified to the solver!
            raise ValueError('No variables to solve for.')

        return soln

    def _solve_univar(self, bounds, const_P, **kwargs):
        hx = OffDesignHX(hfld=self.hfld, cfld=self.cfld, geom=self.get_geom(), flow=self.get_flow())
        var = self._solve_vars[0]
        geom = self.get_geom()
        flow = self.get_flow()

        def _obj(x):
            if var in geom.keys():
                geom[var] = x
                hx.set_geom(geom)
            elif var in flow.keys():
                flow[var] = x
                hx.set_flow(flow)
            else:
                raise ValueError

            try:
                Q_dot = hx.solve(const_P=const_P)
            except ValueError or AssertionError:
                cost = 0
            else:
                cost = -hx.effectiveness

            return cost

        min_res = minimize_scalar(fun=_obj, bounds=bounds, method='bounded')
        assert min_res.success

        self.designed_hx = hx

        return min_res.x, min_res.y

    def _solve_multivar(self, bounds, constraints, const_P, **kwargs):
        # t_af = 2e-3
        t_af = kwargs.get('t_af', 2e-3)
        max_Ac = kwargs.get('max_Ac', 0.05 ** 2)

        hx = OffDesignHX(hfld=self.hfld, cfld=self.cfld, geom=self.get_geom(), flow=self.get_flow())
        geom = self.get_geom()
        geom['x'] = geom['a']  # force x = a
        geom['p'] = geom['a'] + t_af  # didn't add a wall thickness parameter to geom...

        # Force m_dot_c = m_dot_h
        flow = self.get_flow()
        flow['m_dot_h'] = flow['m_dot_c']

        # Set N to be the maximum number of channels allowable for the indicated maximum cross-sectional area
        Ac_unitcell = geom['p'] * (geom['b'] + geom['y'] + 2 * geom['t'])
        geom['N'] = int(max_Ac / Ac_unitcell)

        def _obj(x):
            for var, val in zip(self._solve_vars, x):
                if var in geom.keys():
                    geom[var] = val
                elif var in flow.keys():
                    flow[var] = val
                else:
                    raise KeyError('Key {} found in neither geom nor flow.'.format(var))
            geom['x'] = geom['a']
            geom['p'] = geom['a'] + t_af
            flow['m_dot_h'] = flow['m_dot_c']
            Ac_unitcell = geom['p'] * (geom['b'] + geom['y'] + 2 * geom['t'])
            geom['N'] = int(max_Ac / Ac_unitcell)
            
            hx.set_geom(geom)
            hx.set_flow(flow)
            try:
                Q_dot = hx.solve(const_P=const_P)
            except AssertionError:  # bad geometry
                cost = 1e9
            except ValueError:  # convergence failure
                cost = 1e9
            else:
                cost = -hx.effectiveness
            return cost

        # nonlin_constr1 = NonlinearConstraint(fun=lambda x: x[6] * x[4] * (x[1] + x[3] + x[5]),
        # nonlin_constr1 = NonlinearConstraint(fun=lambda x: x[4] * (x[0] + t_af) * (x[1] + x[2] + 2 * x[3]),  # N * (a + 0.8e-3) * (b + y + t)
        #                                      lb=0,
        #                                      ub=max_Ac)
        # nonlin_constr2 = NonlinearConstraint(fun=lambda x: x[4] - max(x[0], x[2]),
        #                                      lb=0.001,
        #                                      ub=np.inf)

        # min_res = differential_evolution(func=_obj, bounds=bounds, polish=False, disp=True, constraints=[nonlin_constr1, nonlin_constr2])
        # min_res = differential_evolution(func=_obj, bounds=bounds, polish=False, disp=True, constraints=nonlin_constr1)
        min_res = differential_evolution(func=_obj, bounds=bounds, polish=False, disp=True)
        # min_res = differential_evolution(func=self._multivar_obj, bounds=bounds, polish=False, workers=-1, args=(hx, geom, const_P, self._solve_vars, self.Q_dot, lam))
        assert min_res.success

        self.designed_hx = hx

        return min_res.x
