import numpy as np
from scipy.integrate import odeint


class EnhancedPBPKModel:
    def __init__(self, physio_params, drug_params):
        self._validate_parameters(physio_params, drug_params)

        self.V = {
            'art': physio_params['V_art'],
            'ven': physio_params['V_ven'],
            'lung': physio_params['V_lung'],
            'liver': physio_params['V_liver'],
            'kidney': physio_params['V_kidney'],
            'other': physio_params['V_other']
        }

        self.Q = {
            'total': physio_params['Q_total'],
            'liver': physio_params['Q_liver'],
            'kidney': physio_params['Q_kidney'],
            'other': physio_params['Q_other']
        }
        self.GFR = physio_params['GFR']


        if drug_params['fu'] * 100 < 0:
            self.fu = 1
        else:
            self.fu = drug_params['fu'] * 100
        self.BP = drug_params['BP']
        self.CL_liv = drug_params['CL']
        self.Kp = drug_params['Kp']

    def _validate_parameters(self, physio, drug):
        sum_Q = physio['Q_liver'] + physio['Q_kidney'] + physio['Q_other']
        assert np.isclose(sum_Q, physio['Q_total'], rtol=0.01), "血流分配异常"

        required_Kp = ['lung', 'liver', 'kidney', 'other']
        assert all(k in drug['Kp'] for k in required_Kp), "缺失Kp值"

    def _ode_system(self, y, t):
        C_ven, C_art, C_lun, C_liv, C_kid, C_rem = y

        dCven_dt = (
                           (self.Q['liver'] / self.Kp['liver']) * C_liv +
                           (self.Q['kidney'] / self.Kp['kidney']) * C_kid +
                           (self.Q['other'] / self.Kp['other']) * C_rem -
                           self.Q['total'] * C_ven
                   ) / self.V['ven']

        dCart_dt = (
                           (self.Q['total'] / self.Kp['lung']) * C_lun -
                           (self.Q['liver'] + self.Q['kidney'] + self.Q['other']) * C_art
                   ) / self.V['art']

        dClun_dt = (
                           self.Q['total'] * C_ven -
                           (self.Q['total'] / self.Kp['lung']) * C_lun
                   ) / self.V['lung']

        dCliv_dt = (
                           self.Q['liver'] * C_art -
                           (self.Q['liver'] / self.Kp['liver']) * C_liv -
                           self.CL_liv * (self.fu / self.Kp['liver']) * (C_liv / self.BP)
                   ) / self.V['liver']

        dCkid_dt = (
                           self.Q['kidney'] * C_art -
                           (self.Q['kidney'] / self.Kp['kidney']) * C_kid -
                           self.GFR * (self.fu / self.Kp['kidney']) * (C_kid / self.BP)
                   ) / self.V['kidney']

        dCrem_dt = (
                           self.Q['other'] * C_art -
                           (self.Q['other'] / self.Kp['other']) * C_rem
                   ) / self.V['other']

        return [dCven_dt, dCart_dt, dClun_dt, dCliv_dt, dCkid_dt, dCrem_dt]

    def simulate(self, dose, t_end=24, n_points=100):
        t = np.linspace(0, t_end, n_points)
        y0 = [dose, 0, 0, 0, 0, 0]
        sol = odeint(self._ode_system, y0, t)
        return t, sol
