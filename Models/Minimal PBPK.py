import numpy as np
from scipy.integrate import odeint

class IVPBPKModel:
    def __init__(self, params, physiology_params):
        self.Qc = physiology_params['cardiac_output']
        self.V_blood = physiology_params['blood_volume']
        self.V_liver = physiology_params['liver_volume']
        self.V_kidney = physiology_params['kidney_volume']
        self.V_other = physiology_params['other_volume']

        self.fu = params['fu']
        self.CL_total = params['CL']
        self.Kp = params['Kp']


        self.Q_liver = 0.25 * self.Qc
        self.Q_kidney = 0.19 * self.Qc
        self.Q_other = self.Qc - self.Q_liver - self.Q_kidney

    def differential_eq(self, y, t):
        A_blood, A_liver, A_kidney, A_other = y
        C_blood = A_blood / self.V_blood
        C_liver = A_liver / self.V_liver
        C_kidney = A_kidney / self.V_kidney
        C_other = A_other / self.V_other


        CL_hepatic = self.fu * self.CL_total * 0.8
        CL_renal = self.fu * self.CL_total * 0.2

        dA_blood = (
            self.Q_liver * (C_liver / self.Kp['liver']) +
            self.Q_kidney * (C_kidney / self.Kp['kidney']) +
            self.Q_other * (C_other / self.Kp['other']) -
            self.Q_liver * C_blood -
            self.Q_kidney * C_blood -
            self.Q_other * C_blood
        )

        dA_liver = self.Q_liver * (C_blood - C_liver / self.Kp['liver']) - CL_hepatic * C_liver
        dA_kidney = self.Q_kidney * (C_blood - C_kidney / self.Kp['kidney']) - CL_renal * C_kidney
        dA_other = self.Q_other * (C_blood - C_other / self.Kp['other'])

        return [dA_blood, dA_liver, dA_kidney, dA_other]


    def simulate(self, dose, t_end=24, n_points=100):
        t = np.linspace(0, t_end, n_points)
        y0 = [dose, 0, 0, 0]
        sol = odeint(self.differential_eq, y0, t)
        return t, sol