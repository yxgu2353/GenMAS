import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles, rdMolDescriptors, MACCSkeys
from joblib import load
import warnings
import os
from numpy import array
from scipy.integrate import trapz


warnings.filterwarnings('ignore')
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = load(model_file)
    return model
def predict_property(model, fps):
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(fps)
        y_pred_score = [model.predict_proba(fps[i].reshape(1, -1))[0, y] for i, y in enumerate(y_pred)]
        return y_pred, y_pred_score
    else:
        y_pred = model.predict(fps)
        return y_pred, y_pred

def Property_profile(input_data):
    Fps1 = []
    Fps2 = []
    for smi in input_data:
        mol = MolFromSmiles(smi)
        if not mol:
            print(f'Cannot calculate {smi}')
            Fps1.append([0] * 167)
            Fps2.append([0] * 2048)
        else:
            fps1 = MACCSkeys.GenMACCSKeys(mol).ToBitString()
            Fps1.append(list(fps1))
            fps2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
            Fps2.append(list(fps2))
    Fps1 = array(Fps1).astype('int')
    Fps2 = array(Fps2).astype('int')


    Model_dir = os.path.join(os.getcwd(), 'GPhaMAS_models_demo')
    maccs_model_files = {
        'logD': 'logD_rf_MACCS.model',
        'logP': 'logP_rf_MACCS.model',
        'Caco2_r': 'Caco2_r_rf_MACCS.model',
        'HLM': 'HLM_rf_MACCS.model',
        'RLM': 'RLM_rf_MACCS.model',
        '3A4s': '3A4s_rf_MACCS.model',
        '2D6s': '2D6s_rf_MACCS.model',
        'hPPB': 'hPPB_rf_MACCS.model',
        'rPPB': 'rPPB_rf_MACCS.model',
    }
    morgan_model_files = {
        'Caco2': 'Caco2_svm_Morgan.model',
        'DLM': 'DLM_rf_Morgan.model',
        '1A2i': '1A2i_rf_Morgan.model',
        '2C9i': '2C9i_svm_Morgan.model',
        '3A4i': '3A4i_rf_Morgan.model',
        '2C19i': '2C19i_svm_Morgan.model',
        '2D6i': '2D6i_svm_Morgan.model',
        '1A2s': '1A2s_rf_Morgan.model',
        '2C9s': '2C9s_rf_Morgan.model',
        '2C19s': '2C19s_svm_Morgan.model',
        '1A2i_r': '1A2i_r_svm_Morgan.model',
        '2C9i_r': '2C9i_r_svm_Morgan.model',
        '2C19i_r': '2C19i_r_svm_Morgan.model',
        '2D6i_r': '2D6i_r_svm_Morgan.model',
        '3A4i_r': '3A4i_r_svm_Morgan.model',
        'hERG': 'hERG_rf_Morgan.model',
    }

    predictions = {}

    for model_name, model_file in maccs_model_files.items():
        model_path = os.path.join(Model_dir, model_file)
        model = load_model(model_path)

        y_pred, y_pred_score = predict_property(model, Fps1)
        if model_name == 'Caco2_r':
            y_pred = np.power(10, y_pred) * 1e6
            y_pred_score = np.power(10, y_pred_score) * 1e6
            predictions[f'{model_name} (10-6cm/s)'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score

        else:
            predictions[f'{model_name} Label'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score
    for model_name, model_file in morgan_model_files.items():
        model_path = os.path.join(Model_dir, model_file)
        model = load_model(model_path)
        y_pred1, y_pred_score1 = predict_property(model, Fps2)
        if model_name == '1A2i_r':
            y_pred = np.power(10, -y_pred1) * 1e6
            y_pred_score = np.power(10, -y_pred_score1) * 1e6
            predictions[f'{model_name} (uM)'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score
        elif model_name == '2C9i_r':
            y_pred = np.power(10, -y_pred1) * 1e6
            y_pred_score = np.power(10, -y_pred_score1) * 1e6
            predictions[f'{model_name} (uM)'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score
        elif model_name == '2C19i_r':
            y_pred = np.power(10, -y_pred1) * 1e6
            y_pred_score = np.power(10, -y_pred_score1) * 1e6
            predictions[f'{model_name} (uM)'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score
        elif model_name == '2D6i_r':
            y_pred = np.power(10, -y_pred1) * 1e6
            y_pred_score = np.power(10, -y_pred_score1) * 1e6
            predictions[f'{model_name} (uM)'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score
        elif model_name == '3A4i_r':
            y_pred = np.power(10, -y_pred1) * 1e6
            y_pred_score = np.power(10, -y_pred_score1) * 1e6
            predictions[f'{model_name} (uM)'] = y_pred
            predictions[f'{model_name} Prob'] = y_pred_score
        else:
            predictions[f'{model_name} Label'] = y_pred1
            predictions[f'{model_name} Prob'] = y_pred_score1

    results = pd.DataFrame(predictions)
    results.insert(0, 'SMILES', input_data)
    return results

class CompoundPropertiesPredictor:
    def __init__(self, model_paths):
        self.models = {
            'PPB': load(model_paths['PPB']),
            'CL': load(model_paths['CL']),
            'Vd': load(model_paths['Vd']),
            'Kp': load(model_paths['Kp'])
        }

    def smiles_to_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)

    def predict_properties(self, smiles):
        fp = self.smiles_to_fingerprint(smiles).reshape(1, -1)
        ppb = self.models['PPB'].predict(fp)[0]
        fup = 1 - ppb *0.01
        logVd = self.models['Vd'].predict(fp)[0]
        VDss = np.power(10, logVd)
        Kp_value = self.models['Kp'].predict(fp)[0]
        Kp_dict = {
            'liver': Kp_value[0],
            'kidney': Kp_value[1],
            'other': Kp_value[2]
        }

        return {
            'fu': fup,
            'CL': self.models['CL'].predict(fp)[0],
            'Vd': VDss,
            'Kp': Kp_dict
        }

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

    def calculate_pk_parameters(self, t, sol, dose):
        A_blood = sol[:, 0]
        C_blood = A_blood / self.V_blood
        auc = trapz(C_blood, t)
        last_phase = C_blood[-10:]
        if np.all(last_phase > 0):
            log_conc = np.log(last_phase)
            slope, _ = np.polyfit(t[-10:], log_conc, 1)
            kel = -slope
            t_half = np.log(2) / kel if kel > 0 else np.nan
        else:
            t_half = np.nan

        CL = dose / auc if auc > 0 else np.nan
        C0 = C_blood[0] if len(C_blood) > 0 else 0
        Vd = dose / C0 if C0 > 0 else np.nan
        aumc = trapz(t * C_blood, t)
        mrt = aumc / auc if auc > 0 else np.nan

        return {
            'AUC (mg·h/L)': auc,
            't1/2 (h)': t_half,
            'CL (L/h)': CL,
            'Vd (L)': Vd,
            'MRT (h)': mrt
        }

def filter_compounds(df, rules):
    mask = pd.Series([True] * len(df), index=df.index)

    for condition in rules:
        col, operator, value = condition

        try:
            if operator == 'between':
                mask &= df[col].between(value[0], value[1])
            elif operator == '>':
                mask &= (df[col] > value)
            elif operator == '<':
                mask &= (df[col] < value)
            elif operator == '>=':
                mask &= (df[col] >= value)
            elif operator == '<=':
                mask &= (df[col] <= value)
            elif operator == '==':
                mask &= (df[col] == value)
            else:
                raise ValueError(f"Not support: {operator}")
        except KeyError:
            print(f"Warning: column {col} not exist!")
            continue

    return df[mask].reset_index(drop=True)

if __name__ == "__main__":
    physiology_params = {
        'cardiac_output': 14.58,
        'blood_volume': 5.0,
        'liver_volume': 1.8,
        'kidney_volume': 0.3,
        'other_volume': 40.0
    }

    model_paths = {
        'PPB': 'hPPB_rf_Morgan.model',
        'CL': 'CLtot_rf_Morgan.model',
        'Vd': 'VDss_rf_Morgan.model',
        'Kp': 'kp_prediction.model'
    }

    df_test = pd.read_csv('ai_pbpk_test.csv')
    test_compounds = df_test['SMILES'].tolist()
    test_compounds = [
        'CN1C=NC2=C1C=CC=N2',
    ]
    predictor = CompoundPropertiesPredictor(model_paths)
    pk_results = []
    for smiles in test_compounds:
        properties = predictor.predict_properties(smiles)
        properties_df = pd.DataFrame([{
            **properties,
            'Kp_liver': properties['Kp']['liver'],
            'Kp_kidney': properties['Kp']['kidney'],
            'Kp_other': properties['Kp']['other']
        }]).drop('Kp', axis=1)


        model = IVPBPKModel(properties, physiology_params)
        t, concentrations = model.simulate(dose=10, t_end=24, n_points=100)
        pk_param = model.calculate_pk_parameters(t, concentrations, dose=10)
        pk_calc_df = pd.DataFrame([pk_param])
        pk_calc_result = pd.concat([properties_df.reset_index(drop=True),
                                   pk_calc_df.reset_index(drop=True)], axis=1)
        pk_results.append(pk_calc_result)

    df_admet = Property_profile(test_compounds)
    pk_results_df = pd.concat(pk_results, ignore_index=True)


    merged_df = pd.concat([
        df_admet.reset_index(drop=True),
        pk_results_df.reset_index(drop=True)
    ], axis=1)






