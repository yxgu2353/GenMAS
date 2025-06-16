from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MACCSkeys
import pandas as pd
from numpy import array
from pickle import load
import frozen_dir


frozen = frozen_dir.app_path()

def FP_profile(input_data):
    if isinstance(input_data, list):
        input_data = pd.DataFrame(input_data, columns=['SMILES'])
    input_data = input_data.iloc[:, 0]
    fps = []
    for smi in input_data:
        print('Caculating smiles %s'% smi)
        mol = MolFromSmiles(smi)
        if not mol:
            print('Cannot calculate %s' % smi)
            continue
        # MACCS fingerprint
        fingerprint = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        bitarray = [int(bit) for bit in fingerprint]
        fps.append(bitarray)
    print('is ~~~ Finish!')
    return fps

def FP_calc(input_data):
    fps = []
    mol = Chem.MolFromSmiles(input_data)
    fingerprint = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    bitarray = [int(bit) for bit in fingerprint]
    fps.append(bitarray)
    return fps

def cyp3a4i(data):
    fps = FP_calc(data)
    need_valid_data = array(fps)
    Model_dir = frozen + '\\GenMAS_Models\\'
    model_load_CYP3A4i_c = open(f'{Model_dir}' + '3A4i_rf_MACCS.model', 'rb')
    model_3A4i_c = load(model_load_CYP3A4i_c)
    model_load_CYP3A4i_c.close()
    y_pred_3A4i_c = model_3A4i_c.predict(need_valid_data)
    y_pred_score_3A4i_c = []
    for i, pred in enumerate(y_pred_3A4i_c):
        if pred == 1:
            y_pred_score_3A4i_c.append(model_3A4i_c.predict_proba(need_valid_data)[i].reshape(1, -1)[0, 1])
        else:
            y_pred_score_3A4i_c.append(model_3A4i_c.predict_proba(need_valid_data)[i].reshape(1, -1)[0, 0])
    cyp3a4i_result = {'SMILES': data, '3A4i_Prob': y_pred_score_3A4i_c, '3A4i_Label': y_pred_3A4i_c}
    cyp3a4i_result = pd.DataFrame(cyp3a4i_result)
    return cyp3a4i_result

data = 'c1ccccc1'
result = cyp3a4i(data)

