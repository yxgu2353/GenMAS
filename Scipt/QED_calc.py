import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd

df = pd.read_csv('pubchem_10m_clean.txt')
SMILES_list = df['SMILES'].tolist()
QEDs = []
for smi in SMILES_list:
    mol = Chem.MolFromSmiles(smi)
    qed_value = QED.qed(mol)
    QEDs.append(qed_value)
    print('QED: ', qed_value)

df['QED'] = QEDs
df = pd.DataFrame(df)
df.to_csv('pubchem_10m_QED.csv', index=False)

