import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Just get one of the K-fold data
def data_reader(filename):
    data = pd.read_csv(filename,header=None).values
    x = data[1:, 0]
    y = data[1:, 1]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
    for train_index, test_index in cv.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return x_train, y_train, x_test, y_test

# All data for model training included train-data & test-data
All_train_data = data_reader('training_set.csv')

# K-fold split for train-data & test-data (SMILES and label)
train_smi = All_train_data[0]
train_label = All_train_data[1]
test_smi = All_train_data[2]
test_label = All_train_data[3]

#10-fold 9:1 training_dataset & test_dataset
#training_dataset
training_set = pd.DataFrame({'SMILES': train_smi, 'label': train_label})
training_set.to_csv('train_set.csv', index= False, sep=',')
# training_data = pd.read_csv('train_set.csv', encoding='utf-8', sep=',')
print('='*20 + 'Transforming training-data to DGL data form' + '='*20)


# test_dataset
test_set = pd.DataFrame({'SMILES': test_smi, 'label': test_label})
test_set.to_csv('test_set.csv', index= False, sep=',')
# test_data = pd.read_csv('test_set.csv', encoding='utf-8', sep=',')
print('='*20 + 'Transforming test-data to DGL data form' + '='*20)