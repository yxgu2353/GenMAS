import pandas as pd
import dgl
import torch
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from eval_meter import Meter
from featurizers import CanonicalAtomFeaturizer
from featurizers import CanonicalBondFeaturizer
import os
import random
import numpy as np
from dgllife.utils import one_hot_encoding
from functools import partial
import torch.nn as nn
import model_predictor
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt




if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda' 

else:
    print('use CPU')
    device = 'cpu'

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# 设置全局随机种子



def set_random_seed(seed=16):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_random_seed(16)

def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg = bg.to(device)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

from dgllife.data import MoleculeCSVDataset
atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he', self_loop=True)
def load_data(data,path,load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop='self_loop'),
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=path+ '_graph.bin',
                                 load=load,init_mask=True,n_jobs=16
                                 )

    return dataset

def run_a_train_epoch(n_epochs,epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        bg=bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg, n_feats, e_feats)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())
    total_r2 = np.mean(train_meter.compute_metric('r2'))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training r2 {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_r2,
                                                                                 total_loss))
    return total_r2, total_loss

def run_an_eval_epoch(model, data_loader,loss_criterion):
    model.eval()
    val_losses=[]
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction = model(bg, n_feats, e_feats)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_losses.append(val_loss.data.item())
            eval_meter.update(vali_prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric('r2'))
        total_loss = np.mean(val_losses)
    return total_score, total_loss

def eval(i,dataloader, model,name):
    model.eval()
    meter = Meter()
    smis = []
    labels = []
    preds = []
    for batch_id, batch_data in enumerate(dataloader):
        smiles, bg, label, masks = batch_data
        bg = bg.to(device)
        label = label.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        pred = model(bg, n_feats, e_feats)
        meter.update(pred, label, masks)
        smis.extend(smiles)
        label = label.tolist()
        labels.extend(label)
        preds.extend(pred)

    task_names = ['pKa', 'pKa_acid', 'pKa_base', 'Caco2_r', 'hPPB',
                  'rPPB', 'hCLint', 'MkCLint', 'dCLint', 'rCLint', 'mCLint', 'Vd', 'hCLtot', 'rCLtot',
                  ]

    # task_names = ['logS','pKa']
    R2 = meter.compute_metric('r2')
    print(pathresults + str(i) + str(name) + '_R2:', R2)
    if len(R2) < len(task_names):
        R2.extend([np.nan] * (len(task_names) - len(R2)))
    R2_avg = meter.compute_metric('r2', reduction='mean')
    print(pathresults + str(i) + str(name) + '_R2_avg:', R2_avg)
    MAE = meter.compute_metric('mae')
    if len(MAE) < len(task_names):
        MAE.extend([np.nan] * (len(task_names) - len(MAE)))
    print(pathresults + str(i) + str(name) + '_MAE:', MAE)
    MAE_avg = meter.compute_metric('mae', reduction='mean')
    print(pathresults + str(i) + str(name) + '_MAE_avg:', MAE_avg)
    RMSE = meter.compute_metric('rmse')
    if len(RMSE) < len(task_names):
        RMSE.extend([np.nan] * (len(task_names) - len(RMSE)))
    print(pathresults + str(i) + str(name) + '_RMSE:', RMSE)
    RMSE_avg = meter.compute_metric('rmse', reduction='mean')
    print(pathresults + str(i) + str(name) + '_RMSE_avg:', RMSE_avg)


    results_df = pd.DataFrame({
        'SMILES': smis,
        'True': [str(lst) for lst in labels],
        'Pred': [str(lst) for lst in preds],
    })

    results_df.to_csv(pathresults+str(i)+ str(name) + '_out.csv')

    results_eval = pd.DataFrame({
        'Task': task_names,
        'R2': R2,
        'MAE': MAE,
        'RMSE': RMSE,
    })
    results_eval .to_csv(pathresults + str(i) + str(name) + '_evalue.csv')
    return R2_avg, MAE_avg, RMSE_avg


path='Regress_new/'
pathresults='Regress_new/Results/'
datasets=pd.read_csv(path+'Final_Regress_data.csv')
sfolder = MultilabelStratifiedKFold(n_splits=10,random_state=16,shuffle=True)
X = datasets['SMILES']
y = datasets.drop('SMILES', axis=1)
y = y.fillna(0)
masks = (y != 0).astype(int)

train_r2=[]
train_mae=[]
train_rmse=[]
valid_r2=[]
valid_mae=[]
valid_rmse=[]
for fold_idx, (train_idx, valid_idx) in enumerate(sfolder.split(X, y)):
    print(fold_idx)
    train_sets = datasets.iloc[train_idx]
    train_sets.to_csv(path+str(fold_idx)+'Reg_label_train.csv')
    train_datasets = load_data(train_sets, path + f'{fold_idx}_admet_train', False)
    train_loader = DataLoader(train_datasets, batch_size=1024, shuffle=True,
                              collate_fn=collate_molgraphs)
    valid_sets = datasets.iloc[valid_idx]
    valid_sets.to_csv(path + str(fold_idx) + 'Reg_label_valid.csv')
    valid_datasets = load_data(valid_sets, path + str(fold_idx)+'_admet_valid', False)
    valid_loader = DataLoader(valid_datasets, batch_size=1024, shuffle=True,
                              collate_fn=collate_molgraphs)

    model = model_predictor.ModelPredictor(node_feat_size=atom_featurizer.feat_size('hv'),
                                           edge_feat_size=bond_featurizer.feat_size('he'),
                                           num_layers=2,
                                           num_timesteps=2,
                                           graph_feat_size=256,
                                           predictor_hidden_feats=256,
                                           dropout=0.3,
                                           n_tasks=y.shape[1])

    fn = '/home/guyaxin/DMPK/5_CL_learning/CL_model/model_80.pth'

    state_dict = torch.load(fn, map_location=torch.device('cpu'))
    model.load_my_state_dict(state_dict)
    model = model.to(device)

    #Train
    n_epochs = 501
    #loss_fn =FocalLoss()
    loss_fn = nn.MSELoss(reduction='none')
    lr_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.00001, last_epoch=-1)
    min_score = 0.8
    epochs = []
    scores = []
    val_scores=[]
    losses = []
    val_losses=[]
    for e in range(n_epochs):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
        val_score = run_an_eval_epoch(model, valid_loader,loss_fn)
        epochs.append(e)
        scores.append(score[0])
        val_scores.append(val_score[0])
        losses.append(score[-1])
        val_losses.append(val_score[-1])
        if e % 10 == 0:
            print("第%d个epoch的学习率：%f" % (e + 1, optimizer.param_groups[0]['lr']))
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}'.format(
                e + 1, n_epochs, 'r2', val_score[0], 'loss', val_score[-1]))
        if val_score[0] > min_score:
            torch.save(model.state_dict(), os.path.join(pathresults, str(fold_idx) + 'model_{}.pth'.format(str(e))))
        # if early_stop:
        #     break

    a1 = pd.DataFrame(range(n_epochs), columns=['epoch'])
    a1['lr'] = pd.DataFrame(lr_list)
    a1.to_csv(pathresults + str(fold_idx) + 'lr.csv', index=False)
    plt.plot(range(n_epochs), lr_list, color='r')
    plt.savefig(pathresults + str(fold_idx) + 'lr.png', bbox_inches='tight', dpi=500)

    a = pd.DataFrame(scores, columns=['train_R2'])
    b = pd.DataFrame(val_scores, columns=['validation_R2'])
    c = pd.DataFrame(losses, columns=['train_loss'])
    d = pd.DataFrame(val_losses, columns=['validation_loss'])
    e = pd.concat([a, b, c, d], axis=1)
    e.to_csv(pathresults + str(fold_idx) + 'loss_r2.csv')

    fn = pathresults+str(fold_idx)+'model_class.pt'
    torch.save(model.state_dict(), fn)

    model = model_predictor.ModelPredictor(node_feat_size=atom_featurizer.feat_size('hv'),
                                           edge_feat_size=bond_featurizer.feat_size('he'),
                                           num_layers=2,
                                           num_timesteps=2,
                                           graph_feat_size=256,
                                           predictor_hidden_feats=256,
                                           n_tasks=y.shape[1])


    model.load_state_dict(torch.load(fn,map_location=torch.device('cpu')))
    gcn_net = model.to(device)

    print('Training sets'+str(fold_idx))
    r2_train,mae_train,rmse_train= eval(fold_idx,train_loader,gcn_net,'train')
    train_r2.append(r2_train)
    train_mae.append(mae_train)
    train_rmse.append(rmse_train)

    print('Validation sets'+str(fold_idx))
    r2_valid,mae_valid,rmse_valid = eval(fold_idx,valid_loader, gcn_net,'valid')
    valid_r2.append(r2_valid)
    valid_mae.append(mae_valid)
    valid_rmse.append(rmse_valid)
    
def print_gpu_utilization():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: 已用 {alloc:.2f}GB / 保留 {reserv:.2f}GB")
metric_config = {

    'train': [
        ('R2', train_r2),
        ('MAE', train_mae),
        ('RMSE', train_rmse),
    ],
  
    'valid': [
        ('R2', valid_r2),
        ('MAE', valid_mae),
        ('RMSE', valid_rmse),
    ]
}


dfs = []
for prefix, metrics in metric_config.items():
    for metric_name, data in metrics:
        col_name = f"{prefix}_{metric_name}"
        dfs.append(pd.DataFrame(data, columns=[col_name]))


df = pd.concat(dfs, axis=1)
df.to_csv(f"{pathresults}Mean_results.csv", index=False)
