import pandas as pd
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping
from eval_meter import Meter
from featurizers import CanonicalAtomFeaturizer
from featurizers import CanonicalBondFeaturizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import model_predictor
import attentivefp_encoder
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import warnings

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

from dgllife.utils import one_hot_encoding
from functools import partial


warnings.filterwarnings('ignore')

seed = 16
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)           
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)

def set_random_seed(seed=16):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    total_auc = np.mean(train_meter.compute_metric('roc_auc_score'))
    total_acc = np.mean(train_meter.compute_metric('accuracy'))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training_auc {:.4f}, training_acc {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_auc,total_acc,
                                                                                  total_loss))
    return total_auc, total_acc, total_loss

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
        total_auc = np.mean(eval_meter.compute_metric('roc_auc_score'))
        total_acc = np.mean(eval_meter.compute_metric('accuracy'))
        total_loss = np.mean(val_losses)
    return total_auc, total_acc, total_loss

def eval(i,dataloader, model,name):
    model.eval()
    meter = Meter()
    smis = []
    model_ids=[]
    labels = []
    preds = []
    probs = []
    for batch_id, batch_data in enumerate(dataloader):
        smiles, bg, label, masks = batch_data
        bg = bg.to(device)
        label = label.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        logit = model(bg, n_feats, e_feats)
        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).float()
        meter.update(logit, label, masks)
        smis.extend(smiles)
        model_ids.extend(model_ids)
        label = label.tolist()
        labels.extend(label)
        preds.extend(pred)
        prob = prob.tolist()
        probs.extend(prob)

    task_names = ['Caco2', 'CYP1A2i_c', 'CYP2C9i_c', 'CYP2C19i_c',
                  'CYP3A4i_c', 'CYP2D6i_c', 'CYP1A2s', 'CYP2C9s',
                  'CYP2C19s', 'CYP2D6s', 'CYP3A4s', 'HLM', 'DLM', 'RLM', 'hERG']
    auc = meter.compute_metric('roc_auc_score')
    auc_avg = meter.compute_metric('roc_auc_score', reduction='mean')
    if len(auc) < len(task_names):
        auc.extend([np.nan] * (len(task_names) - len(auc)))
    print(pathresults+str(i)+str(name) + '_AUC:', auc)
    print(pathresults + str(i) + str(name) + '_AUC_avg:', auc_avg)
    auprc = meter.compute_metric('pr_auc_score')
    if len(auprc) < len(task_names):
        auprc.extend([np.nan] * (len(task_names) - len(auprc)))
    print(pathresults + str(i) + str(name) + '_AUPRC:', auprc)
    auprc_avg = meter.compute_metric('pr_auc_score', reduction='mean')
    print(pathresults + str(i) + str(name) + '_AUPRC_avg:', auprc_avg)
    accuracy = meter.compute_metric('accuracy')
    if len(accuracy) < len(task_names):
        accuracy.extend([np.nan] * (len(task_names) - len(accuracy)))
    print(pathresults+str(i)+str(name) + '_Accuracy:', accuracy)
    accuracy_avg = meter.compute_metric('accuracy', reduction='mean')
    print(pathresults + str(i) + str(name) + '_Accuracy_avg:', accuracy_avg)
    recall = meter.compute_metric('recall')
    if len(recall) < len(task_names):
        recall.extend([np.nan] * (len(task_names) - len(recall)))
    print(pathresults+str(i)+str(name) + '_Recall:', recall)
    recall_avg = meter.compute_metric('recall', reduction='mean')
    print(pathresults + str(i) + str(name) + '_Recall_avg:', recall_avg)
    precision = meter.compute_metric('precision')
    if len(precision) < len(task_names):
        precision.extend([np.nan] * (len(task_names) - len(precision)))
    print(pathresults+str(i)+str(name) + '_Precision:', precision)
    precision_avg = meter.compute_metric('precision', reduction='mean')
    print(pathresults + str(i) + str(name) + '_Precision_avg:', precision_avg)
    f1_score = meter.compute_metric('f1')
    if len(f1_score) < len(task_names):
        f1_score.extend([np.nan] * (len(task_names) - len(f1_score)))
    print(pathresults+str(i)+str(name) + '_F1: ', f1_score)
    f1_avg = meter.compute_metric('f1', reduction='mean')
    print(pathresults + str(i) + str(name) + '_F1_avg: ', f1_avg)
    specificity = meter.compute_metric('specificity')
    if len(specificity) < len(task_names):
        specificity.extend([np.nan] * (len(task_names) - len(specificity)))
    print(pathresults + str(i) + str(name) + '_Specificity: ', specificity)
    specificity_avg = meter.compute_metric('specificity', reduction='mean')
    print(pathresults + str(i) + str(name) + '_Specificity_avg: ', specificity_avg)
    results_df = pd.DataFrame({
        'SMILES': smis,
        'True': [str(lst) for lst in labels],
        'Pred': [str(lst) for lst in preds],
        'Prob': [str(lst) for lst in probs]
    })
    results_df.to_csv(f"{pathresults}{i}_out.csv", index=False)

    results_eval = pd.DataFrame({
        'Task': task_names,
        'AUC': auc,
        'ACC': accuracy,
        'PRAUC': auprc,
        'Recall': recall,
        'Precision': precision,
        'F1': f1_score,
        'Specificity': specificity
    })
    results_eval.to_csv(f"{pathresults}result_evalue.csv", index=False)
    return auc_avg,auprc_avg,accuracy_avg,recall_avg,precision_avg,f1_avg,specificity_avg


path='Class/'
pathresults='Class/Results/'
datasets=pd.read_csv(path+'Classify_endpoints.csv')
sfolder = MultilabelStratifiedKFold(n_splits=10,random_state=16,shuffle=True)
X = datasets['SMILES']
y = datasets.drop('SMILES', axis=1)
y = y.fillna(0)
masks = (y != 0).astype(int)
train_aucs=[]
train_auprcs=[]
train_accuracys=[]
train_recalls=[]
train_precisions=[]
train_f1s=[]
train_specificitys=[]
valid_aucs=[]
valid_auprcs=[]
valid_accuracys=[]
valid_recalls=[]
valid_precisions=[]
valid_f1s=[]
valid_specificitys=[]

for fold_idx, (train_idx, valid_idx) in enumerate(sfolder.split(X, y)):
    print(fold_idx)
    train_sets=datasets.iloc[train_idx]
    train_datasets = load_data(train_sets, path + f'{fold_idx}_admet_train', False)
    train_loader = DataLoader(train_datasets, batch_size=1024, shuffle=True,
                              collate_fn=collate_molgraphs)
    valid_sets=datasets.iloc[valid_idx]
    valid_datasets = load_data(valid_sets, path + f'{fold_idx}_admet_train', False)
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
    n_task = y.shape[1]
    print(n_task)
    fn = '/home/guyaxin/DMPK/5_CL_learning/CL_model/model_80.pth'
    state_dict = torch.load(fn, map_location=torch.device('cpu'))
    model.load_my_state_dict(state_dict)
    model = model.to(device)

    #Train
    n_epochs = 1
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    lr_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.00001, last_epoch=-1)

    min_score1 = 0.8595
    min_score2 = 0.8295
    epochs = []
    scores1 = []
    scores2 = []
    val_scores1=[]
    val_scores2 = []
    losses = []
    val_losses=[]
    for e in range(n_epochs):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
        val_score = run_an_eval_epoch(model, valid_loader,loss_fn)
        epochs.append(e)
        scores1.append(score[0])
        scores2.append(score[1])
        val_scores1.append(val_score[0])
        val_scores2.append(val_score[1])
        losses.append(score[-1])
        val_losses.append(val_score[-1])
        if e % 10 == 0:
            print("第%d个epoch的学习率：%f" % (e + 1, optimizer.param_groups[0]['lr']))
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, validation {} {:.4f}'.format(
                e + 1, n_epochs, 'auc', val_score[0], 'acc', val_score[1], 'loss', val_score[-1]))
        if val_score[0] > min_score1 and val_score[1] > min_score2:
            torch.save(model.state_dict(), os.path.join(pathresults, str(fold_idx) + 'model_{}.pth'.format(str(e))))

    a1 = pd.DataFrame(range(n_epochs), columns=['epoch'])
    a1['lr'] = pd.DataFrame(lr_list)
    a1.to_csv(pathresults + str(fold_idx) + 'lr.csv', index=False)
    plt.plot(range(n_epochs), lr_list, color='r')
    plt.savefig(pathresults + str(fold_idx) + 'lr.png', bbox_inches='tight', dpi=500)


    a = pd.DataFrame(scores1, columns=['train_auc'])
    b = pd.DataFrame(val_scores1, columns=['validation_auc'])
    a2 = pd.DataFrame(scores2, columns=['train_acc'])
    b2 = pd.DataFrame(val_scores2, columns=['validation_acc'])
    c = pd.DataFrame(losses, columns=['train_loss'])
    d = pd.DataFrame(val_losses, columns=['validation_loss'])
    e = pd.concat([a, b, a2, b2, c, d], axis=1)
    e.to_csv(pathresults+'loss_auc_acc.csv')

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
    auc,auprc,accuracy,recall,precision,f1,specificity = eval(fold_idx,train_loader,gcn_net,'train')

    train_aucs.append(auc)
    train_auprcs.append(auprc)
    train_accuracys.append(accuracy)
    train_recalls.append(recall)
    train_precisions.append(precision)
    train_f1s.append(f1)
    train_specificitys.append(specificity)

    print('Validation sets'+str(fold_idx))
    auc,auprc,accuracy,recall,precision,f1,specificity = eval(fold_idx,valid_loader, gcn_net,'valid')
    valid_aucs.append(auc)
    valid_auprcs.append(auprc)
    valid_accuracys.append(accuracy)
    valid_recalls.append(recall)
    valid_precisions.append(precision)
    valid_f1s.append(f1)
    valid_specificitys.append(specificity)

metric_config = {

    'train': [
        ('AUC', train_aucs),
        ('AUPRC', train_auprcs),
        ('Accuracy', train_accuracys),
        ('Recall', train_recalls),
        ('Precision', train_precisions),
        ('F1', train_f1s)
    ],

    'valid': [
        ('AUC', valid_aucs),
        ('AUPRC', valid_auprcs),
        ('Accuracy', valid_accuracys),
        ('Recall', valid_recalls),
        ('Precision', valid_precisions),
        ('F1', valid_f1s)
    ]
}


dfs = []
for prefix, metrics in metric_config.items():
    for metric_name, data in metrics:
        col_name = f"{prefix}_{metric_name}"  # 例如 'train_AUC'
        dfs.append(pd.DataFrame(data, columns=[col_name]))


df = pd.concat(dfs, axis=1)
df.to_csv(f"{pathresults}evaluate_results.csv", index=False)
