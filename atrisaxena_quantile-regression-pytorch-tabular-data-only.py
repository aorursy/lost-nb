#!/usr/bin/env python
# coding: utf-8



import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange
from time import time




root_dir = Path('/kaggle/input/osic-pulmonary-fibrosis-progression')
model_dir = '/kaggle/working/model_states'
num_kfolds = 5
batch_size = 32
learning_rate = 3e-3
num_epochs = 1000
es_patience = 10
quantiles = (0.2, 0.5, 0.8)
model_name ='descartes'
tensorboard_dir = Path('/kaggle/working/runs')




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform
        self.mode = mode

        tr = pd.read_csv(Path(root_dir)/"train.csv")
        tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
        chunk = pd.read_csv(Path(root_dir)/"test.csv")

        sub = pd.read_csv(Path(root_dir)/"sample_submission.csv")
        sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
        sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
        sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
        sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

        tr['WHERE'] = 'train'
        chunk['WHERE'] = 'val'
        sub['WHERE'] = 'test'
        data = tr.append([chunk, sub])

        data['min_week'] = data['Weeks']
        data.loc[data.WHERE == 'test', 'min_week'] = np.nan
        data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

        base = data.loc[data.Weeks == data.min_week]
        base = base[['Patient', 'FVC']].copy()
        base.columns = ['Patient', 'min_FVC']
        base['nb'] = 1
        base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
        base = base[base.nb == 1]
        base.drop('nb', axis=1, inplace=True)

        data = data.merge(base, on='Patient', how='left')
        data['base_week'] = data['Weeks'] - data['min_week']
        del base

        COLS = ['Sex', 'SmokingStatus']
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)

        data['age'] = (data['Age'] - data['Age'].min()) /                       (data['Age'].max() - data['Age'].min())
        data['BASE'] = (data['min_FVC'] - data['min_FVC'].min()) /                        (data['min_FVC'].max() - data['min_FVC'].min())
        data['week'] = (data['base_week'] - data['base_week'].min()) /                        (data['base_week'].max() - data['base_week'].min())
        data['percent'] = (data['Percent'] - data['Percent'].min()) /                           (data['Percent'].max() - data['Percent'].min())
        self.FE += ['age', 'percent', 'week', 'BASE']

        self.raw = data.loc[data.WHERE == mode].reset_index()
        del data

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'patient_id': self.raw['Patient'].iloc[idx],
            'features': self.raw[self.FE].iloc[idx].values,
            'target': self.raw['FVC'].iloc[idx]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.raw['Patient']
        for train_idx, val_idx in gkf.split(self.raw, self.raw, groups):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val

    def group_split(self, test_size=0.2):
        """To test no-kfold
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw['Patient']
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val




class QuantModel(nn.Module):
    def __init__(self, in_tabular_features=9, out_quantiles=3):
        super(QuantModel, self).__init__()
        self.fc1 = nn.Linear(in_tabular_features, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, out_quantiles)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantile_loss(preds, target, quantiles):
    #assert not target.requires_grad
    assert len(preds) == len(target)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

def metric_loss(pred_fvc,true_fvc):
        #Implementation of the metric in pytorch
    sigma = pred_fvc[:, 2] - pred_fvc[:, 0]
    true_fvc=torch.reshape(true_fvc,pred_fvc[:,1].shape)
    sigma_clipped=torch.clamp(sigma,min=70)
    delta=torch.clamp(torch.abs(pred_fvc[:,1]-true_fvc),max=1000)
    metric=torch.div(-torch.sqrt(torch.tensor([2.0]))*delta,sigma_clipped)-torch.log(torch.sqrt(torch.tensor([2.0]))*sigma_clipped)
    return metric




models = []

train_loss = []
val_lll = []
# Load the data
data = ClinicalDataset(root_dir=root_dir, mode='train')
folds = data.group_kfold(num_kfolds)
#t0 = time()
#if len(testfiles) == 5:
    #f= open("/kaggle/working/training.log","w+") 
for fold, (trainset, valset) in enumerate(folds):
    best_val = None
    patience = es_patience
    model = QuantModel().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=learning_rate)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)


    print("=="*20+"Fold "+str(fold+1)+"=="*20)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = model_dir+f'/fold_{fold}.pth'
    now = datetime.now()
    dataset_sizes = {'train': len(trainset), 'val': len(valset)}
    dataloaders = {
            'train': DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, num_workers=2),
            'val': DataLoader(valset, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    }
    train_loss_epoch = []
    val_lll_epoch = []
    for epoch in range(num_epochs):
        start_time = time()
        itr = 1
        model.train()
        train_losses =[]
        for batch in dataloaders['train']:
            inputs = batch['features'].float().to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                preds = model(inputs)
                loss = quantile_loss(preds, targets, quantiles)
                train_losses.append(loss.tolist())
                loss.backward()
                optimizer.step()
           
            if itr % 50 == 0:
                print(f"Epoch #{epoch+1} Iteration #{itr} loss: {loss}")
            itr += 1
            
        model.eval()
        all_preds = []
        all_targets = []
        for batch in dataloaders['val']:
            inputs = batch['features'].float().to(device)
            targets = batch['target']
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                preds = model(inputs)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist()) # np.append(an_array, row_to_append, 0)
        all_preds =torch.FloatTensor(all_preds)
        all_targets =torch.FloatTensor(all_targets)
        val_metric_loss = metric_loss(all_preds, all_targets)
        val_metric_loss = torch.mean(val_metric_loss).tolist()

        lr_scheduler.step()
        print(f"Epoch #{epoch+1}","Training loss : {0:.4f}".format(np.mean(train_losses)),"Validation LLL : {0:.4f}".format(val_metric_loss),"Time taken :",str(timedelta(seconds=time() - start_time))[:7])
        train_loss_epoch.append(np.mean(train_losses))
        val_lll_epoch.append(val_metric_loss)
        if not best_val:
            best_val = val_metric_loss  # So any validation roc_auc we have is the best one for now
            print("Info : Saving model")
            torch.save(copy.deepcopy(model.state_dict()), model_path)  # Saving the model
        if val_metric_loss > best_val:
            print("Info : Saving model as Laplace Log Likelihood is increased from {0:.4f}".format(best_val),"to {0:.4f}".format(val_metric_loss))
            best_val = val_metric_loss
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            torch.save(copy.deepcopy(model.state_dict()), model_path)  # Saving current best model torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Validation Laplace Log Likelihood: {:.3f}'.format(best_val))
                break
    model.load_state_dict(torch.load(model_path))
    models.append(model)
    train_loss.append(train_loss_epoch)
    val_lll.append(val_lll_epoch)
print('Finished Training of BiLSTM Model')




data = ClinicalDataset(root_dir, mode='test')

avg_preds = np.zeros((len(data), len(quantiles)))

for model in models:
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    preds = []
    for batch in dataloader:
        inputs = batch['features'].float()
        inputs = inputs.cuda()
        with torch.no_grad():
            x = model(inputs)
            preds.append(x)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    avg_preds += preds

avg_preds /= len(models)
df = pd.DataFrame(data=avg_preds, columns=list(quantiles))
df['Patient_Week'] = data.raw['Patient_Week']
df['FVC'] = df[quantiles[1]]
df['Confidence'] = df[quantiles[2]] - df[quantiles[0]]
df = df.drop(columns=list(quantiles))
df.to_csv('submission.csv', index=False)




print(len(df))
df.head()




df.to_csv('submission.csv', index = False)




get_ipython().system("head 'submission.csv'")

