#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torchvision import models
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

import cv2 
import pydicom
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms

from torch import nn
# from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import MemoryEfficientSwish
import warnings

import random
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import pydicom
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
import sys

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from datetime import datetime, timedelta
from time import time
import torch.nn.functional as F
import copy


# In[2]:


package_path = '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)


# In[3]:


from efficientnet_pytorch import EfficientNet


# In[4]:


warnings.simplefilter('ignore')
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =True
    
seed_everything(42)


# In[5]:


class Config:
    def __init__(self):
        self.FOLDS = 2
        self.EPOCHS = 40
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.TRAIN_BS = 32
        self.VALID_BS = 128
        self.model_type = 'efficientnet-b3'
        self.loss_fn = nn.L1Loss()
        
config = Config()


# In[6]:


path = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/')
path.ls()


# In[7]:


train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[8]:


train_df = train_df.drop(np.nonzero(np.array(train_df['Patient'] == 'ID00011637202177653955184',dtype=float))[0], axis=0).reset_index(drop=True)
train_df = train_df.drop(np.nonzero(np.array(train_df['Patient'] == 'ID00052637202186188008618',dtype=float))[0], axis=0).reset_index(drop=True)


# In[9]:


def get_tab(df):
    vector = [(df.Weeks.values[0] - 30 )/30]
    
    if df.Sex.values[0] == 'male':
       vector.append(0)
    else:
       vector.append(1)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
    return np.array(vector) 


# In[10]:


TAB = {}
TARGET = {}
Person = []

for i, p in tqdm(enumerate(train_df.Patient.unique())):
    sub = train_df.loc[train_df.Patient == p]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc)[0]
    
    TARGET[p] = a
    TAB[p] = get_tab(sub)
    Person.append(p)

Person = np.array(Person)


# In[11]:


def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize(d.pixel_array / 2**11, (512, 512))


# In[12]:


class Dataset:
    def __init__(self, path, df, tabular, targets, mode , folder = 'train' ):
        self.df = df
        self.tabular = tabular
        self.targets = targets
        self.folder = folder
        self.mode = mode
        self.path = path
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.loc[idx,:]
        pid = row['Patient']
        # Path to record
        record = self.path/self.folder/pid
        # select image id
        try: 
            
            img_id =  np.random.choice(len(record.ls()))
            
            img = get_img(record.ls()[img_id])
            img = self.transform(img)
            tab = torch.from_numpy(self.tabular[pid]).float()
            if self.mode == 'train':
                target = torch.tensor(self.targets[pid])
                return (img,tab), target
            else:
                return (img,tab)
        except Exception as e:
            print(e)
            print(pid, img_id)


# In[13]:


def collate_fn(b):
    xs, ys = zip(*b)
    imgs, tabs = zip(*xs)
    return (torch.stack(imgs).float(),torch.stack(tabs).float()),torch.stack(ys).float()


# In[14]:


pretrained_model = {
    'efficientnet-b0': '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',
    'efficientnet-b3': '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth'
}


# In[15]:


class OSIC_Model(nn.Module):
    def __init__(self,eff_name='efficienet-b0'):
        super().__init__()
        self.input = nn.Conv2d(1,3,kernel_size=3,padding=1,stride=2)
        self.bn = nn.BatchNorm2d(3)
        #self.model = EfficientNet.from_pretrained(f'efficientnet-{eff_name}-c8376fa2.pth')
        self.model = EfficientNet.from_name(eff_name)
        self.model.load_state_dict(torch.load(pretrained_model[eff_name]))
        self.model._fc = nn.Linear(1536, 500, bias=True)
        self.meta = nn.Sequential(nn.Linear(4, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500,250),
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.output = nn.Linear(500+250, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x,tab):
        x = self.relu(self.bn(self.input(x)))
        x = self.model(x)
        tab = self.meta(tab)
        x = torch.cat([x, tab],dim=1)
        return self.output(x)


# In[16]:


from sklearn.model_selection import KFold

def get_split_idxs(n_folds=5):
    kv = KFold(n_splits=n_folds)
    splits = []
    for i,(train_idx, valid_idx) in enumerate(kv.split(Person)):
        splits.append((train_idx, valid_idx))
        
    return splits


# In[17]:


splits = get_split_idxs(n_folds=config.FOLDS)


# In[18]:


def train_loop(model, dl, opt, sched, device, loss_fn):
    model.train()
    for X,y in dl:
        imgs = X[0].to(device)
        tabs = X[1].to(device)
        y = y.to(device)
        outputs = model(imgs, tabs)
        loss = loss_fn(outputs.squeeze(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
            

def eval_loop(model, dl, device, loss_fn):
    model.eval()
    final_outputs = []
    final_loss = []
    with torch.no_grad():
        for X,y in dl:
            imgs = X[0].to(device)
            tabs = X[1].to(device)
            y=y.to(device)

            outputs = model(imgs, tabs)
            loss = loss_fn(outputs.squeeze(), y)

            final_outputs.extend(outputs.detach().cpu().numpy().tolist())
            final_loss.append(loss.detach().cpu().numpy())
        
    return final_outputs, final_loss


# In[19]:


from functools import partial

def apply_mod(m,f):
    f(m)
    for l in m.children(): apply_mod(l,f)

def set_grad(m,b):
    if isinstance(m, (nn.Linear, nn.BatchNorm2d)): return 
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)


# In[20]:


models = {}
for i in range(config.FOLDS):
    models[i] = OSIC_Model(config.model_type)


# In[21]:


for k,v in models.items():
    apply_mod(v.model, partial(set_grad, b=False))


# In[22]:


train = train_df.loc[train_df['Patient'].isin(Person[:21])].reset_index(drop=True)
train_ds = Dataset(path, train, TAB, TARGET, mode='train')
train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=config.TRAIN_BS,
    shuffle=True,
    collate_fn=collate_fn        
)


# In[23]:


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4
i=1
for X,y in train_dl:
    pass
j=0
for i in range(1, columns*rows +1):
    img = np.array(X[0][j].permute(1,2,0))
    img = cv2.cvtColor(img ,cv2.COLOR_GRAY2RGB)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    j += 1
plt.show()


# In[24]:


history = []


# In[25]:


for i, (train_idx, valid_idx) in enumerate(splits):
    print(f"===================Fold : {i} ================")

    train = train_df.loc[train_df['Patient'].isin(Person[train_idx])].reset_index(drop=True)
    valid = train_df.loc[train_df['Patient'].isin(Person[valid_idx])].reset_index(drop=True)


    train_ds = Dataset(path, train, TAB, TARGET, mode= 'train')
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=config.TRAIN_BS,
        shuffle=True,
        collate_fn=collate_fn        
    )

    valid_ds = Dataset(path, valid, TAB, TARGET, mode='train')
    valid_dl = torch.utils.data.DataLoader(
        dataset=valid_ds,
        batch_size=config.VALID_BS,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = models[i]
    model.to(config.DEVICE)
    lr=1e-3
    momentum = 0.9
    
    num_steps = len(train_dl)
    optimizer = Adam(model.parameters(), lr=lr,weight_decay=0.1)
    scheduler = OneCycleLR(optimizer, 
                           max_lr=lr,
                           epochs=config.EPOCHS,
                           steps_per_epoch=num_steps
                           )
    sched = ReduceLROnPlateau(optimizer,
                              verbose=True,
                              factor=0.1)
    losses = []
    for epoch in range(config.EPOCHS):
        print(f"=================EPOCHS {epoch+1}================")
        train_loop(model, train_dl, optimizer, scheduler, config.DEVICE,config.loss_fn)
        metrics = eval_loop(model, valid_dl,config.DEVICE,config.loss_fn)
        total_loss = np.array(metrics[1]).mean()
        losses.append(total_loss)
        print("Loss ::\t", total_loss)
        sched.step(total_loss)
        
    model.to('cpu')
    history.append(losses)
    
    
        


# In[26]:


for k, m in models.items():
    torch.save(m.state_dict(), f'fold_{k}.pth')


# In[27]:


test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')


# In[28]:


test_df.head()


# In[29]:


test_data= []
for i in range(len(test_df)):
    for j in range(-12, 134):
        test_data.append([test_df['Patient'][i],j,test_df['Age'][i],test_df['Sex'][i],test_df['SmokingStatus'][i], test_df['FVC'][i],test_df['Percent'][i
        ],str(test_df['Patient'][i])+'_'+str(j)])

test_data = pd.DataFrame(test_data, columns=['Patient','Weeks','Age','Sex','SmokingStatus','FVC','Percent','Patient_Week'])


# In[30]:


test_data.head()


# In[31]:


TAB_test = {}

Person_test = []

for i, p in tqdm(enumerate(test_data.Patient.unique())):
    sub = test_data.loc[test_data.Patient == p]

    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T

    TAB_test[p] = get_tab(sub)
    Person_test.append(p)

Person_test = np.array(Person_test)


# In[32]:


def collate_fn_test(b):
    imgs, tabs = zip(*b)
    return (torch.stack(imgs).float(),torch.stack(tabs).float())


# In[33]:


TARGET = {}
test = test_data
test_ds = Dataset(path, test_data, TAB_test,TARGET, mode= 'test')
test_dl = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn_test        
)


# In[34]:


avg_predictions= np.zeros((730,1))

for i in range(len(models)):
    
    predictions = []
    model = models[i]
    model = model.to(config.DEVICE)
    model.load_state_dict(torch.load('./fold_' +str(i)+'.pth'))
    model.eval()
    with torch.no_grad():
        for X in test_dl:
            imgs = X[0].to(config.DEVICE)
            tabs = X[1].to(config.DEVICE)

            pred = model(imgs, tabs)

            predictions.extend(pred.detach().cpu().numpy().tolist())
    avg_predictions += predictions


# In[35]:


predictions = avg_predictions / len(models)


# In[36]:


fvc = []
conf = []
for i in range(len(test_data)):
    p =test_data['Patient'][i]
    B_test = predictions[i][0] * test_df.Weeks.values[test_df.Patient == p][0]
    fvc.append(predictions[i][0] * test_data['Weeks'][i] + test_data['FVC'][i] - B_test)
    conf.append(test_data['Percent'][i] + abs(predictions[i][0]) * abs(test_df.Weeks.values[test_df.Patient == p][0] - test_data['Weeks'][i]))


# In[37]:



submission = test_data[['Patient_Week']]


# In[38]:


sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
sub.head()


# In[39]:


subm ={}
for i in range(len(submission)):
    subm[submission['Patient_Week'][i]]=[float(fvc[i]),float(conf[i])]


# In[40]:


sub['FVC'] = sub['FVC'].astype(float)
sub['Confidence'] = sub['Confidence'].astype(float)
for i in range(len(sub)):
    id = sub['Patient_Week'][i]
    sub['FVC'][i]= float(subm[id][0])
    sub['Confidence'][i] = float(subm[id][1])


# In[41]:


sub.head()


# In[42]:


sub.to_csv('submission_img.csv', index=False)


# In[43]:


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


# In[44]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[45]:


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


# In[46]:


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


# In[47]:


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


# In[48]:


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


# In[49]:


print(len(df))
df.head()


# In[50]:


df.to_csv('submission_reg.csv', index = False)


# In[51]:


sub_img = pd.read_csv('./submission_img.csv')
sub_reg = pd.read_csv('./submission_reg.csv')


# In[52]:


sub_img.head(2)


# In[53]:


sub_reg.head(2)


# In[54]:


for i in range(len(sub_img)):
    sub_img['FVC'][i] = 0.25*sub_img['FVC'][i] + 0.75*sub_reg.loc[sub_reg.Patient_Week == sub_img['Patient_Week'][i]]['FVC']
    sub_img['Confidence'][i] = 0.26*sub_img['Confidence'][i] + 0.74*sub_reg.loc[sub_reg.Patient_Week == sub_img['Patient_Week'][i]]['Confidence']


# In[55]:


sub = sub_img
sub.head()


# In[56]:


sub.to_csv('submission.csv',index= False)

