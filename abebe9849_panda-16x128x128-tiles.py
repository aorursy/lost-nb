#!/usr/bin/env python
# coding: utf-8

# In[1]:


###lafossのモデル　nakayaさんのものをほぼ使う。


# In[2]:


class CFG:
    debug=False
    #height=256
    #width=256
    lr=1e-4
    batch_size=16
    epochs=1 # you can train more epochs
    seed=777
    target_size=1
    target_col='isup_grade'
    n_fold=4


# In[3]:


import os
import numpy as np 
import pandas as pd
os.listdir('../input/prostate-cancer-grade-assessment')


# In[4]:


train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
test = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')
sample = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


sample.head()


# In[8]:


train['isup_grade'].hist()


# In[9]:


# ====================================================
# Library
# ====================================================

import sys

import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter

import skimage.io
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,RandomGamma, RandomRotate90,GaussNoise
from albumentations.pytorch import ToTensorV2
"""
import warnings 
warnings.filterwarnings('ignore')"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[10]:


# ====================================================
# Utils
# ====================================================

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

    
def init_logger(log_file='train.log'):
    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler
    
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    
    logger = getLogger('PANDA')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

LOG_FILE = 'train.log'
LOGGER = init_logger(LOG_FILE)

#再現性の確保
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)


# In[11]:


def tile(img, sz=120, N=16):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N
                           -len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img#[N,size,size,3]

class TrainDataset_lafoss(Dataset):
    def __init__(self, df, labels, transform1=None,tensor=True):
        self.df = df
        self.labels = labels
        self.transform = transform1
        self.tensor = tensor
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'../input/prostate-cancer-grade-assessment/train_images/{file_name}.tiff'
        images = skimage.io.MultiImage(file_path)[2]
        images = tile(images)
        if self.transform:
            images = [cv2.cvtColor(self.transform(image=img)['image'], cv2.COLOR_BGR2RGB) for img in images]
        else:
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        #ここまでは、ndarray
        images = np.stack(images, 0)
        if self.tensor:
            images = torch.from_numpy(images.transpose((0,3,1,2)))

        
            
        label = torch.tensor(self.labels[idx]).float()
        
        return images, label
    
class TestDataset_lafoss(Dataset):
    def __init__(self, df, dir_name, transform1=None,tensor =True):
        self.df = df
        self.dir_name = dir_name
        self.transform = transform1
        self.tensor = tensor#bool
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'../input/prostate-cancer-grade-assessment/{self.dir_name}/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[2]
        images = tile(image)
        
        if self.transform:
            images = [cv2.cvtColor(self.transform(image=img)['image'], cv2.COLOR_BGR2RGB) for img in images]
        else:
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        images = np.stack(images, 0)#bs,n,h,w,c
        
        if self.tensor:
            images = torch.from_numpy(images.transpose((0,3,1,2)))
        return images


# In[12]:


def get_transforms1(*, data):

    #train,valid以外だったら怒る
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            #GaussNoise(p=0.5),
            #RandomAugMix(severity=3, width=3, alpha=1., p=0.2),
            #GridMask(num_grid=3, p=0.2),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])


# In[13]:


import matplotlib.pyplot as plt


train_dataset = TrainDataset_lafoss(train, train[CFG.target_col], transform1=get_transforms1(data='train'),tensor=False)# train[CFG.target_col]は0~5
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


# In[14]:


get_ipython().run_cell_magic('time', '', '\nfor img, label in train_loader:\n    for j in range(img.shape[1]):\n        plt.imshow(img[0][j])\n\n        plt.show()\n    break')


# In[15]:


if CFG.debug:
    folds = train.sample(n=200, random_state=CFG.seed).reset_index(drop=True).copy()
else:
    folds = train.copy()


# In[16]:


train_labels = folds[CFG.target_col].values
kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[val_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)
folds.to_csv('folds.csv', index=None)
folds.head()


# In[17]:


import sys
sys.path.insert(0, '/kaggle/input/pytorch-efnet-ns/')
import geffnet


# In[18]:


import torch
import torch.nn as nn
#!pip install efficientnet_pytorch
#!pip install geffnet
#from efficientnet_pytorch import EfficientNet
class Model(nn.Module):
    def __init__(self,n=1):
        super().__init__()
        m = geffnet.efficientnet_b0(pretrained=False)
        self.enc = nn.Sequential(*list(m.children())[:-3])    
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(nc,512),
                            nn.ReLU(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
    def forward(self,x):
        shape = x.size()
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
          #print(x.size())##orch.Size([160, 3, 128, 128])
        x = self.enc(x)
          #print("finish_enc",x.size())
        shape = x.shape#torch.Size([160, 1280, 4, 4])
          #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
          #print("to_head",x.size())#torch.Size([10, 1280, 64, 4])
        x = self.head(x)
        return x
    
def fix_model_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("enc.*.*.*."):
            name = name[10:]  # remove 'model.' of dataparallel
        elif name.startswith('head.*.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


# In[19]:


weights_path = "/kaggle/input/panda-efnetb2-180-weight/fold0_efnet_2020-09-05-114528.pth"
model = Model()
state_dict = torch.load(weights_path,map_location=device)
model.load_state_dict(state_dict)
print(model)


# In[20]:


from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

#回帰に対して適切な閾値を決めて分類クラスを返す流れ。
class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)#self._kappa_lossの引数は3つだが、coefを固定するもの
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')#第一引数にある関数を最適化するように第二引数のパラメータを調整。この場合は回帰→分類のための閾値
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:





# In[21]:


Ng = 6
def Kloss(x, target,df):
    y_shift = df.isup_grade.mean()
    x = Ng*torch.sigmoid(x.float()).view(-1) - 0.5
    target = target.float()
    return 1.0 - (2.0*((x-y_shift)*(target-y_shift)).sum() - 1e-3)/        (((x-y_shift)**2).sum() + ((target-y_shift)**2).sum() + 1e-3)


# In[22]:


def train_fn(fold):
    print(f"### fold: {fold} ###")
    
    optimized_rounder = OptimizedRounder()
        
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    
    #タイルごとに拡張を入れる
    train_dataset = TrainDataset_lafoss(folds.loc[trn_idx].reset_index(drop=True), 
                                 folds.loc[trn_idx].reset_index(drop=True)[CFG.target_col], 
                                 transform1=get_transforms1(data='train'),tensor=True)
    valid_dataset = TrainDataset_lafoss(folds.loc[val_idx].reset_index(drop=True), 
                                 folds.loc[val_idx].reset_index(drop=True)[CFG.target_col], 
                                 transform1=get_transforms1(data='valid'),tensor=True)
    
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
    
    model = Model()
    weights_path = "/kaggle/input/panda-efnetb2-180-weight/fold{}_efnet_2020-09-05-114528.pth".format(fold)
    state_dict = torch.load(weights_path,map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
    
    criterion = nn.MSELoss()#分類の時はnn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss()
    best_score = -100
    best_loss = np.inf
    best_preds = None
    
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()

        model.train()
        avg_loss = 0.

        optimizer.zero_grad()
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:

            images = images.to(device)
            labels = labels.to(device)
            
            y_preds = model(images.float())
            #loss = criterion(y_preds.view(-1), labels)
            loss = Kloss(y_preds.view(-1), labels,folds.loc[trn_idx].reset_index(drop=True))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:
            
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                y_preds = model(images.float())
            
            
            valid_labels.append(labels.to('cpu').numpy())

            #loss = criterion(y_preds.view(-1), labels)
            loss = Kloss(y_preds.view(-1), labels,folds.loc[val_idx].reset_index(drop=True))
            y_preds = 6*torch.sigmoid(y_preds.float()).view(-1) - 0.5
            #print("valid_preds",y_preds.size())
            preds.append(y_preds.to('cpu').numpy())
            
            avg_val_loss += loss.item() / len(valid_loader)
        
        scheduler.step(avg_val_loss)
            
        preds = np.concatenate(preds)
        #print("preds",preds.shape)
        valid_labels = np.concatenate(valid_labels)
        #回帰の値を分類にする流れ
        
        optimized_rounder.fit(preds, valid_labels)
        coefficients = optimized_rounder.coefficients()
        final_preds = optimized_rounder.predict(preds, coefficients)
        #print("final_preds",final_preds.shape)
        LOGGER.debug(f'Counter preds: {Counter(final_preds)}')#np.concatenate(final_preds)
        LOGGER.debug(f'coefficients: {coefficients}')
        score = quadratic_weighted_kappa(valid_labels, final_preds)
        #score = quadratic_weighted_kappa(valid_labels, preds)

        elapsed = time.time() - start_time#loggerのためのもの
        
        LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.debug(f'  Epoch {epoch+1} - QWK: {score}  coefficients: {coefficients}')
        #LOGGER.debug(f'  Epoch {epoch+1} - QWK: {score}')
        
        if score>best_score:#QWKのスコアが良かったら予測値を更新...best_epochをきめるため
            best_score = score
            best_preds = preds
            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f}')
            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model  coefficients: {coefficients}')
            torch.save(model.state_dict(), f'fold{fold}_efnet_b2_ns.pth')#各epochのモデルを保存。。。best_epoch終了時のモデルを推論に使用する？
    
    return best_preds, valid_labels,model
    #return preds, valid_labels


# In[23]:


"""
preds = []
valid_labels = []
models = []
for fold in range(CFG.n_fold):
    _preds, _valid_labels,_model = train_fn(fold)
    preds.append(_preds)
    valid_labels.append(_valid_labels)"""


# In[24]:


"""
preds = np.concatenate(preds)
valid_labels = np.concatenate(valid_labels)

optimized_rounder = OptimizedRounder()
optimized_rounder.fit(preds, valid_labels)
coefficients = optimized_rounder.coefficients()#どうする？
final_preds = optimized_rounder.predict(preds, coefficients)
LOGGER.debug(f'Counter preds: {Counter(final_preds)}')#np.concatenate()
LOGGER.debug(f'coefficients: {coefficients}')

score = quadratic_weighted_kappa(valid_labels, final_preds)
LOGGER.debug(f'CV QWK: {score}')"""


# In[25]:


def inference(model, test_loader, device):
    
    model.to(device) 
    
    probs = []

    for i, images in tqdm(enumerate(test_loader), total=len(test_loader)):
            
        images = images.to(device)
        if i==0:
            print(images.size())
            
        with torch.no_grad():
            y_preds = model(images)
            y_preds = 6*torch.sigmoid(y_preds.float()).view(-1) - 0.5
            
        probs.append(y_preds.to('cpu').numpy())

    probs = np.concatenate(probs)
    
    return probs


# In[26]:


coefficients=np.array([0.5060126 ,1.50290319, 2.56765878, 3.3614414, 4.60342127])


# In[27]:


def submit_l(sample, coefficients, dir_name='test_images'):
    if os.path.exists(f'../input/prostate-cancer-grade-assessment/{dir_name}'):
        print('run inference')
        test_dataset = TestDataset_lafoss(sample, dir_name,get_transforms1(data='valid'),tensor = True)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        probs = []
        for fold in range(CFG.n_fold):
            weights_path = "/kaggle/input/panda-efnetb2-180-weight/fold{}_efnet_2020-09-09-185804.pth".format(fold)
            model = Model()
            state_dict = torch.load(weights_path,map_location=device)
            model.load_state_dict(state_dict)
            _probs = inference(model, test_loader, device)
            probs.append(_probs)
            #if fold ==0:break
        optimized_rounder = OptimizedRounder()
        probs = np.mean(probs, axis=0)
        preds = optimized_rounder.predict(probs, coefficients)
        sample['isup_grade'] = preds
    return sample


# In[28]:


# check using train_images
submission = submit_l(train.head(), coefficients, dir_name='train_images')
submission['isup_grade'] = submission['isup_grade'].astype(int)
#submission.to_csv('submission.csv', index=False)
#score = quadratic_weighted_kappa(folds["isup_grade"], submission['isup_grade'])
#print("QWK:",score)
submission.head()


# In[29]:


# test submission
submission = submit_l(sample, coefficients, dir_name='test_images')
submission['isup_grade'] = submission['isup_grade'].astype(int)
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




