#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import os
import sys
import time
import random
import logging
import datetime as dt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision as vision

from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path
from PIL import Image
from contextlib import contextmanager

from joblib import Parallel, delayed
from tqdm import tqdm
from fastprogress import master_bar, progress_bar

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score


# In[2]:


#!mkdir -p /tmp/.torch/models/
#!wget -O /tmp/.torch/models/se_resnet152-d17c99b7.pth http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth
#import pretrainedmodels
torch.cuda.is_available()


# In[3]:


@contextmanager
def timer(name="Main", logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0} s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
        

def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# In[4]:


logger = get_logger(name="Main", tag="Pytorch-VGG16")


# In[5]:


@contextmanager
def timer(name="Main", logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0} s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
        

def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# In[6]:


get_ipython().system('ls ../input')
labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")
train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")
sample = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")
train.head()

cultures = [x for x in labels.attribute_name.values if x.startswith("culture")]
tags = [x for x in labels.attribute_name.values if x.startswith("tag")]
len(cultures), len(tags)


# In[7]:


import cv2
def split_culture_tag(x):
    cultures_ = list()
    tags_ = list()
    for i in x.split(" "):
        if int(i) <= len(cultures):
            cultures_.append(i)
        else:
            tags_.append(str(int(i) - len(cultures)))
    if not cultures_:
        cultures_.append(str(len(cultures)))
    if not tags_:
        tags_.append(str(len(tags)))
    return " ".join(cultures_), " ".join(tags_)

culture_ids = list()
tag_ids = list()

for v in tqdm(train.attribute_ids.values):
    c, t = split_culture_tag(v)
    culture_ids.append(c)
    tag_ids.append(t)

num_classes_c = len(cultures) + 1
num_classes_t = len(tags) + 1

train["culture_ids"] = culture_ids
train["tag_ids"] = tag_ids


def obtain_y_c(ids):
    y = np.zeros(num_classes_c)
    for idx in ids.split(" "):
        y[int(idx)] = 1
    return y

def obtain_y_t(ids):
    y = np.zeros(num_classes_t)
    for idx in ids.split(" "):
        y[int(idx)] = 1
    return y

paths = ["../input/imet-2019-fgvc6/train/{}.png".format(x) for x in train.id.values]

targets_c = np.array([obtain_y_c(y) for y in train.culture_ids.values])
targets_t = np.array([obtain_y_t(y) for y in train.tag_ids.values])
print(targets_c.shape)

def rem_bkg(img):
    y_size,x_size,col = img.shape
    
    for y in range(y_size):
        for r in range(1,6):
            col = img[y, x_size-r] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]
        for l in range(5):
            col = img[y, l] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]

    for x in range(x_size):
        for d in range(1,6):
            col = img[y_size-d, x] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]
        for u in range(5):
            col = img[u, x] 
            img[np.where((img == col).all(axis = 2))] = [255,255,255]
    
    return img

class ImageDataLoader(data.DataLoader):
    def __init__(self, root_dir: Path, 
                 df: pd.DataFrame, 
                 mode="train", 
                 transforms=None):
        self._root = root_dir
        self.transform = transforms[mode]
        self._img_id = (df["id"] + ".png").values
        
    def __len__(self):
        return len(self._img_id)
    
    def __getitem__(self, idx):
        img_id = self._img_id[idx]
        file_name = self._root / img_id
        img = Image.open(file_name)
        #img = cv2.imread(file_name.absolute().as_posix())[...,[2, 1, 0]]
        #img = rem_bkg(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
            
        return [img]
    
    
data_transforms = {
    'train': vision.transforms.Compose([
        vision.transforms.RandomResizedCrop(224),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'val': vision.transforms.Compose([
        vision.transforms.Resize(256),
        vision.transforms.CenterCrop(224),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
}

data_transforms["test"] = data_transforms["val"]


# In[8]:


class IMetDataset(data.Dataset):
    def __init__(self, tensor, device="cuda:0", labels=None):
        self.tensor = tensor
        self.labels = labels
        self.device= device
        
    def __len__(self):
        return self.tensor.size(0)
    
    def __getitem__(self, idx):
        tensor = self.tensor[idx, :]
        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.zeros((1, 1103))
            y_c = torch.FloatTensor(targets_c[idx]).to(self.device)
            y_t = torch.FloatTensor(targets_t[idx]).to(self.device)
            for i in label:
                label_tensor[0, int(i)] = 1
            label_tensor = label_tensor.to(self.device)
            return [tensor, [y_c, y_t]]
        else:
            return [tensor]


# In[9]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
    def forward(self, x):
        return x


class Densenet121(nn.Module):
    def __init__(self, pretrained: Path):
        super(Densenet121, self).__init__()
        self.densenet121 = vision.models.densenet121()
        self.densenet121.load_state_dict(torch.load(pretrained))
        self.densenet121.classifier = Classifier()
        
        dense = nn.Sequential(*list(self.densenet121.children())[:-1])
        for param in dense.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.densenet121(x)
    
class Resnet50(nn.Module):
    def __init__(self, pretrained: Path):
        super(Resnet50, self).__init__()
        self.resnet50 = vision.models.resnet50()
        self.resnet50.load_state_dict(torch.load(pretrained))
        self.resnet50.classifier = Classifier()
        
        dense = nn.Sequential(*list(self.resnet50.children())[:-1])
        for param in dense.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.resnet50(x)
    
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.linear11 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1103)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu2(self.linear11(x))
        x = self.dropout(x)
        return self.sigmoid(self.linear2(x))
    
class MultiLayerPerceptron1(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron1, self).__init__()
        self.linear1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.linear11 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 399)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu2(self.linear11(x))
        x = self.dropout(x)
        return self.sigmoid(self.linear2(x))
    
class MultiLayerPerceptron2(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron2, self).__init__()
        self.linear1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.linear11 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 706)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu2(self.linear11(x))
        x = self.dropout(x)
        return self.sigmoid(self.linear2(x))


# In[10]:


train_dataset = ImageDataLoader(
    root_dir=Path("../input/imet-2019-fgvc6/train/"),
    df=train,
    mode="train",
    transforms=data_transforms)
train_loader = data.DataLoader(dataset=train_dataset,
                               shuffle=False,
                               batch_size=64)
test_dataset = ImageDataLoader(
    root_dir=Path("../input/imet-2019-fgvc6/test/"),
    df=sample,
    mode="test",
    transforms=data_transforms)
test_loader = data.DataLoader(dataset=test_dataset,
                              shuffle=False,
                              batch_size=64)


# In[11]:


from torchvision import models
def get_feature_vector(df, loader, device):
    matrix = torch.zeros((df.shape[0], 1024)).to(device)
    model = Densenet121('../input/pytorch-pretrained-image-models/densenet121.pth') #Resnet50('../input/pytorch-pretrained-image-models/resnet50.pth')

    model.to(device)
    batch = loader.batch_size
    for i, (i_batch,) in tqdm(enumerate(loader)):
        i_batch = i_batch.to(device)
        pred = model(i_batch).detach()
        matrix[i * batch:(i + 1) * batch] = pred
    return matrix


# In[12]:


train_tensor = get_feature_vector(train, train_loader, "cuda:0")
test_tensor = get_feature_vector(sample, test_loader, "cuda:0")


# In[13]:


del train_dataset, train_loader
del test_dataset, test_loader
gc.collect()


# In[14]:


from numpy.random import beta
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.sum(F_loss)
        else:
            return F_loss
        
def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)

    return mixed_x, mixed_y

class Trainer:
    def __init__(self, 
                 model1,
                 model2,
                 logger,
                 n_splits=5,
                 seed=42,
                 device="cuda:0",
                 train_batch=32,
                 valid_batch=128,
                 kwargs={}):
        self.model1 = model1
        self.model2 = model2
        self.logger = logger
        self.device = device
        self.n_splits = n_splits
        self.seed = seed
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.kwargs = kwargs
        
        self.best_score = None
        self.tag = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.loss_fn = nn.BCELoss(reduction="mean").to(self.device)
        
        path = Path(f"bin1/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path1 = path
        path = Path(f"bin2/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path2 = path
        
    def fit(self, X, y, n_epochs=10):
        train_preds1 = np.zeros((len(X), num_classes_c))
        train_preds2 = np.zeros((len(X), num_classes_t))
        fold = KFold(n_splits=self.n_splits, random_state=self.seed)
        for i, (trn_idx, val_idx) in enumerate(fold.split(X)):
            self.fold_num = i
            self.logger.info(f"Fold {i + 1}")
            X_train, X_val = X[trn_idx, :], X[val_idx, :]
            y_train, y_val = y[trn_idx], y[val_idx]
            
            valid_preds1, valid_preds2 = self._fit(X_train, y_train, X_val, y_val, n_epochs)
            #print('tp1 ' + str(train_preds1.shape[1]))
            #print('vp1 ' + str(valid_preds1.shape[1]))
            train_preds1[val_idx] = valid_preds1
            train_preds2[val_idx] = valid_preds2
        return train_preds1, train_preds2
    
    def _fit(self, X_train, y_train, X_val, y_val, n_epochs):
        seed_torch(self.seed)
        train_dataset = IMetDataset(X_train, labels=y_train, device=self.device)
        train_loader = data.DataLoader(train_dataset, 
                                       batch_size=self.train_batch,
                                       shuffle=True)

        valid_dataset = IMetDataset(X_val, labels=y_val, device=self.device)
        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=self.valid_batch,
                                       shuffle=False)
        
        model1 = self.model1(**self.kwargs)
        model1.to(self.device)
        
        model2 = self.model2(**self.kwargs)
        model2.to(self.device)
        
        optimizer1 = optim.Adam(params=model1.parameters(), 
                                lr=0.0001)
        optimizer2 = optim.Adam(params=model2.parameters(), 
                                lr=0.0001)
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=n_epochs)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=n_epochs)
        best_score1 = np.inf
        best_score2 = np.inf
        mb = master_bar(range(n_epochs))
        for epoch in mb:
            model1.train()
            model2.train()
            avg_loss1 = 0.0
            avg_loss2 = 0.0
            for i_batch, y_batch in progress_bar(train_loader, parent=mb):
                #i_batch, y_batch = mixup(i_batch, y_batch, beta(1.0, 1.0))
                y_pred1 = model1(i_batch)
                y_pred2 = model2(i_batch)
                loss1 = self.loss_fn(y_pred1, y_batch[0])
                loss2 = self.loss_fn(y_pred2, y_batch[1])
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()
                avg_loss1 += loss1.item() / len(train_loader)
                avg_loss2 += loss2.item() / len(train_loader)
            valid_preds1, avg_val_loss1, valid_preds2, avg_val_loss2 = self._val(valid_loader, model1, model2)
            scheduler1.step()
            scheduler2.step()

            self.logger.info("=========================================")
            self.logger.info(f"Epoch {epoch + 1} / {n_epochs}")
            self.logger.info("=========================================")
            self.logger.info(f"avg_loss: {avg_loss1:.8f}")
            self.logger.info(f"avg_val_loss: {avg_val_loss1:.8f}")
            self.logger.info(f"avg_loss: {avg_loss2:.8f}")
            self.logger.info(f"avg_val_loss: {avg_val_loss2:.8f}")
            
            if best_score1 > avg_val_loss1:
                torch.save(model1.state_dict(),
                           self.path1 / f"1best{self.fold_num}.pth")
                self.logger.info(f"Save model at Epoch {epoch + 1}")
                best_score1 = avg_val_loss1
                
            if best_score2 > avg_val_loss2:
                torch.save(model2.state_dict(),
                           self.path2 / f"2best{self.fold_num}.pth")
                self.logger.info(f"Save model at Epoch {epoch + 1}")
                best_score2 = avg_val_loss2
                
        model1.load_state_dict(torch.load(self.path1 / f"1best{self.fold_num}.pth"))
        model2.load_state_dict(torch.load(self.path2 / f"2best{self.fold_num}.pth"))
        
        valid_preds1, avg_val_loss1, valid_preds2, avg_val_loss2 = self._val(valid_loader, model1, model2)
        #print('vpp'+str(valid_preds1.shape[1]))
        #self.logger.info(f"Best Validation Loss: {avg_val_loss:.8f}")
        return valid_preds1, valid_preds2
    
    def _val(self, loader, model1, model2):
        model1.eval()
        model2.eval()
        valid_preds1 = np.zeros((len(loader.dataset), num_classes_c))
        valid_preds2 = np.zeros((len(loader.dataset), num_classes_t))
        avg_val_loss1 = 0.0
        avg_val_loss2 = 0.0
        for i, (i_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                y_pred1 = model1(i_batch).detach()
                avg_val_loss1 += self.loss_fn(y_pred1, y_batch[0]).item() / len(loader)
                valid_preds1[i * self.valid_batch:(i + 1) * self.valid_batch] =                     y_pred1.cpu().numpy()
                y_pred2 = model2(i_batch).detach()
                avg_val_loss2 += self.loss_fn(y_pred2, y_batch[1]).item() / len(loader)
                valid_preds2[i * self.valid_batch:(i + 1) * self.valid_batch] =                     y_pred2.cpu().numpy()
        #print('vp1'+str(valid_preds1.shape[1]))
        return valid_preds1, avg_val_loss1, valid_preds2, avg_val_loss2
    
    def predict(self, X):
        #print('pred')
        dataset = IMetDataset(X, labels=None)
        loader = data.DataLoader(dataset, 
                                 batch_size=self.valid_batch, 
                                 shuffle=False)
        model1 = self.model1(**self.kwargs)
        model2 = self.model2(**self.kwargs)
        preds1 = np.zeros((X.size(0), num_classes_c))
        #print(list(self.path1.iterdir()))
        for path in self.path1.iterdir():
            with timer(f"Using {str(path)}", self.logger):
                model1.load_state_dict(torch.load(path))
                model1.to(self.device)
                model1.eval()
                temp1 = np.zeros_like(preds1)
                #print('try')
                for i, (i_batch, ) in enumerate(loader):
                    with torch.no_grad():
                        y_pred1 = model1(i_batch).detach()
                        #print(y_pred1[y_pred1 != 0])
                        temp1[i * self.valid_batch:(i + 1) * self.valid_batch] =                             y_pred1.cpu().numpy()
                preds1 += temp1 / self.n_splits
        preds2 = np.zeros((X.size(0), num_classes_t))
        for path in self.path2.iterdir():
            with timer(f"Using {str(path)}", self.logger):
                model2.load_state_dict(torch.load(path))
                model2.to(self.device)
                model2.eval()
                temp2 = np.zeros_like(preds2)
                for i, (i_batch, ) in enumerate(loader):
                    with torch.no_grad():
                        y_pred2 = model2(i_batch).detach()
                        temp2[i * self.valid_batch:(i + 1) * self.valid_batch] =                             y_pred2.cpu().numpy()
                preds2 += temp2 / self.n_splits
        return preds1, preds2


# In[15]:


trainer = Trainer(MultiLayerPerceptron1, MultiLayerPerceptron2, logger, train_batch=64, kwargs={})


# In[16]:


from sklearn.model_selection import train_test_split
y = train.attribute_ids.map(lambda x: x.split()).values
valid_preds1, valid_preds2 = trainer.fit(train_tensor, y, n_epochs=40)


# In[17]:


def threshold_search(y_pred, y_true):
    score = []
    candidates = np.arange(0, 1.0, 0.01)
    for th in progress_bar(candidates):
        yp = (y_pred > th).astype(int)
        score.append(fbeta_score(y_pred=yp, y_true=y_true, beta=2, average="samples"))
    score = np.array(score)
    pm = score.argmax()
    best_th, best_score = candidates[pm], score[pm]
    return best_th, best_score


# In[18]:


y_true = np.zeros((train.shape[0], 1103)).astype(int)
for i, row in enumerate(y):
    for idx in row:
        y_true[i, int(idx)] = 1


# In[19]:


best_threshold1, best_score1 = threshold_search(valid_preds1, targets_c)
best_score1
best_threshold2, best_score2 = threshold_search(valid_preds2, targets_t)
best_score2


# In[20]:


test_preds1, test_preds2  = trainer.predict(test_tensor)


# In[21]:


preds1 = (test_preds1 > best_threshold1).astype(int)
preds2 = (test_preds2 > best_threshold2).astype(int)


# In[22]:


prediction = []
for i in range(preds1.shape[0]):
    pred1 = [i for i in np.argwhere(preds1[i] == 1.0).reshape(-1).tolist() if i != (num_classes_c - 1)]
    pred2 = [(i + num_classes_c - 1) for i in np.argwhere(preds2[i] == 1.0).reshape(-1).tolist() if i != (num_classes_c + num_classes_t - 2)]
    pred_str = " ".join(list(map(str, pred1 + pred2)))
    prediction.append(pred_str)
#print(test_preds1[test_preds1 != 0])
sample.attribute_ids = prediction
sample.to_csv("submission.csv", index=False)
sample.head()


# In[23]:




