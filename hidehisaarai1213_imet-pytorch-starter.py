#!/usr/bin/env python
# coding: utf-8



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

torch.multiprocessing.set_start_method("spawn")




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




logger = get_logger(name="Main", tag="Pytorch-VGG16")




get_ipython().system('ls ../input/imet-2019-fgvc6/')




labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")
train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")
sample = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")
train.head()




get_ipython().system('cp ../input/pytorch-pretrained-image-models/* ./')
get_ipython().system('ls')




# This loader is to extract 1024d features from the images.
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




# This loader is to be used for serving image tensors for the MLP.
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
            for i in label:
                label_tensor[0, int(i)] = 1
            label_tensor = label_tensor.to(self.device)
            return [tensor, label_tensor]
        else:
            return [tensor]




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
    
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1103)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.sigmoid(self.linear2(x))




train_dataset = ImageDataLoader(
    root_dir=Path("../input/imet-2019-fgvc6/train/"),
    df=train,
    mode="train",
    transforms=data_transforms)
train_loader = data.DataLoader(dataset=train_dataset,
                               shuffle=False,
                               batch_size=128)
test_dataset = ImageDataLoader(
    root_dir=Path("../input/imet-2019-fgvc6/test/"),
    df=sample,
    mode="test",
    transforms=data_transforms)
test_loader = data.DataLoader(dataset=test_dataset,
                              shuffle=False,
                              batch_size=128)




def get_feature_vector(df, loader, device):
    matrix = torch.zeros((df.shape[0], 1024)).to(device)
    model = Densenet121("densenet121.pth")
    model.to(device)
    batch = loader.batch_size
    for i, (i_batch,) in tqdm(enumerate(loader)):
        i_batch = i_batch.to(device)
        pred = model(i_batch).detach()
        matrix[i * batch:(i + 1) * batch] = pred
    return matrix




train_tensor = get_feature_vector(train, train_loader, "cuda:0")
test_tensor = get_feature_vector(sample, test_loader, "cuda:0")




del train_dataset, train_loader
del test_dataset, test_loader
gc.collect()




class Trainer:
    def __init__(self, 
                 model, 
                 logger,
                 n_splits=5,
                 seed=42,
                 device="cuda:0",
                 train_batch=32,
                 valid_batch=128,
                 kwargs={}):
        self.model = model
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
        
        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path
        
    def fit(self, X, y, n_epochs=10):
        train_preds = np.zeros((len(X), 1103))
        fold = KFold(n_splits=self.n_splits, random_state=self.seed)
        for i, (trn_idx, val_idx) in enumerate(fold.split(X)):
            self.fold_num = i
            self.logger.info(f"Fold {i + 1}")
            X_train, X_val = X[trn_idx, :], X[val_idx, :]
            y_train, y_val = y[trn_idx], y[val_idx]
            
            valid_preds = self._fit(X_train, y_train, X_val, y_val, n_epochs)
            train_preds[val_idx] = valid_preds
        return train_preds
    
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
        
        model = self.model(**self.kwargs)
        model.to(self.device)
        
        optimizer = optim.Adam(params=model.parameters(), 
                                lr=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_score = np.inf
        mb = master_bar(range(n_epochs))
        for epoch in mb:
            model.train()
            avg_loss = 0.0
            for i_batch, y_batch in progress_bar(train_loader, parent=mb):
                y_pred = model(i_batch)
                loss = self.loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            valid_preds, avg_val_loss = self._val(valid_loader, model)
            scheduler.step()

            self.logger.info("=========================================")
            self.logger.info(f"Epoch {epoch + 1} / {n_epochs}")
            self.logger.info("=========================================")
            self.logger.info(f"avg_loss: {avg_loss:.8f}")
            self.logger.info(f"avg_val_loss: {avg_val_loss:.8f}")
            
            if best_score > avg_val_loss:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold_num}.pth")
                self.logger.info(f"Save model at Epoch {epoch + 1}")
                best_score = avg_val_loss
        model.load_state_dict(torch.load(self.path / f"best{self.fold_num}.pth"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        self.logger.info(f"Best Validation Loss: {avg_val_loss:.8f}")
        return valid_preds
    
    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros((len(loader.dataset), 1103))
        avg_val_loss = 0.0
        for i, (i_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(i_batch).detach()
                avg_val_loss += self.loss_fn(y_pred, y_batch).item() / len(loader)
                valid_preds[i * self.valid_batch:(i + 1) * self.valid_batch] =                     y_pred.cpu().numpy()
        return valid_preds, avg_val_loss
    
    def predict(self, X):
        dataset = IMetDataset(X, labels=None)
        loader = data.DataLoader(dataset, 
                                 batch_size=self.valid_batch, 
                                 shuffle=False)
        model = self.model(**self.kwargs)
        preds = np.zeros((X.size(0), 1103))
        for path in self.path.iterdir():
            with timer(f"Using {str(path)}", self.logger):
                model.load_state_dict(torch.load(path))
                model.to(self.device)
                model.eval()
                temp = np.zeros_like(preds)
                for i, (i_batch, ) in enumerate(loader):
                    with torch.no_grad():
                        y_pred = model(i_batch).detach()
                        temp[i * self.valid_batch:(i + 1) * self.valid_batch] =                             y_pred.cpu().numpy()
                preds += temp / self.n_splits
        return preds




trainer = Trainer(MultiLayerPerceptron, logger, train_batch=64, kwargs={})




y = train.attribute_ids.map(lambda x: x.split()).values
valid_preds = trainer.fit(train_tensor, y, n_epochs=40)




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




y_true = np.zeros((train.shape[0], 1103)).astype(int)
for i, row in enumerate(y):
    for idx in row:
        y_true[i, int(idx)] = 1




best_threshold, best_score = threshold_search(valid_preds, y_true)
best_score




test_preds = trainer.predict(test_tensor)




preds = (test_preds > best_threshold).astype(int)




prediction = []
for i in range(preds.shape[0]):
    pred1 = np.argwhere(preds[i] == 1.0).reshape(-1).tolist()
    pred_str = " ".join(list(map(str, pred1)))
    prediction.append(pred_str)
    
sample.attribute_ids = prediction
sample.to_csv("submission.csv", index=False)
sample.head()






