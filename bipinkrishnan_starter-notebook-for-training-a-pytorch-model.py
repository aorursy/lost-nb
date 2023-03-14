#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')


# In[ ]:


import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm.notebook import tqdm
import warnings
warnings.simplefilter('ignore')


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = '../input/siim-isic-melanoma-classification/jpeg/train/'
test_path = '../input/siim-isic-melanoma-classification/jpeg/test/'

data = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df_test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
malignant = data.iloc[data[data['target'] == 1].index]
benign = data.iloc[(data[data['target'] == 0].index)[:586]]
df = pd.concat([benign, malignant], axis=0)
df['target'].value_counts()


# In[ ]:


df_test.head()


# In[ ]:


def create_meta(df):
    create_meta.dic = {}
    
    for i, j in enumerate(df['anatom_site_general_challenge'].unique()):
      create_meta.dic.update({j:i})
    
    sex = np.array(df['sex'].map({'female': 0, 'male': 1}).fillna(-1)).reshape(-1, 1)
    age = np.array(df['age_approx'].fillna(-1)).reshape(-1, 1)
    part = np.array(pd.get_dummies(df['anatom_site_general_challenge'].map(create_meta.dic))).reshape(-1, 7)
    
    return np.concatenate((sex, age, part), axis=1)


# In[ ]:


class LoadDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_path: str, train: bool=True, transform=None, meta_features=None):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.train = train
        self.transform = transform
        self.meta = meta_features
        
    def __getitem__(self, idx):
        img_file = os.path.join(self.img_path, self.df['image_name'][idx] + '.jpg')
        img = Image.open(img_file)
        
        if self.transform:
            img = self.transform(img)
        
        if self.train:
            return (img, self.meta[idx]), self.df.iloc[idx]['target']
        else:
            return (img, self.meta[idx])
        
    def __len__(self):
        return len(self.df)
            


# In[ ]:


class Model(nn.Module):
    def __init__(self, arch, n_meta: int):
        super().__init__()
        self.arch = arch
        self.n_meta = n_meta
        
        self.arch._fc = nn.Linear(self.arch._fc.in_features, 500)
        
        self.meta = nn.Sequential(nn.Linear(self.n_meta, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        
        self.output = nn.Linear(500+250, 1)
        
    def forward(self, inputs):
        img, meta = inputs
        
        cnn_out = self.arch(img)
        meta_out = self.meta(meta)
        
        features = torch.cat((cnn_out, meta_out), dim=1)
        out = self.output(features)
        
        return out


# In[ ]:


train_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_df = create_meta(df)
test_df = create_meta(df_test)


# In[ ]:


gkf = GroupKFold(5)
cnn = EfficientNet.from_pretrained('efficientnet-b1')
cnn = cnn.to(device)
for params in cnn.parameters():
    params.requires_grad = False

model = Model(cnn, train_df.shape[1])
model = model.to(device)

for params in model.parameters():
    if params.requires_grad == True:
        print(params.shape)

next(model.parameters()).is_cuda


# In[ ]:


opt = optim.Adam(model.parameters(), 0.001)
criterion = nn.BCEWithLogitsLoss()

bs = 4
epochs = 12
es_patience = 3
correct = 0.0
best_roc = 0

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                                    optimizer=opt, 
                                                    mode='max', 
                                                    patience=1, 
                                                    verbose=True, 
                                                    factor=0.2
)


# In[ ]:


for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['target'], df['patient_id']), 1):
    print("Fold", fold)
    
    cnn = EfficientNet.from_pretrained('efficientnet-b1')
    cnn = cnn.to(device)
    for params in cnn.parameters():
        params.requires_grad = False

    model = Model(cnn, train_df.shape[1])
    model = model.to(device)

    trainset = LoadDataset(df.iloc[train_idx].reset_index(), train_path, True, train_transform, train_df)
    valset = LoadDataset(df.iloc[val_idx].reset_index(), train_path, True, train_transform, train_df)
    testset = LoadDataset(df_test, test_path, False, test_transform, test_df)
    
    trainloader = DataLoader(trainset, bs, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, bs, shuffle=False, num_workers=2)   
    testloader = DataLoader(testset, bs, shuffle=False, num_workers=2)
    
    for epoch in range(epochs):
        model.train()
        
        for (img, meta_feature), label in tqdm(trainloader, total=len(trainloader)):
            img = img.to(device)
            meta = meta_feature.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)
            
            opt.zero_grad()
            out = model((img, meta))
            loss = criterion(out, label.unsqueeze(1))
            loss.backward()
            
            opt.step()
            
            pred = torch.sigmoid(out)
            correct += (pred.cpu()==label.cpu().unsqueeze(1)).sum().item()  
            
        train_acc = correct / len(trainloader)
        
        #validaton step
        model.eval()
        with torch.no_grad():
            val_pred = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
            
            for idx, ((img_val, meta_feature_val), label_val) in enumerate(valloader):
                img_val = img_val.to(device)
                meta_val = meta_feature_val.type(torch.float32).to(device)
                label_val = label_val.type(torch.float32).to(device)
                
                out_val = torch.sigmoid(model((img_val, meta_val)))
                loss_val = criterion(out_val, label_val.unsqueeze(1))
                val_pred[idx*valloader.batch_size: idx*valloader.batch_size+valloader.batch_size] = out_val
                
      
            roc_val = roc_auc_score(df.iloc[val_idx]['target'], val_pred.cpu())
                
                
            print("Epoch: {}/{}  train_loss: {:0.4f}  val_loss: {:0.4f}  roc: {:0.4f}".format(
                epoch+1, epochs, loss.item(), loss_val.item(), roc_val))
               
            scheduler.step(roc_val)
            
            if roc_val >= best_roc:
                save_path = f'model_{fold}.pth'
                best_roc = roc_val
                patience = es_patience
                torch.save(model, save_path)
#             else:
#                 patience -= 1
                
#         if patience == 0:
#             print("Early stopping\tBest roc score: {:0.3f}".format(best_roc))
#             break
        
        model = torch.load(save_path)
        model.eval()   


# In[ ]:




