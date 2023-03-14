#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import json
import torch
from sklearn.decomposition import PCA


# In[ ]:


train_csv = pd.read_csv('./moral_TFT_train.csv')
test_csv = pd.read_csv('./moral_TFT_test.csv')


# In[ ]:


train_csv.head()


# In[ ]:


train_csv.iloc[:,1:52]


# In[ ]:


pca_cham = PCA(3)
pca_cham.fit(train_csv.iloc[:,1:52], train_csv.Ranked)


# In[ ]:


cham = pca_cham.transform(train_csv.iloc[:,1:52])


# In[ ]:


test_csv.iloc[:,:51]


# In[ ]:


test_cham = pca_cham.transform(test_csv.iloc[:,:51])


# In[ ]:


pca_comb = PCA(3)
pca_comb.fit(train_csv.iloc[:,52:], train_csv.Ranked)
comb = pca_comb.transform(train_csv.iloc[:,52:])


# In[ ]:


test_comb= pca_comb.transform(test_csv.iloc[:,51:])


# In[ ]:


train_D = pd.concat([pd.DataFrame(cham),pd.DataFrame(comb)], axis = 1)


# In[ ]:


train_D = pd.concat([train_D, train_csv.drop(columns = 'Ranked', axis = 1)], axis = 1)


# In[ ]:


train_L = train_csv.Ranked


# In[ ]:


test_D = pd.concat([pd.DataFrame(test_cham), pd.DataFrame(test_comb)], axis = 1)


# In[ ]:


test_D = pd.concat([test_D, test_csv], axis= 1)


# In[ ]:


# train_D = train_csv.drop('Ranked', axis = 1)
# test_D = test_csv


# In[ ]:


train_D.shape


# In[ ]:


train_D = torch.FloatTensor(np.array(train_D))
train_L = torch.FloatTensor(np.array(train_L))
test_D = torch.FloatTensor(np.array(test_D))


# In[ ]:


data_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_D, train_L),
                                          batch_size=100,
                                          shuffle=True,
                                          drop_last=True)


# In[ ]:


linear1= torch.nn.Linear(train_D.shape[1], 512, bias = True)
linear2 = torch.nn.Linear(512,512,bias = True)
linear3 = torch.nn.Linear(512,512,bias = True)
linear4 = torch.nn.Linear(512,512,bias = True)
linear5 = torch.nn.Linear(512,1, bias=  True)
relu = torch.nn.PReLU()
sigmoid = torch.nn.Sigmoid()
dropout = torch.nn.Dropout(p = 0.3)


# In[ ]:


torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)


# In[ ]:


import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


# In[ ]:


model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3, relu, dropout,
                            linear4, relu, dropout,
                            linear5,sigmoid).to(device)


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss =torch.nn.BCELoss()#CrossEntropyLoss()


# In[ ]:


total_batch = len(data_loader)
model.train()


for e in range(20):
  avg_cost= 0
  for x, y in data_loader:
    x = x.to(device)
    y=  y.to(device)
    optimizer.zero_grad()
    h_x = model(x)
    cost = loss(h_x, y)
    cost.backward()
    optimizer.step()
    avg_cost += cost / total_batch
  print('Epoch {}'.format(e), 'cost {}'.format(avg_cost))


# In[ ]:


with torch.no_grad():
  model.eval()
  pred=  model(test_D.to(device))


# In[ ]:


pred


# In[ ]:


real_pred= []
for i in range(3300):
  real_pred.append(int(torch.round(pred[i]).item()))


# In[ ]:


real_pred


# In[ ]:


result ={}
result['id'] = list(i for i in range(3300))
result['result'] = real_pred#torch.argmax(pred,1)


# In[ ]:


pd.DataFrame(result).to_csv('baseline4.csv',index= False)


# In[ ]:


pd.read_csv('baseline4.csv')

