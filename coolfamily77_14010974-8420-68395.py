#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip uninstall --y kaggle')
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install kaggle==1.5.6')




get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle')
get_ipython().system('ls -lha kaggle.json')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')




get_ipython().system('kaggle competitions download -c 2020-ai-termproject-18011817')




get_ipython().system('unzip 2020-ai-termproject-18011817.zip')




import pandas as pd
import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as data
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler




device = torch.device('cuda')

torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 0.01
training_epochs = 6000
batch_size = 60




xy_train = pd.read_csv('train_seoul_grandpark.csv', header = None, skiprows=1, usecols=range(1, 8))

x_data = xy_train.iloc[: , 1:-1]
y_data = xy_train.iloc[: , [-1]]

x_data = np.array(x_data)
y_data = np.array(y_data)

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

x_train.shape




train_dataset = TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           drop_last = True)




class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)




linear1 = torch.nn.Linear(5, 8,bias=True)
linear2 = torch.nn.Linear(8, 8,bias=True)
linear3 = torch.nn.Linear(8, 8,bias=True)
linear4 = torch.nn.Linear(8, 8,bias=True)
linear5 = torch.nn.Linear(8, 1,bias=True)
#dropout = torch.nn.Dropout(p=drop_prob)
mish = Mish()




torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)
torch.nn.init.kaiming_normal_(linear4.weight)
torch.nn.init.kaiming_normal_(linear5.weight)

model = torch.nn.Sequential(linear1,mish,
                            linear2,mish,
                            linear3,mish,
                            linear4,mish,
                            linear5).to(device)




모델 학습




loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

losses = []
model_history = []
err_history = []

total_batch = len(data_loader)

for epoch in range(training_epochs + 1):
  avg_cost = 0
  #model.train()
  
  for X, Y in data_loader:
    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
    
  model_history.append(model)
  err_history.append(avg_cost)
  
  if epoch % 100 == 0:  
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')




best_model = model_history[np.argmin(err_history)]




xy_test = pd.read_csv('test_seoul_grandpark.csv', header = None, skiprows=1, usecols = range(1, 7))
x_data = xy_test.iloc[:, 1:]
x_data = np.array(x_data)
x_data = scaler.transform(x_data)
x_test = torch.FloatTensor(x_data).to(device)

with torch.no_grad():
    model.eval()     
    predict = best_model(x_test)




submit = pd.read_csv('submit_sample.csv')
submit['Expected'] = submit['Expected'].astype(float)
for i in range(len(predict)):
  submit['Expected'][i] = predict[i]
submit.to_csv('submit.csv', mode = 'w', index = False, header = True)
submit




get_ipython().system('kaggle competitions submit -c 2020-ai-termproject-18011817 -f submit.csv -m "14010974_이기택"')

