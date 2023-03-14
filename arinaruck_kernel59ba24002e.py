#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm_notebook
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from IPython.display import clear_output
from scipy.special import expit

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from gensim import models

import re
from collections import Counter
import gensim
import heapq
from operator import itemgetter
from multiprocessing import Pool


# In[3]:


#data = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
#data.head()
train = pd.read_csv('../input/kernel7d1c9fd560/processed_train.csv')
test = pd.read_csv('../input/kernel7d1c9fd560/processed_test.csv')
train = train.fillna('nan')
test = test.fillna('nan')
display(train.head())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


train.rename(columns={'no_misspels' : 'no_misspells'}, inplace=True)
test.rename(columns={'no_misspels' : 'no_misspells'}, inplace=True)


# In[5]:


MAX_THRESHOLD = 0.21
train['len'] = train['basic'].str.split().apply(len)
X, y = train['basic'].str.split().to_numpy(), train['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)


# In[6]:


word2vec_path = '../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
words = word2vec.index2word


# In[7]:


plt.hist(train['len'], bins=100);
lens = np.array(train['len'])
np.quantile(lens, 0.99)


# In[8]:


#https://discuss.pytorch.org/t/vanishing-gradients/46824/5
def plot_grad_flow(named_parameters, title):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            try:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                
            except:
                print(n)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(title)
    plt.show()


# In[9]:


def make_batch(data):
    dim=300
    vectorized = [torch.Tensor([word2vec[word] if word in word2vec else np.random.rand(dim) for word in sen])
                  for sen in data]
    batch = torch.Tensor(pad_sequence(vectorized, batch_first=True))
    return batch


# In[10]:


# Discriminator receives 1x28x28 image and returns a float number
# we can name each layer using OrderedDict

class CNN(nn.Module):
    def __init__(self, dim=300):
        super(CNN,self).__init__()
        self.global_pool_1 = nn.AdaptiveMaxPool2d((1, 1))
        self.global_pool_2 = nn.AdaptiveMaxPool2d((2, 1))
        self.conv1 = nn.Sequential(
                     nn.Conv2d(1, 128, kernel_size=(1, 300)), 
                     nn.BatchNorm2d(128),
                     nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(1, 128, kernel_size=(2, 300), padding=(1, 0)), 
                     nn.BatchNorm2d(128),
                     nn.LeakyReLU()
        )
        self.conv2_dilation = nn.Sequential(
                     nn.Conv2d(1, 128, kernel_size=(2, 300), padding=(1, 0), dilation=(2, 1)), 
                     nn.BatchNorm2d(128),
                     nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
                     nn.Conv2d(1, 128, kernel_size=(3, 300), padding=(1, 0)), 
                     nn.BatchNorm2d(128),
                     nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
                     nn.Conv2d(1, 128, kernel_size=(4, 300), padding=(2, 0)), 
                     nn.BatchNorm2d(128),
                     nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
                        nn.Conv2d(1, 128, kernel_size=(5, 300), padding=(2, 0)),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU()
        )
        
        self.bottleneck = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.BatchNorm1d(128),
                        nn.Dropout(0.5),
                        nn.LeakyReLU()
        )
        
        self.fc2 = nn.Sequential(
                        nn.Linear(128, 32),
                        nn.BatchNorm1d(32),
                        nn.Dropout(0.5),
                        nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
                        nn.Linear(32, 1)
        )
        


    def forward(self,x):
        conv1 = self.global_pool_2(self.conv1(x)) #batch x 64 x 2
        conv2 = self.global_pool_2(self.conv2(x)) #batch x 64  x 2
        conv2_dilation = self.global_pool_1(self.conv2_dilation(x)) #batch x 64 x 1
        conv3 = self.global_pool_1(self.conv3(x)) #batch x 64 x 1
        conv4 = self.global_pool_1(self.conv4(x)) #batch x 64 x 1
        conv5 = self.global_pool_1(self.conv5(x)) #batch x 64 x 1
        concatenated = torch.cat((conv1, conv2, conv2_dilation, conv3, conv4, conv5), 2) #batch x 64 x 8
        res = self.bottleneck(concatenated) #batch x 64 x 8
        res = res.view(res.shape[0], -1) #batch x 256
        res = self.fc1(res)
        res = self.fc2(res)
        res = self.fc3(res)
        return res


# In[11]:


cnn = CNN().to(device)
cnn.load_state_dict(torch.load('/kaggle/input/quora-models/model2.ckpt'))
opt = torch.optim.Adam(cnn.parameters(), lr=0.1)
criterion = nn.BCEWithLogitsLoss()


# In[12]:


def train_n_test(epochs, data_train, target_train, data_test, target_test, batch_size=128):
    batch_num = len(data_train) // batch_size
    for epoch in tqdm_notebook(range(epochs)):
        cnn.train()
        losses = []
        data_train, target_train = shuffle(data_train, target_train)
        for i in tqdm_notebook(range(batch_num)):
            opt.zero_grad()
            batch = make_batch(data_train[i * batch_size : (i + 1) * batch_size]).to(device)
            s = batch.shape
            batch = batch.view(s[0], 1, s[1], s[2])
            batch_y = torch.Tensor(target_train[i * batch_size : (i + 1) * batch_size]).to(device)
            predict = cnn(batch).view(-1)
            loss = criterion(predict, batch_y)
            losses.append(loss)
            loss.backward()
            opt.step()
            #plot_grad_flow(cnn.named_parameters(), 'cnn')
            if i % 1000 == 0:
                print("Epoch: {}, i: {}, loss: {}".format(epoch, i, loss))
        cnn.eval()
        data_test, target_test = shuffle(data_test, target_test)
        batch_size_test = 1024
        batch_test_num = len(data_test) // batch_size_test
        predictions = np.array([])
        for i in tqdm_notebook(range(batch_test_num)):
            batch_test = make_batch(data_test[i * batch_size_test : (i + 1) * batch_size_test]).to(device)
            s = batch_test.shape
            batch_test = batch_test.view(s[0], 1, s[1], s[2])
            predict = cnn(batch_test).view(-1)
            if predictions.size == 0:
                predictions = predict.cpu().detach().numpy()
            else:
                predictions = np.hstack((predictions, predict.view(predict.shape[0]).cpu().detach().numpy()))
        average_precision = average_precision_score(target_test[: len(predictions)], predictions)
        plt.hist(expit(predictions[target_test[: len(predictions)] == 0]), alpha=0.2, color='c')
        plt.hist(expit(predictions[target_test[: len(predictions)] != 0]), alpha=0.2, color='b')
        plt.show()
        f1 = f1_score(target_test[: len(predictions)], (expit(predictions) >= MAX_THRESHOLD).astype(int))
        print("Epoch num: {}, average_precision: {}, f1: {}".format(epoch, average_precision, f1), flush=True)


# In[13]:


batch_size_test = 64
batch_test_num = len(X_test) // batch_size_test
predictions = np.array([])
for i in tqdm_notebook(range(batch_test_num)):
    batch_test = make_batch(X_test[i * batch_size_test : (i + 1) * batch_size_test]).to(device)
    s = batch_test.shape
    batch_test = batch_test.view(s[0], 1, s[1], s[2])
    predict = cnn(batch_test).view(-1)
    if predictions.size == 0:
        predictions = predict.cpu().detach().numpy()
    else:
        predictions = np.hstack((predictions, predict.view(predict.shape[0]).cpu().detach().numpy()))
average_precision = average_precision_score(y_test[: len(predictions)], predictions)
thresholds = np.linspace(0, 1, 100)
max_f1, max_threshold = 0, 0 
for threshold in thresholds:
    f1 = f1_score(y_test[: len(predictions)], (expit(predictions) >= threshold).astype(int))
    if f1 > max_f1:
        max_f1, max_threshold = f1, threshold
print(max_f1, max_threshold)


# In[14]:


X_real_test = test['basic'].str.split().to_numpy()
test_size, batch_size = len(X_real_test), 256
batch_size_num = test_size // batch_size


# In[15]:


predictions = np.array([])
for i in tqdm_notebook(range(batch_size_num)):
    batch_test = make_batch(X_real_test[i * batch_size : (i + 1) * batch_size]).to(device)
    s = batch_test.shape
    batch_test = batch_test.view(s[0], 1, s[1], s[2])
    predict = cnn(batch_test).view(-1)
    if predictions.size == 0:
        predictions = predict.cpu().detach().numpy()
    else:
        predictions = np.hstack((predictions, predict.view(predict.shape[0]).cpu().detach().numpy()))
batch_test = make_batch(X_real_test[batch_size_num * batch_size : ]).to(device)
s = batch_test.shape
batch_test = batch_test.view(s[0], 1, s[1], s[2])
predict = cnn(batch_test).view(-1)
predictions = np.hstack((predictions, predict.view(predict.shape[0]).cpu().detach().numpy()))
res = expit(predictions) >= threshold


# In[16]:


test.head()


# In[17]:


submission = pd.DataFrame({'qid' : test['qid'], 'prediction' : res})
print(len(submission.index))
submission.to_csv('submission.csv')


# In[18]:


'''
train_size = len(X_train)
train_n_test(1, X_train[: train_size // 2], y_train[: train_size // 2], X_test, y_test)
torch.cuda.empty_cache()
train_n_test(1, X_train[train_size // 2 :], y_train[train_size // 2 :], X_test, y_test)
torch.save(cnn.state_dict(), 'model3.ckpt')
''';

