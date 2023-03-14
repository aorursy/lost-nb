#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
import numpy as np


# In[2]:


x = torch.zeros(5, 3)
print(x)


# In[3]:


x = torch.zeros(5, 3, 2)
print(x)


# In[4]:


x = torch.rand(5, 3)
print(x)


# In[ ]:





# In[5]:


a = torch.ones(5)
b = a.numpy()
print(b)


# In[6]:


a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# In[ ]:





# In[7]:


# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


# In[8]:


x = torch.ones(2, 2, requires_grad=True)
print(x)


# In[9]:


y = x + 2
print(y)


# In[10]:


print(y.grad_fn)


# In[11]:


z = y * y * 3
out = z.mean()

print(z, out)


# In[12]:


out.backward()


# In[13]:


print(x.grad) #Print gradients d(out)/dx


# In[14]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #not obligatory 
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


# In[15]:


params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


# In[16]:


input = torch.randn(1, 1, 32, 32)
print(input)


# In[17]:



out = net(input)
print(out)


# In[18]:


# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))


# In[19]:


output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# In[20]:


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# In[ ]:





# In[21]:


learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# In[22]:


import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


# In[ ]:





# In[23]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[24]:


DATA_FOLDER = '../input'
LABELS = f'{DATA_FOLDER}/train_labels.csv'
TRAIN_IMAGES_FOLDER = f'{DATA_FOLDER}/train'
USE_GPU = torch.cuda.is_available()


# In[25]:


def read_labels(path_to_file):
    labels = pd.read_csv(path_to_file)
    return labels


def format_labels_for_dataset(labels):
    return labels['label'].values.reshape(-1, 1)


def format_path_to_images_for_dataset(pd.DataFrame, path):
    return [os.path.join(path, f'{f}.tif') for f in labels['id'].values]


def train_valid_split(df):
    limit_df = 50000
    df = df.sample(n = df.shape[0])
    df = df.iloc[:limit_df]
    split = 40000
    train = df.iloc[:split]
    valid = df.iloc[:split]
    return train, valid


# In[26]:


class MainDataset(Dataset):
    def __init__(self,
                 x_dataset,
                 y_dataset,
                 x_tfms):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_tfms = x_tfms

    def __len__(self):
        return self.x_dataset.__len__()

    def __getitem__(self, index):
        x = self.x_dataset[index]
        y = self.y_dataset[index]
        if self.x_tfms is not None:
            x = self.x_tfms(x)
        return x, y
    
class ImageDataset(Dataset):
    def __init__(self, paths_to_imgs):
        self.paths_to_imgs = paths_to_imgs

    def __len__(self):
        return len(self.paths_to_imgs)

    def __getitem__(self, index):
        img = Image.open(self.paths_to_imgs[index])
        return img


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self) :
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]


# In[27]:


labels = read_labels(LABELS)
train, valid = train_valid_split(labels)

train_labels = format_labels_for_dataset(train)
valid_labels = format_labels_for_dataset(valid)

train_images = format_path_to_images_for_dataset(train, TRAIN_IMAGES_FOLDER)
valid_images = format_path_to_images_for_dataset(valid, TRAIN_IMAGES_FOLDER)

train_images_dataset = ImageDataset(train_images)
valid_images_dataset = ImageDataset(valid_images)
train_labels_dataset = LabelDataset(train_labels)
valid_labels_dataset = LabelDataset(valid_labels)

train_dataset = MainDataset(train_images_dataset, train_labels_dataset, x_tfms)
valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset, x_tfms)


# In[ ]:




