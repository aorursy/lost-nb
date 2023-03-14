#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

from os import listdir, makedirs
from os.path import join, exists, expanduser

from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler


# In[ ]:


ls ../input


# In[ ]:


ls ../working


# In[ ]:


INPUT_SIZE = 224
NUM_CLASSES = 16
data_dir = '../input/'
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print(len(listdir(join(data_dir, 'train'))), len(labels))
print(len(listdir(join(data_dir, 'test'))), len(sample_submission))


# In[ ]:


selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

train = labels_pivot.sample(frac=0.8)
valid = labels_pivot[~labels_pivot['id'].isin(train['id'])]
print(train.shape, valid.shape)


# In[ ]:


class DogsDataset(Dataset):
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname)
        labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
        labels = np.argmax(labels)
        if self.transform:
            image = self.transform(image)
        return [image, labels]


# In[ ]:


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
ds_trans = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])
train_ds = DogsDataset(train, data_dir+'train/', transform=ds_trans)
valid_ds = DogsDataset(valid, data_dir+'train/', transform=ds_trans)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=1)


# In[ ]:


#create cnn class
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DBCNN(torch.nn.Module) :
    def __init__(self):
        super(DBCNN, self).__init__()
        self.baseblock = Bottleneck
        
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False), 
                     nn.BatchNorm2d(64),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                     )
        
        downsample = None
        downsample = nn.Sequential(
                     nn.Conv2d(64, 64*self.baseblock.expansion, kernel_size=1, stride=1, bias=False),
                     nn.BatchNorm2d(64*self.baseblock.expansion)
                     )
        self.layer1 = nn.Sequential(
                      self.baseblock(64, 64, 1, downsample),
                      self.baseblock(256, 64),
                      self.baseblock(256, 64)
                      )
        
        downsample = nn.Sequential(
                     nn.Conv2d(256, 128*self.baseblock.expansion, kernel_size=1, stride=2, bias=False),
                     nn.BatchNorm2d(128*self.baseblock.expansion)
                     )
        self.layer2 = nn.Sequential(
                      self.baseblock(256, 128, 2, downsample),
                      self.baseblock(512, 128),
                      self.baseblock(512, 128),            
                      self.baseblock(512, 128)
                      )
        
        downsample = nn.Sequential(
                     nn.Conv2d(512, 256*self.baseblock.expansion, kernel_size=1, stride=2, bias=False),
                     nn.BatchNorm2d(256*self.baseblock.expansion)
                     )
        self.layer3 = nn.Sequential(
                      self.baseblock(512, 256, 2, downsample),
                      self.baseblock(1024, 256),
                      self.baseblock(1024, 256),            
                      self.baseblock(1024, 256),            
                      self.baseblock(1024, 256),            
                      self.baseblock(1024, 256)
                      )
        
        downsample = nn.Sequential(
                     nn.Conv2d(1024, 512*self.baseblock.expansion, kernel_size=1, stride=2, bias=False),
                     nn.BatchNorm2d(512*self.baseblock.expansion)
                     )
        self.layer4 = nn.Sequential(
                      self.baseblock(1024, 512, 2, downsample),
                      self.baseblock(2048, 512),
                      self.baseblock(2048, 512)
                      )
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*self.baseblock.expansion, 1000)
 
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # 展平为
        x = self.fc(x)
        return x


# In[ ]:


dbcnn = DBCNN()
#if exists("../working/dbcnn.pkl"):
#    print("load model")
#    dbcnn = torch.load('dbcnn.pkl')
    
#dbcnn = models.resnet50(pretrained=False)
#print(dbcnn)


# In[ ]:


def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}
    print(dataset_sizes)

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0] * labels.size()[0]
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]
                
            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        if (epoch + 1)%100 == 0 :
            print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
              'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch + 1, num_epochs,
                train_epoch_loss, train_epoch_acc, 
                valid_epoch_loss, valid_epoch_acc))
            
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


# In[ ]:


md = dbcnn
use_gpu = torch.cuda.is_available()
# freeze all model parametersstart_time = time.time()
for param in md.parameters():
    param.requires_grad = False

# new final layer with 16 classes
num_ftrs = md.fc.in_features
md.fc = torch.nn.Linear(num_ftrs, 16)
if use_gpu:
    md = md.cuda()

criterion = torch.nn.CrossEntropyLoss()

parameters = filter(lambda p: p.requires_grad, md.parameters())

optimizer = torch.optim.Adam(parameters, lr=0.001, betas=(0.9, 0.99))
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

dloaders = {'train':train_dl, 'valid':valid_dl}


# In[ ]:


start_time = time.time()
model = train_model(dloaders, md, criterion, optimizer, exp_lr_scheduler, num_epochs=1000)
print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

torch.save(model, 'dbcnn.pkl')


# In[ ]:


ls ../working


# In[ ]:


def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.129, 0.124, 0.125])
    #change 0.229,0.224,0.225 to 0.129,0.124,0.125
    #sometimes can not show image
    inp = std * inp + mean
    axis.imshow(inp)


# In[ ]:


def getBreedName(index):
    if index >= 0 and index < len(selected_breed_list) :
        return selected_breed_list[index]
    return "unknown"


# In[ ]:


def visualize_model(dataloders, model, num_images=16):
    cnt = 0
    fig = plt.figure(1, figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.05)
    for i, (inputs, labels) in enumerate(dataloders['valid']):
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            ax = grid[cnt]
            imshow(ax, inputs.cpu().data[j])
            ax.text(10, 210, 'P:{}/R:{}'.format(getBreedName(preds[j]), getBreedName(labels.data[j])), 
                    color='k', backgroundcolor='w', alpha=0.8)
            cnt += 1
            if cnt == num_images:
                return


# In[ ]:


visualize_model(dloaders, dbcnn)


# In[ ]:




