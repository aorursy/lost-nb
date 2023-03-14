#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join, exists, expanduser
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation,     Input, merge, Lambda


# In[ ]:


use_cuda = torch.cuda.is_available()


# In[ ]:


get_ipython().system('ls ../input/dogsdata/data/data')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'AlexNet.py', "import torch.nn as nn\nimport torch.nn.functional as F\nimport numpy as np\n\ndef conv_init(m):\n    classname = m.__class__.__name__\n    if classname.find('Conv') != -1:\n        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))\n        nn.init.normal_(m.weight, mean=0, std=1)\n        nn.init.constant(m.bias, 0)\n\nclass AlexNet(nn.Module):\n\n    def __init__(self, num_classes, inputs=3):\n        super(AlexNet, self).__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(inputs, 64, kernel_size=11, stride=4, padding=5),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(kernel_size=2, stride=2),\n            nn.Conv2d(64, 192, kernel_size=5, padding=2),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(kernel_size=2, stride=2),\n            nn.Conv2d(192, 384, kernel_size=3, padding=1),\n            nn.ReLU(inplace=True),\n            nn.Conv2d(384, 256, kernel_size=3, padding=1),\n            nn.ReLU(inplace=True),\n            nn.Conv2d(256, 256, kernel_size=3, padding=1),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(kernel_size=2, stride=2),\n        )\n        self.classifier = nn.Linear(256, num_classes)\n\n    def forward(self, x):\n        x = self.features(x)\n        x = x.view(x.size(0), -1)\n        x = self.classifier(x)\n        return x")


# In[ ]:


get_ipython().run_line_magic('load', 'AlexNet.py')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class AlexNet(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# In[ ]:


get_ipython().run_cell_magic('writefile', 'train_test.py', 'import torch\nimport torch.optim as optim\nfrom torch.autograd import Variable\nimport config as cf\nimport time\nimport numpy as np\n\nuse_cuda = torch.cuda.is_available()\n\nbest_acc = 0\n\ndef train(epoch, net, trainloader, criterion):\n    net.train()\n    train_loss = 0\n    correct = 0\n    total = 0\n    optimizer = optim.Adam(net.parameters(), lr=cf.lr, weight_decay=5e-4)\n    train_loss_stacked = np.array([0])\n\n    print(\'\\n=> Training Epoch #%d, LR=%.4f\' %(epoch, cf.lr))\n    for batch_idx, (inputs_value, targets) in enumerate(trainloader):\n        if use_cuda:\n            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings\n        optimizer.zero_grad()\n        inputs_value, targets = Variable(inputs_value), Variable(targets)\n        outputs = net(inputs_value)               # Forward Propagation\n        loss = criterion(outputs, targets)  # Loss\n        loss.backward()  # Backward Propagation\n        optimizer.step() # Optimizer update\n\n        train_loss += loss.data[0]\n        _, predicted = torch.max(outputs.data, 1)\n        total += targets.size(0)\n        correct += predicted.eq(targets.data).cpu().sum()\n        train_loss_stacked = np.append(train_loss_stacked, loss.data[0].cpu().numpy())\n    print (\'| Epoch [%3d/%3d] \\t\\tLoss: %.4f Acc@1: %.3f%%\'\n                %(epoch, cf.num_epochs, loss.data[0], 100.*correct/total))\n\n    return train_loss_stacked\n\n\ndef test(epoch, net, testloader, criterion):\n    global best_acc\n    net.eval()\n    test_loss = 0\n    correct = 0\n    total = 0\n    test_loss_stacked = np.array([0])\n    for batch_idx, (inputs_value, targets) in enumerate(testloader):\n        if use_cuda:\n            inputs_value, targets = inputs_value.cuda(), targets.cuda()\n        with torch.no_grad():\n            inputs_value, targets = Variable(inputs_value), Variable(targets)\n        outputs = net(inputs_value)\n        loss = criterion(outputs, targets)\n\n        test_loss += loss.data[0]\n        _, predicted = torch.max(outputs.data, 1)\n        total += targets.size(0)\n        correct += predicted.eq(targets.data).cpu().sum()\n        test_loss_stacked = np.append(test_loss_stacked, loss.data[0].cpu().numpy())\n\n\n    # Save checkpoint when best model\n    acc = 100. * correct / total\n    print("\\n| Validation Epoch #%d\\t\\t\\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.data[0], acc))\n\n\n\n    if acc > best_acc:\n        best_acc = acc\n    print(\'* Test results : Acc@1 = %.2f%%\' % (best_acc))\n\n    return test_loss_stacked\n\ndef start_train_test(net,trainloader, testloader, criterion):\n    elapsed_time = 0\n\n    for epoch in range(cf.start_epoch, cf.start_epoch + cf.num_epochs):\n        start_time = time.time()\n\n        train_loss = train(epoch, net, trainloader, criterion)\n        test_loss = test(epoch, net, testloader, criterion)\n\n        epoch_time = time.time() - start_time\n        elapsed_time += epoch_time\n        print(\'| Elapsed time : %d:%02d:%02d\' % (get_hms(elapsed_time)))\n\n    return train_loss.tolist(), test_loss.tolist()\n\ndef get_hms(seconds):\n    m, s = divmod(seconds, 60)\n    h, m = divmod(m, 60)\n\n    return h, m, s')


# In[ ]:


get_ipython().run_line_magic('load', 'train_test.py')
import torch
import torch.optim as optim
from torch.autograd import Variable
import config as cf
import time
import numpy as np

use_cuda = torch.cuda.is_available()

best_acc = 0

def train(epoch, net, trainloader, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=cf.lr, weight_decay=5e-4)
    train_loss_stacked = np.array([0])

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.lr))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_loss_stacked = np.append(train_loss_stacked, loss.data[0].cpu().numpy())
    print ('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, cf.num_epochs, loss.data[0], 100.*correct/total))

    return train_loss_stacked


def test(epoch, net, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_stacked = np.array([0])
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()
        with torch.no_grad():
            inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss_stacked = np.append(test_loss_stacked, loss.data[0].cpu().numpy())


    # Save checkpoint when best model
    acc = 100. * correct / total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.data[0], acc))



    if acc > best_acc:
        best_acc = acc
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))

    return test_loss_stacked

def start_train_test(net,trainloader, testloader, criterion):
    elapsed_time = 0

    for epoch in range(cf.start_epoch, cf.start_epoch + cf.num_epochs):
        start_time = time.time()

        train_loss = train(epoch, net, trainloader, criterion)
        test_loss = test(epoch, net, testloader, criterion)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    return train_loss.tolist(), test_loss.tolist()

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


# In[ ]:


from train_test import start_train_test
from AlexNet import AlexNet


# In[ ]:


get_ipython().run_cell_magic('writefile', 'config.py', "start_epoch = 1\nnum_epochs = 10\nbatch_size = 256\noptim_type = 'Adam'\nresize=32\nlr=0.001\n\nmean = {\n    'cifar10': (0.4914, 0.4822, 0.4465),\n    'cifar100': (0.5071, 0.4867, 0.4408),\n    'mnist': (0.1307,),\n    'stl10': (0.485, 0.456, 0.406),\n}\n\nstd = {\n    'cifar10': (0.2023, 0.1994, 0.2010),\n    'cifar100': (0.2675, 0.2565, 0.2761),\n    'mnist': (0.3081,),\n    'stl10': (0.229, 0.224, 0.225),\n}")


# In[ ]:


get_ipython().run_line_magic('load', 'config.py')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'transform.py', 'import torchvision.transforms as transforms\n\nimport config as cf\n\ndef transform_training():\n\n    transform_train = transforms.Compose([\n        transforms.Resize((cf.resize, cf.resize)),\n        transforms.RandomCrop(32, padding=4),\n        # transforms.RandomHorizontalFlip(),\n        # CIFAR10Policy(),\n        transforms.ToTensor(),\n    ])  # meanstd transformation\n\n    return transform_train\n\ndef transform_testing():\n\n    transform_test = transforms.Compose([\n        transforms.Resize((cf.resize, cf.resize)),\n        transforms.RandomCrop(32, padding=4),\n        # transforms.RandomHorizontalFlip(),\n        # CIFAR10Policy(),\n        transforms.ToTensor(),\n    ])\n\n    return transform_test')


# In[ ]:


get_ipython().run_line_magic('load', 'transform.py')
import torchvision.transforms as transforms

import config as cf

def transform_training():

    transform_train = transforms.Compose([
        transforms.Resize((cf.resize, cf.resize)),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
    ])  # meanstd transformation

    return transform_train

def transform_testing():

    transform_test = transforms.Compose([
        transforms.Resize((cf.resize, cf.resize)),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
    ])

    return transform_test


# In[ ]:


import torch
import torchvision
import sys
from transform import transform_training, transform_testing
import torchvision.datasets as datasets
import config as cf
import os
data_path = '../input/dogsdata/data/data'
import torchvision.transforms as transforms
def dataset(dataset_name):
    if(dataset_name  == 'dog-breed'):
         print("| Preparing dog-breed dataset...")
         trainset = datasets.ImageFolder(os.path.join(data_path, 'train'),transform_training())
         testset = datasets.ImageFolder(os.path.join(data_path, 'test'),transform=transform_testing())                  
         outputs = 16
         inputs = 3
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader, outputs, inputs


# In[ ]:


data_dir = '../input/dog-breed-identification'
INPUT_SIZE = 224
NUM_CLASSES = 16
SEED = 1987
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))


# In[ ]:


selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
group = labels.groupby(by='breed', as_index=False).agg({'id': pd.Series.nunique})

group = group.sort_values('id',ascending=False)

print(group)

labels['rank'] = group['breed']

labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]


# In[ ]:


trainloader, testloader, outputs, inputs = dataset('dog-breed')
print ('Output classes: {}\nInput channels: {}'.format(outputs, inputs))


# In[ ]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputdata, classes = next(iter(trainloader))
# Make a grid from batch
out = torchvision.utils.make_grid(inputdata)

imshow(out)


# In[ ]:


net = AlexNet(num_classes = outputs, inputs=inputs)
file_name = 'alexnet-'


# In[ ]:


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# In[ ]:


criterion = nn.CrossEntropyLoss()


# In[ ]:


train_loss, test_loss = start_train_test(net, trainloader, testloader, criterion)


# In[ ]:


plt.plot(train_loss)
plt.ylabel('Train Loss')
plt.show()


# In[ ]:


plt.plot(test_loss)
plt.ylabel('Test Loss')
plt.show()


# In[ ]:




