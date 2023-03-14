#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb  # Importing seaborn for plotting

from PIL import Image

import tqdm  # For ProgressBar
import time  # For recording time

from torch.utils.data.dataset import Dataset

from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch
from torch import nn # python module
import torch.nn.functional as F # Regular function

import tensorflow as tf

import skimage.transform
#from sklearn.model_selection import train_test_split

import shutil  # For Copying files(checkpoints)

from sklearn.metrics import classification_report  # For getting a report from the net

# For adding filters
import skimage.color  
from skimage.transform import warp

# For making interaction plot
from ipywidgets import interact

import random




class DriverImageDataset(Dataset):
    """DriverImageDataset."""

    def __init__(self, root_dir = '../input', csv_file = 'driver_imgs_list.csv', download = False,
                 train = True , transform = None, newShape = None, evalData = None, 
                 rand_state = 2, limit = None, filter_skin = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            download(boolean): Download the data first in active path.
            train   (boolean): If True (default), returns train data, and test data otherwise.
            newShape  (tuple): Resize images to the given new shape for instance: (28, 28)
            transform (callable, optional): Optional transform to be applied on a sample.
            limit     (tuple): Picking images based on the given tuple 
                               i.e. limit = (starts, ends (not, included))
            rand_state  (int): int or numpy.random.RandomState, optional Seed for the 
                                random number generator (if int), or numpy RandomState object.
        """
        if download:
            get_ipython().system('pip install kaggle')
            get_ipython().system('kaggle competitions download -c state-farm-distracted-driver-detection')
        
        self._root_dir_       = root_dir
        self._transform_      = transform
        self._newShape_       = newShape
        self._cvs_file_path_  = os.path.join(root_dir, csv_file)
        self._isTrain_        = train
        self._dataPath_       = os.path.join(root_dir, 'train')
        self.driver_imgs_list = pd.read_csv(self._cvs_file_path_)
        self._filter_skin     = filter_skin
        if not self._isTrain_:
            self._isTrain_        = False
            self._dataPath_       = os.path.join(self._root_dir_, 'test')
            self.driver_imgs_list = pd.DataFrame({'img':os.listdir(self._dataPath_)})
        # Shuffles the self.driver_imgs_list
        self.__shuffle__(rand_state, limit)
        
    def __len__(self):
        return len(self.driver_imgs_list)

    def __getitem__(self, idx):
        img_name = ''
        label = 0
        if self._isTrain_:
            class_name = self.driver_imgs_list.iloc[idx, 1]
            img_name = os.path.join(self._dataPath_, class_name, self.driver_imgs_list.iloc[idx, 2])
            label    = int(class_name[1])
        else:
            img_name = os.path.join(self._dataPath_, self.driver_imgs_list.iloc[idx, 0])
        img = Image.open(img_name)
        #img = img.convert('RGB')
        if self._newShape_:
            img = img.resize(self._newShape_)
        if self._filter_skin:
            img = self.__filter_skin__(img)
        if self._transform_:
            img = self._transform_(img)
        if not self._isTrain_:
            return img
        label = torch.from_numpy(np.asarray(label))
        return img, label
    
    def __shuffle__(self, rand_state, limit):
        self.driver_imgs_list = self.driver_imgs_list.sample(frac = 1,                                     random_state = rand_state).reset_index(drop = True)
        if limit:
            self.driver_imgs_list = self.driver_imgs_list.iloc[limit[0]: limit[1]].                                     reset_index(drop = True)
    def __filter_skin__(self, im):
        y_begin, y_end =55,  130
        Cb_begin, Cb_end =0, 123
        Cr_begin, Cr_end =128, 256
        filter1 = np.array([[1., 0., 48.], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
        im = im.copy()
        im = skimage.color.rgb2ycbcr(im)
        im = warp(im, filter1)
        
        
        im[((im[:, :, 0] <= y_begin) | (im[:, :, 0] >= y_end))] = 0
        im[((im[:, :, 1] <= Cb_begin) | (im[:, :, 1] >= Cb_end))] = 0
        im[((im[:, :, 2] <= Cr_begin) | (im[:, :, 2] >= Cr_end))] = 0
        
        im = skimage.color.ycbcr2rgb(im)
        return im




new_shape = (int(640 / 2), int(480 / 2))  # 1/5th of original shape
new_shape




# Hyperparameters
BATCH_SIZE    = 64 #256
LEARNING_RATE = 0.001
WORKERS       = 0 #10

numTrainImgs = len(pd.read_csv('../input/driver_imgs_list.csv'))
# For picking 5% of train data as an Evaluation Data
evalData_start = numTrainImgs - int(numTrainImgs * 0.05)

# Reading Data sets
trainData = DriverImageDataset(root_dir='../input', train = True, download = False, 
            transform = ToTensor(), newShape = new_shape , limit = (0, evalData_start),
                              filter_skin = True)
evalData  = DriverImageDataset(root_dir='../input', train = True, download = False, 
            transform = ToTensor(), newShape = new_shape , limit = (evalData_start, numTrainImgs),
                              filter_skin = True)
testData  = DriverImageDataset(root_dir='../input', train = False, download = False, 
                               transform = ToTensor(), newShape = new_shape, filter_skin = True)

# Checking if it picked all train images as evalData + trainData
assert numTrainImgs == len(evalData) + len(trainData) 

# Setting Data Loaders
train_loader = DataLoader(trainData, batch_size = BATCH_SIZE, 
                          num_workers = WORKERS, shuffle = True)
eval_loader  = DataLoader(evalData, batch_size = BATCH_SIZE, 
                          num_workers = WORKERS, shuffle = True)
test_loader  = DataLoader(testData, batch_size = BATCH_SIZE, 
                          num_workers = WORKERS, shuffle = True)




class simpleCNN(nn.Module):
    def __init__(self, shape = (3, new_shape[0], new_shape[1]), num_classes = 10):
        super().__init__()
        
        self.layer1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.layer3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.layer4 = nn.Linear(40 * 30 * 128, num_classes)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.layer2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.layer3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = x.reshape(-1, 40 * 30 * 128)
        y = self.layer4(x)
        
        return y  # Will learn to treat 'a' as the natural parameters of a multinomial distr. 




cNN_net = simpleCNN()

print(cNN_net)
print("----")
print(list(cNN_net.state_dict())) # Assign names to each one of group of tensor parameters
print("----")
print(cNN_net.parameters)




import torch.cuda
print(torch.cuda.is_available())

if torch.cuda.is_available():
    def togpu(x):
        return x.cuda()
    def tocpu(x):
        return x.cpu()
else:
    def togpu(x):
        return x
    def tocpu(x):
        return x




net = togpu(cNN_net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net.parameters(), lr = LEARNING_RATE)




def compute_eval_loss(net, criterion, loader):
    # Evaluate the model
    with torch.no_grad():
        eval_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(loader), desc = 'Evaluating', 
                                 total = len(loader), leave = False):
            inputs, labels = data
            inputs = inputs.float()
            lables = labels.float()


            
            inputs, labels = togpu(inputs), togpu(labels)
            outputs = net(inputs)               # Predict
            loss = criterion(outputs, labels)   # Grade / Evaluate
            eval_loss += loss.item()
    eval_loss /= len(test_loader)
    return eval_loss




def run_simpleCNN(net, optimizer, criterion, epoch = 2, best_eval_loss = float('inf')):
    for epoch in tqdm.tnrange(epoch):
        running_loss = 0.0
        tstart = time.time()
        for _, data in tqdm.tqdm(enumerate(train_loader), total = len(train_loader), leave = False):
            # get the inputs
            inputs, labels = data
            inputs = inputs.float()
            lables = labels.float()
        
            # Move inputs to the GPU
            inputs, labels = togpu(inputs), togpu(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)               # Predict
            loss = criterion(outputs, labels)   # Grade / Evaluate
            loss.backward()                     # Determine how each parameter effected the loss
            optimizer.step()                    # Update parameters 

            # print statistics
            running_loss += loss.item()
        tend = time.time()
    
        # Save parameters
        running_loss /= len(train_loader)
        # This is for where we can stop
        eval_loss = compute_eval_loss(net, criterion, eval_loader)  
        torch.save(dict(epoch = epoch, 
                     loss = eval_loss,
                    parameters = net.state_dict(),
                    optimizer  = optimizer.state_dict()),
                   'simpleCNN-checkpoint.pth.tar')
    
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            shutil.copyfile('simpleCNN-checkpoint.pth.tar', 'simpleCNN-best.pth.tar')
        
        print("Epoch {: 4}   loss: {: 2.5f}  Eval_loss: {: 2.5f}  time: {}".format(epoch, 
                                                          running_loss / len(train_loader),
                                                          eval_loss,                         
                                                          tend-tstart))




import sys, os
def resume(model, optimizer, fn = 'checkpoint.pth.tar'):
    '''
        Loads a torch net from the given file.
    '''
    if os.path.isfile(fn):
        print("=> loading checkpoint '{}'".format(fn))
        checkpoint  = torch.load(fn)
        model.load_state_dict(checkpoint['parameters'])
        start_epoch = checkpoint['epoch']
        best_loss   = checkpoint['loss']
        ehist       = checkpoint.get('ehist', [])
        thist       = checkpoint.get('thist', [])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Loaded checkpoint '{}' (epoch {})".format(fn, checkpoint['epoch']))
    else:
        start_epoch = 0
        ehist = []
        thist = []
        print ("=> no checkpoint found at '{}'".format(fn))
    return start_epoch, best_loss, ehist, thist

# Qualitative assessment (By Numbers)
def report(net, evalData):
    predictions = np.zeros(len(evalData)) 
    targets = np.zeros(len(evalData))

    for i  in tqdm.tnrange(len(evalData)):
        x, t = evalData[i]
        x = x.float()
        t = t.float()
        # I have to add one extra axis at the beginning by None
        p = tocpu(net(togpu(x[None,...]))).argmax(1)[0]  
        predictions[i] = int(p) # Changing Tensors into integers
        targets[i] = t 

    # Showing classification metrics
    print(classification_report(targets, predictions))




get_ipython().run_cell_magic('time', '', "b_loss = float('inf')  # Assign infinity\nrun_simpleCNN(net, optimizer, criterion, epoch = 10, best_eval_loss = b_loss)\nepoch, b_loss, ehist, thist = resume(net, optimizer, fn = 'simpleCNN-best.pth.tar')\nreport(net, evalData)")




get_ipython().run_cell_magic('time', '', "submission_04 = pd.DataFrame({'img':testData.driver_imgs_list.iloc[:, 0], \n                              'c0':np.zeros(len(testData)),\n                              'c1':np.zeros(len(testData)),\n                              'c2':np.zeros(len(testData)),\n                              'c3':np.zeros(len(testData)),\n                              'c4':np.zeros(len(testData)),\n                              'c5':np.zeros(len(testData)),\n                              'c6':np.zeros(len(testData)),\n                              'c7':np.zeros(len(testData)),\n                              'c8':np.zeros(len(testData)),\n                              'c9':np.zeros(len(testData)) })\n\nfor i  in tqdm.tnrange(len(testData)):\n        x = testData[i].float()\n        # I have to add one extra axis at the beginning by None\n        p = tocpu(net(togpu(x[None,...]))).argmax(1)[0]\n        p = int(p) # Changing Tensors into integers\n        submission_04.at[i, 'c' + str(p)] = 1.0\n        \n    \nprint(submission_04)")




submission_04.to_csv('submission_04_01.csv', index = False)






