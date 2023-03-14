#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files
#in the input directory

import os
print(os.listdir("../input"))




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




train_path = "../input/train"
test_path = "../input/test"




os.listdir(test_path)[:10]




os.listdir(train_path)




driverImgs = pd.read_csv('../input/driver_imgs_list.csv')
print(driverImgs.head(10))




a = pd.DataFrame({'img':os.listdir(test_path)[:10]})
a.iloc[1, 0]




# Loading all train image names inside trainImgs list
# from trainImgs[0..9]
trainImgs = []
for i in range(10):
    # making a path i.e. '../input/train/c0' or '../input/train/c9' and checking the path
    className = "/c" + str(i)
    path = train_path + className 
    assert os.path.exists(path) 
    # reading images from each folder and adding to the list
    trainImgs.append(os.listdir(path))




# Making x, y for plotting
x = range(10)
y = []

# clalculating the total number of training images
numTrainImgs = 0
for i in range(10):
    y.append(len(trainImgs[i]))
    numTrainImgs = numTrainImgs + y[i]
    print("Number of images in c%d folder is: %d" % (i, y[i]))
print("\t\t\tTotal is: %d" % numTrainImgs)

# Plotting
plt.figure(figsize = (13, 8))
colors = mpl.cm.rainbow(np.linspace(0, 1, 10))  # Defining rainbow colors
plt.bar(x, y, color = colors);
plt.xlabel("number of images in each folder from c0..c9")




print("number of imgaes in driver list file is: %d (unique: %d)" 
      % (driverImgs['img'].count(), driverImgs['img'].nunique()))




# Plotting a 13 x 8 figure
plt.figure(figsize = (13, 8))
sb.countplot(driverImgs['subject'], palette = 'Set3');
# Changing the label of axis
plt.xlabel("Subjects")
plt.ylabel("number of existance")
# Showing the total number of subject as a title
plt.title(str("Total Number of subjects is:" + str(driverImgs['subject'].nunique())));




# Plotting a 13 x 8 figure
plt.figure(figsize = (13, 8))
sb.countplot(driverImgs['classname'], palette = 'Set3');
# Changing the label of axis
plt.xlabel("Class Names")
plt.ylabel("Number")
# Showing the total number of subject as a title
plt.title("How many images there are for each class");




class DriverImageDataset(Dataset):
    """DriverImageDataset."""

    def __init__(self, root_dir = '../input', csv_file = 'driver_imgs_list.csv', download = False,
                 train = True , transform = None, newShape = None,rand_state = 2, limit = None):
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




new_shape = (int(640 / 5), int(480 / 5))  # 1/5th of original shape
new_shape




trainData = DriverImageDataset(root_dir='../input', train = True, download = False, 
                                newShape = new_shape)
testData = DriverImageDataset(root_dir='../input', train = False, download = False, 
                               newShape = new_shape)




# Hyperparameters
BATCH_SIZE    = 25 #256
LEARNING_RATE = 0.001
WORKERS       = 0 #10 




def show_batch(images, targets = None, predictions = None):
    '''This method gets a list of images with their target value
       and plot them in equal rows and column.
       Also can get the predictions values and show them with targets.
       images(list) : List of images
       targets(list): Labels for each image
       predictions(list): If not None, is list of predicted values
                          corresponding with each given image'''
    plt.figure(figsize = (15, 15))
    ncols = np.ceil(np.sqrt(len(images)))
    nrows = np.ceil(len(images) / ncols)
    for i in range(len(images)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i][0].numpy().squeeze())
        plt.xticks([]); plt.yticks([]); #plt.axis('off');
        if predictions is not None:
            plt.xlabel("P:{}, T:{}".format( predictions[i].numpy(),                                            targets[i].numpy(), fontsize = 'small'))
        elif targets is not None:
            plt.xlabel("T:{}".format(targets[i].numpy()))




import random

rand_list = random.sample(range(numTrainImgs), 10)

plot_list = []
plot_labels = []
for i in rand_list:
    im, lab = trainData[i]
    plot_list.append(im)
    plot_labels.append(lab)
    
show_batch(plotImgs, plotLables)




# Hyperparameters
BATCH_SIZE    = 64 #256
LEARNING_RATE = 0.001
WORKERS       = 2 #10

# For picking 5% of train data as an Evaluation Data
evalData_length = numTrainImgs - int(numTrainImgs * 0.05)

# Reading Data sets
trainData = DriverImageDataset(root_dir='../input', train = True, download = False, 
            transform = ToTensor(), newShape = new_shape , limit = (0, evalData_length))
evalData  = DriverImageDataset(root_dir='../input', train = True, download = False, 
            transform = ToTensor(), newShape = new_shape , limit = (evalData_length, numTrainImgs))
testData  = DriverImageDataset(root_dir='../input', train = False, download = False, 
                               transform = ToTensor(), newShape = new_shape)

# Checking if it picked all train images as evalData + trainData
assert numTrainImgs == len(evalData) + len(trainData) 

# Setting Data Loaders
train_loader = DataLoader(trainData, batch_size = BATCH_SIZE, 
                          num_workers = WORKERS, shuffle = True)
eval_loader  = DataLoader(evalData, batch_size = BATCH_SIZE, 
                          num_workers = WORKERS, shuffle = True)
test_loader  = DataLoader(testData, batch_size = BATCH_SIZE, 
                          num_workers = WORKERS, shuffle = True)




class MLP(nn.Module):
    def __init__(self, shape = (3, new_shape[0], new_shape[1]), num_classes = 10):
        super().__init__()
        
        num_inputs = np.product(shape)
    
        self.layer1 = nn.Linear(num_inputs, 50)
        self.layer2 = nn.Linear(50, 100)
        self.layer3 = nn.Linear(100, num_classes)
        
    
    def forward(self, x):
        
        #Conv up here
        
        
        x = x.reshape(x.shape[0], -1)  # Flattening the input
        h1 = self.layer1(x)            # First layer
        #!nvidia-smi 
        h1 = F.relu(h1)                # Apply nonlinearity
        h2 = self.layer2(h1)
        h2 = F.relu(h2)
        y = self.layer3(h2)
        
        return y  # Will learn to treat 'a' as the natural parameters of a multinomial distr. 




MLPNet = MLP()




print(MLPNet)
print("----")
print(list(MLPNet.state_dict())) # Assign names to each one of group of tensor parameters
print("----")
print(MLPNet.parameters)




import torch.cuda
torch.cuda.is_available()




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




net = togpu(MLPNet)




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net.parameters(), lr = LEARNING_RATE)




def compute_eval_loss(net, criterion, loader):
    # Evaluate the model
    with torch.no_grad():
        eval_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(loader), desc = 'Evaluating', 
                                 total = len(loader), leave = False):
            inputs, labels = data
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
                   'simplecnn-checkpoint.pth.tar')
    
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            shutil.copyfile('simplecnn-checkpoint.pth.tar', 'simplecnn-best.pth.tar')
        
        print("Epoch {: 4}   loss: {: 2.5f}  Eval_loss: {: 2.5f}  time: {}".format(epoch, 
                                                          running_loss / len(train_loader),
                                                          eval_loss,                         
                                                          tend-tstart))




get_ipython().system('nvidia-smi ')




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
        # I have to add one extra axis at the beginning by None
        p = tocpu(net(togpu(x[None,...]))).argmax(1)[0]  
        predictions[i] = int(p) # Changing Tensors into integers
        targets[i] = t 

    # Showing classification metrics
    print(classification_report(targets, predictions))




b_loss = float('inf')  # Assign infinity
run_simpleCNN(net, optimizer, criterion, epoch = 2, best_eval_loss = b_loss)
epoch, b_loss, ehist, thist = resume(net, optimizer, fn = 'simplecnn-best.pth.tar')
report(net, evalData)




run_simpleCNN(net, optimizer, criterion, epoch = 10, best_eval_loss = b_loss)
epoch, b_loss, ehist, thist = resume(net, optimizer, fn = 'simplecnn-best.pth.tar')
report(net, evalData) 




run_simpleCNN(optimizer, criterion, epoch = 10, best_eval_loss = b_loss)
optimizer, epoch, b_lost, ehist, thist = resume(net, optimizer, fn = 'simplecnn-best.pth.tar')
report(net, evalData)




sample_submission = pd.read_csv('../input/sample_submission.csv') 




submission_01 = pd.DataFrame({'img':testData.driver_imgs_list.iloc[:, 0], 
                              'c0':np.zeros(len(testData)),
                              'c1':np.zeros(len(testData)),
                              'c2':np.zeros(len(testData)),
                              'c3':np.zeros(len(testData)),
                              'c4':np.zeros(len(testData)),
                              'c5':np.zeros(len(testData)),
                              'c6':np.zeros(len(testData)),
                              'c7':np.zeros(len(testData)),
                              'c8':np.zeros(len(testData)),
                              'c9':np.zeros(len(testData)) })

for i  in tqdm.tnrange(len(testData)):
        x = testData[i]
        # I have to add one extra axis at the beginning by None
        submission_01.at[i, 1:] = tocpu(net(togpu(x[None,...])))[0].detach().numpy()
        
    
print(submission_01)




submission_01.to_csv('submission_01_last.csv', index = False)






