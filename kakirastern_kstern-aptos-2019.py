#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from PIL import Image
import cv2

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import time
import copy
import glob
import sys
sys.setrecursionlimit(100000)  # To increase the capacity of the stack

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = get_ipython().getoutput("ldconfig -p|grep cudart.so|sed -e 's/.*\\.\\([0-9]*\\)\\.\\([0-9]*\\)$/cu\\1\\2/'")
accelerator = cuda_output[0]

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')




# Load data
train_dir = '../input/aptos2019-blindness-detection/train_images/'

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

# Split off data for validation set
train, valid = train_test_split(train, train_size=0.75, test_size=0.25, shuffle=False)




print('Number of train samples: ', train.shape[0])
print('Number of validation samples: ', valid.shape[0])
print('Number of test samples: ', test.shape[0])
display(train.head(10))




df = pd.DataFrame(train)

ClassCounts = pd.value_counts(df['diagnosis'], sort=True)
print(ClassCounts)
plt.figure(figsize=(10, 7))
ClassCounts.plot.bar(rot=0);
plt.title('Severity Counts for Training Data');




# To display 5 unique retina images from each of the 5 classes

j = 1
fig=plt.figure(figsize=(15, 15))
for class_id in sorted(train['diagnosis'].unique()):
    plot_no = j
    for i, (idx, row) in enumerate(train.loc[train['diagnosis'] == class_id].sample(5).iterrows()):
        ax = fig.add_subplot(5, 5, plot_no)
        im = Image.open(f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png")
        plt.imshow(im)
        ax.set_title(f'Label: {class_id}')
        plot_no += 5
    j += 1

plt.show()
plt.savefig("samples_viz.png")




import PIL

class ImageLoader(Dataset):
    
    def __init__(self, df, datatype):
        self.datatype = datatype
        #self.labels = df['diagnosis'].values
        if self.datatype == 'train':
            self.image_files = [f'../input/aptos2019-blindness-detection/train_images/{i}.png' for i in train['id_code'].values]
            self.transform = transforms.Compose([
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 #transforms.Grayscale(num_output_channels=3),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
            self.labels = train['diagnosis'].values
        elif self.datatype == 'valid':
            self.image_files = [f'../input/aptos2019-blindness-detection/train_images/{i}.png' for i in valid['id_code'].values]
            self.transform = transforms.Compose([
                                                #transforms.Grayscale(num_output_channels=3),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
            self.labels = valid['diagnosis'].values
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        image = image.convert('RGB')
        image = image.resize((224, 224))
        #image = PIL.ImageOps.autocontrast(image)
        image = self.transform(image)
        if self.datatype == 'train':
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return image, label
        elif self.datatype == 'valid':
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return image, label




retinaImages = ImageLoader(df=valid, datatype='valid')




len(retinaImages)




valid['diagnosis'][13+len(train)]




type(retinaImages[13][0])
print(retinaImages[13][0])




plt.imshow(retinaImages[13][0].permute(1, 2, 0))
print("Label: " + str(retinaImages[13][1]))




trainloader = torch.utils.data.DataLoader(ImageLoader(df=train, datatype='train'), batch_size=60, shuffle=True)
testloader = torch.utils.data.DataLoader(ImageLoader(df=valid, datatype='valid'), batch_size=60, shuffle=False)  # serving as validation set...




#model = models.densenet121(pretrained=False)
model = models.resnet50(pretrained=True)
model

if train_on_gpu:
    model = model.cuda()
model




cuda_output[0]




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model




# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

from collections import OrderedDict
fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(1024, 5)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
  
weights = torch.tensor([2., 11., 5., 13., 12.])
weights = weights.to(device)
criterion = nn.NLLLoss(weight=weights, reduction='mean')
#criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.SGD(model.fc.parameters(), lr=0.0005, momentum=0.9)

#model = load_checkpoint('/kaggle/checkpoint.pth')

model.fc = nn.Linear(512, 5)
model.fc = fc

model.to(device)    




checkpoint = {'model': fc,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, '/kaggle/checkpoint.pth')




get_ipython().system('ls /kaggle/working')




for i, (inputs, labels) in enumerate(trainloader):
    # Move input and label tensors to the GPU
    
    inputs, labels = inputs.to(device), labels.to(device)
    
    start = time.time()

    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i==3:
        break
        
print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")




epochs = 5
steps = 0
running_loss = 0
print_every = 10

validation_accuracy = []

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    #print(logps)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    #print(ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    #print(top_class)
                    equals = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    validation_accuracy.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}... "
                  f"Validation accuracy: {accuracy/len(testloader):.3f}"
                  )
            
        running_loss = 0
        model.train()




get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt




plt.plot(validation_accuracy, label='Validation accuracy')
plt.legend(frameon=False)




print(test['id_code'].values)




class SubmissionLoader(Dataset):
    
    def __init__(self, df):
        self.datatype = 'test'
        self.image_files = [f'../input/aptos2019-blindness-detection/test_images/{i}.png' for i in test['id_code'].values]
        self.transform = transforms.Compose([
                                            #transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        self.id_code = test['id_code'].values

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        image = image.convert('RGB')
        image = image.resize((224, 224))
        #image = PIL.ImageOps.autocontrast(image)
        image = self.transform(image)
        id_code = self.id_code[index]
        return image, id_code




submissions = torch.utils.data.DataLoader(SubmissionLoader(df=test), batch_size=1, shuffle=False)




len(submissions)




preds = []
id_codes = []

model.eval()
with torch.no_grad():

    for i, (image, id_code) in enumerate(submissions):

        image = image.to(device)
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        pred = torch.squeeze(top_class).item()
        preds.append(pred)
        id_codes.append(id_code)




output = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
preds = list(map(int, preds))
output.diagnosis = preds
output.to_csv("submission.csv", index=False)




pd.read_csv('/kaggle/working/submission.csv')




freq, _ = np.histogram(output.diagnosis, density=True, bins=5)
freq

