#!/usr/bin/env python
# coding: utf-8



ls ../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/




import os
import time
import copy
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from ipywidgets import IntProgress

#Computer Vision Libraries
import cv2
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor

#Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, random_split

from matplotlib import pyplot as plt




data_path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
train_data_images = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x/'

train = pd.read_csv(os.path.join(data_path,'stage_2_train.csv')) #Read file
train[['ID','Image','Diagnosis']]=train['ID'].str.split('_',expand= True) #Split the ID column at each _
train = train[['Image','Diagnosis','Label']] #reorder the columns
train.drop_duplicates(inplace= True)  #drop duplicates
train = train.pivot(index = 'Image' , columns = 'Diagnosis', values = 'Label').reset_index() #Reorganizes csv to make columns with labels instead of 6 rows for each image
train['Image'] = 'ID_' + train['Image'] #Put ID_ back with picture ID's

#Remove files that aren't of png type
png = glob.glob(os.path.join(train_data_images, '*.png')) #list of paths to the pictures
png = [os.path.basename(png)[:-4] for png in png] #drop the .png at the end 
png = np.array(png) #convert to a NumPy array


train = train[train['Image'].isin(png)] # Reconcile the lists and images
train = train[:300000] #Take the first 300k pictures
train.to_csv('train.csv', index = False)
print(train.shape) #just to know shape of dataset 




#Same code as before, just changing names
data_path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
test_data_images = '../input/stage-2-png/'

test = pd.read_csv(os.path.join(data_path,'stage_2_sample_submission.csv')) #Read file
test[['ID','Image','Diagnosis']]=test['ID'].str.split('_',expand= True) #Split the ID column at each _
test['Image'] = 'ID_' + test['Image'] #Put ID_ back with picture ID's
test = test [['Image','Label']]
test.drop_duplicates(inplace= True)  #drop duplicates
test.to_csv('test.csv', index = False)
print(test.shape) #just to know shape of dataset 




class RSNA(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):       
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):        
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)   
        
        if self.transform:                   
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:                  
            return {'image': img}




transform_train = Compose([ShiftScaleRotate(),ToTensor()])
transform_test = Compose([ToTensor()])

train_dataset = RSNA(csv_file='train.csv', path=train_data_images, transform=transform_train, labels=True)
test_dataset = RSNA(csv_file='test.csv', path=test_data_images, transform=transform_test, labels=False)

batch_size = 64
data_train_generator = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
data_test_generator = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model0 = models.googlenet(pretrained = True)
model = torch.nn.Sequential(model0, torch.nn.Linear(1000, 6)).to(device)


#model = {alexnet,vgg, resnet18}




num_epochs = 3
optimizer = optim.Adam(model.parameters(), lr=4e-5)
criterion = torch.nn.BCEWithLogitsLoss()




def train_model(model,criterion,optimizer,num_epochs=1):
    for epoch in range(1, num_epochs+1):

        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        model.train()    
        tr_loss = 0

        tk0 = tqdm(data_train_generator, desc="Iteration")

        for step, batch in enumerate(tk0):

            inputs = batch["image"]
            labels = batch["labels"]

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        torch.save(model.state_dict(), f'resnext50_{epoch}.pth') 

        epoch_loss = tr_loss / len(data_train_generator)
        print('Training Loss: {:.4f}'.format(epoch_loss))
        
    return model




model_trained = train_model(model,criterion,optimizer,num_epochs)




for param in model.parameters():
    param.requires_grad = False

model.eval()

test_pred = np.zeros((len(test_dataset) * 6, 1))

for i, batch_ in enumerate(tqdm(data_test_generator)):
    batch_ = batch_["image"]
    batch_ = batch_.to(device, dtype=torch.float)
    
    with torch.no_grad():
        
        pred = model(batch_)
        
        test_pred[(i * batch_size * 6):((i + 1) * batch_size * 6)] = torch.sigmoid(
            pred).detach().cpu().reshape((len(batch_) * 6, 1))  




ls ../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/




submission =  pd.read_csv(os.path.join(data_path, 'stage_2_sample_submission.csv'))
submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission.columns = ['ID', 'Label']
submission.to_csv('submission-1.csv', index=False)

