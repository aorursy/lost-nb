#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import cv2

from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from torch.optim import lr_scheduler
from collections import OrderedDict
from PIL import Image
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
import json

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

device = torch.device("cuda:0")

import os
print(os.listdir("../input"))




get_ipython().system('ls ../input/resized-2015-2019-blindness-detection-images/labels')




get_ipython().system('ls ../input/aptos2019-blindness-detection')




train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')




df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
df.head()




type_percents = df["diagnosis"].value_counts(normalize=True)
type_percents.values




sns.barplot(x=type_percents.index, y=type_percents.values*100)          .set(xlabel="severity of diabetic retinopathy", ylabel='Percent (%)')    

plt.tight_layout()
plt.show()




fig = plt.figure(figsize=(25, 25))
# display 20 images
train_imgs = os.listdir("../input/aptos2019-blindness-detection/train_images")
for idx, img in enumerate(np.random.choice(train_imgs, 15)):
    ax = fig.add_subplot(5, 15//5, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/aptos2019-blindness-detection/train_images/" + img)
    plt.imshow(im)
    lab = df.loc[df['id_code'] == img.split('.')[0], 'diagnosis'].values[0]
    ax.set_title(f'Label: {lab}')




# def crop_image1(img,tol=7):
#     # img is image data
#     # tol  is tolerance
        
#     mask = img>tol
#     return img[np.ix_(mask.any(1),mask.any(0))]


# def crop_image(img,tol=7):
#     if img.ndim ==2:
#         mask = img>tol
#         return img[np.ix_(mask.any(1),mask.any(0))]
#     elif img.ndim==3:
#         h,w,_=img.shape
# #         print(h,w)
#         img1=cv2.resize(crop_image1(img[:,:,0]),(w,h))
#         img2=cv2.resize(crop_image1(img[:,:,1]),(w,h))
#         img3=cv2.resize(crop_image1(img[:,:,2]),(w,h))
        
# #         print(img1.shape,img2.shape,img3.shape)
#         img[:,:,0]=img1
#         img[:,:,1]=img2
#         img[:,:,2]=img3
#         return img




class AptosDrTrainDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.targets = self.data.diagnosis
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        img_name = os.path.join("../input/aptos2019-blindness-detection/train_images", 
                                self.data.loc[idx, 'id_code'] + '.png')
#         IMG_SIZE = 224
#         image = cv2.imread(img_name)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#         image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line
#         image = crop_image(image)
        image = Image.open(img_name)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        if self.transform:
            transformed_image = self.transform(image)
            return transformed_image, label
        
        return image, label




# train_transforms = transforms.Compose([transforms.ToPILImage(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.485, 0.456, 0.406], 
#                                                             [0.229, 0.224, 0.225])])

# train_transforms = transforms.Compose([transforms.ToPILImage(),
#                                        transforms.Resize((224, 224)),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.RandomVerticalFlip(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.485, 0.456, 0.406], 
#                                                             [0.229, 0.224, 0.225])])

train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

dataset = AptosDrTrainDataset(csv_file="../input/aptos2019-blindness-detection/train.csv", transform=train_transforms)

label = dataset.targets

train_idx, valid_idx= train_test_split(
    np.arange(len(label)), test_size=0.2, random_state=42, shuffle=True, stratify=label)

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, 
                                                sampler=train_sampler)

valid_data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, 
                                                sampler=valid_sampler)




for image, label in train_data_loader:
    print(image.shape)
    break




model = models.resnet152(pretrained=True)
model




model.fc = nn.Sequential(
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=2048, bias=True),
                          nn.ReLU(),
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.5),
                          nn.Linear(in_features=2048, out_features=1, bias=True),
                         )




model




# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')




model.features[28]




for param in model.parameters():
    param.requires_grad = False
# for param in model.avgpool.parameters():
#     param.requires_grad = True
# for param in model.features[28].parameters():
#      param.requires_grad = True
# for param in model.features[29].parameters():
#      param.requires_grad = True
# for param in model.features[30].parameters():
#      param.requires_grad = True
for param in model.avgpool.parameters():
    param.requires_grad = True
for param in model.layer4[2].parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True




criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)




if train_on_gpu:
    model.cuda()

# Number of epochs:
n_epochs = 25
valid_loss_min = np.Inf
  
for epoch in range(1, n_epochs + 1):
  # keep track of training & validation loss
  train_loss = 0.0
  valid_loss = 0.0
  
  model.train()
  exp_lr_scheduler.step()
  
  for data, target in train_data_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    data = data.to(device, dtype=torch.float)
    target = target.view(-1, 1)
    target = target.to(device, dtype=torch.float)
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    output.cuda()
    # calculate batch loss
    loss = criterion(output, target)
    # backward pass: compute gradient of loss wrt model parameters
    loss.backward()
    # perform single optimization step (parameter update)
    optimizer.step()
    # update training loss
    train_loss += loss.item()*data.size(0)
    
   ### Validating the model ###
  
  model.eval()
  
#   class_correct = list(0. for i in range(5))
#   class_total = list(0. for i in range(5))

  for data, target in valid_data_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    data = data.to(device, dtype=torch.float)
    target = target.view(-1, 1)
    target = target.to(device, dtype=torch.float)
    # forward pass:
    output = model(data)
    # calculate batch loss
    loss = criterion(output, target)
    # update validation loss
    valid_loss += loss.item()*data.size(0)
    
#     # convert output probabilities to predicted class
#     _, pred = torch.max(output, 1)    
#     # compare predictions to true label
#     correct_tensor = pred.eq(target.data.view_as(pred))
#     correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
#     # calculate validation accuracy for each object class
#     for i in range(target.data.size()[0]):
#         label = target.data[i]
#         class_correct[label] += correct[i].item()
#         class_total[label] += 1
    
  # calculate average losses
  train_loss = train_loss / len(train_data_loader.dataset)
  valid_loss = valid_loss / len(valid_data_loader.dataset)
  
  # print training/validation statistics 
  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
  
#     print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))
    
  # Save state_dict if validation loss decreased
  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
    valid_loss_min = valid_loss
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'valid_loss_min': valid_loss_min
    }
    torch.save(state, 'aptos_resnet_152_9.pt')




model.load_state_dict(torch.load("../input/aptos-resnet-152-8/aptos_resnet_152_8.pt")["state_dict"])
model = model.to(device)




for param in model.parameters():
    param.requires_grad = False

model.eval()




get_ipython().system('ls /working')




class AptosDrTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        img_name = os.path.join('../input/resized-2015-2019-blindness-detection-images/resized test 15', 
                                self.data.loc[idx, 'image'] + '.jpg')
        image = Image.open(img_name)
        label = torch.tensor(self.data.loc[idx, 'level'])
        if self.transform:
            transformed_image = self.transform(image)
            return transformed_image, label
        return image, label




test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_dataset = AptosDrTestDataset(csv_file='../input/resized-2015-2019-blindness-detection-images/labels/testLabels15.csv', transform=test_transforms)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)




preds = np.empty((0,1), int)
target_lst = np.empty((0,1), int)

for data, target in valid_data_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # convert output probabilities to predicted class
#     _, pred = torch.max(output, 1) 
    preds = np.append(preds,  output.cpu().numpy(), axis=0)
    target_ex =  np.expand_dims(target.cpu().numpy(), axis=1)
    print(target_ex.shape)
    target_lst = np.append(target_lst, target_ex, axis=0)
    print(preds.shape)
    print(target_lst.shape)




class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']




preds.shape




preds.T.shape




target_lst.T




optR = OptimizedRounder()
optR.fit(preds, target_lst)
coefficients = optR.coefficients()




coefficients




sample_df = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample_df.shape




coef = [0.5, 1.5, 2.5, 3.5]

for i, pred in enumerate(preds):
    if pred < coef[0]:
        preds[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        preds[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        preds[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        preds[i] = 3
    else:
        preds[i] = 4




sample_df.diagnosis = preds




sample_df.to_csv("submission.csv", index=False)

