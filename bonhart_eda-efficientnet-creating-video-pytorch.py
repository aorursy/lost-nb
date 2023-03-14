#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import cm

import os
print(os.listdir("../input/"))


# In[2]:


pixel_status = pd.read_csv('../input/recursion-cellular-image-classification/pixel_stats.csv')
train_df = pd.read_csv('../input/recursion-cellular-image-classification/train.csv')
test_df = pd.read_csv('../input/recursion-cellular-image-classification/test.csv')

train_controls = pd.read_csv('../input/recursion-cellular-image-classification/train_controls.csv')
test_controls = pd.read_csv('../input/recursion-cellular-image-classification/test_controls.csv')

sub = pd.read_csv('../input/recursion-cellular-image-classification/sample_submission.csv')

print('Dimensions: \n pixel_status: %s'     '\n train_df: %s \n test_df: %s'       '\n train_controls: %s \n test_controls: %s'       '\n submission: %s' % (pixel_status.shape, train_df.shape, 
                            test_df.shape, train_controls.shape,
                            test_controls.shape, sub.shape))


# In[3]:


pixel_status.head()


# In[4]:


train_controls.head()


# In[5]:


test_df.head()


# In[6]:


# Image from dataset with index 1
exp, well, plate = train_df.loc[1,['experiment', 'well', 'plate']]

# List of arrays of different channels(total 6) of the same image
img_names = [np.array(Image.open(os.path.join('../input/recursion-cellular-image-classification/train/',
                                              exp,
                                              f'Plate{plate}',
                                              f'{well}_s{1}_w{channel}.png')),
                      dtype=np.float32) for channel in range(1,7)]

# Сonversion to a six-channel image
sample = np.stack([img_ar for img_ar in img_names],axis=0)
sample.shape


# In[7]:


def plot_cell(sample_img):    
    channels = ['Nuclei', 'Endoplasmic reticuli', 'Actin', 'Nucleoli', 'Mitochondria', 'Golgi apparatus']
    cmaps = ['gist_ncar','terrain', 'gnuplot' ,'rainbow','PiYG', 'gist_earth']

    fig=plt.figure(figsize=(20, 15))
    for i in range(1,6+1):
        fig.add_subplot(1, 6, i)
        plt.imshow(sample_img[i-1, :, :,],cmap=cmaps[i-1]);
        plt.axis('off');
        plt.title(f'{channels[i-1]}')
    fig.suptitle("Single image channels", y=0.65, fontsize=15)
    plt.show()
    
## Let's looking on image channels
plot_cell(sample)


# In[8]:


# Loading libraries
import sys

package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'
sys.path.append(package_path)


# In[9]:


# Loading libraries
import sys
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


# In[10]:


class CellDataset(Dataset):
    def __init__(self, df, img_dir, site=1, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.site = site
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        exp, well, plate = self.df.loc[idx,['experiment', 'well', 'plate']].values
        img_channels = [np.array(Image.open(os.path.join(self.img_dir,
                                             exp,
                                             f'Plate{plate}',
                                             f'{well}_s{self.site}_w{channel}.png')), 
                                          dtype=np.float32) for channel in range(1,7)]
        
        one_img = np.stack([channel for channel in img_channels],axis=2)
        
        if self.transforms is not None:
            one_img = self.transforms(one_img)
        if self.img_dir == '../input/recursion-cellular-image-classification/train/':
            return one_img, self.df.loc[idx,['sirna']].astype('int32').values
        else:
            return one_img
                                 
            


# In[11]:


# Augmentations for data
aug = transforms.Compose([
      # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.456, 0.456, 0.406, 0.406],
                                 std=[0.229, 0.229, 0.225, 0.225, 0.224, 0.224])
])

# Dataset & data loaders
dataset = CellDataset(df=train_df, img_dir='../input/recursion-cellular-image-classification/train/', transforms=aug)
train_loader = DataLoader(dataset=dataset, batch_size=15, shuffle=True)

test_dataset = CellDataset(df=test_df, img_dir='../input/recursion-cellular-image-classification/test/', transforms=aug)
test_loader = DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)


# In[12]:


#train_loader checking
data, target = next(iter(train_loader))
print(data.shape, target.shape)


# In[13]:


#test_loader checking
test_data = next(iter(test_loader))
print(test_data.shape)


# In[14]:


data, target = next(iter(train_loader))
print('Dimension:', data.shape, ",", target[:, 0].shape)
print('Datatype: ', data.type(),",", target.type())


# In[15]:


plot_cell(data.numpy()[1,:,:,:])


# In[16]:


# Model parameters
num_epochs = 10
total_step = len(train_loader)
in_ch = 6
lr = 0.001


# In[17]:


model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1108)

# Changes count input channels of our model
trained_kernel = model._conv_stem.weight
new_conv = nn.Sequential(nn.Conv2d(in_ch, 32, kernel_size=(3,3), stride=(2,2), bias=False),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)))
with torch.no_grad():
    new_conv[0].weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
model._conv_stem = new_conv
model = model.cuda()


# In[18]:


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[19]:


# Train model
for epoch in range(num_epochs):
    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target[:,0].long().cuda()
        #print(data.shape)
        outputs = model(data)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                 .format(epoch+1, num_epochs, batch_i+1, total_step, loss.item()))
torch.save(model.state_dict(), 'model.pt')


# In[20]:


predictions = []
model.eval()
with torch.no_grad():
    for data in test_loader:
        data = data.cuda()
        output = model(data)
        batch_idx = output.max(dim=-1)[1].cpu().numpy()
        for pred in batch_idx:
            predictions.append(pred.astype(int))


# In[21]:


sub['sirna'] = predictions
sub.to_csv('submission.csv', index=False, columns=['id_code','sirna'])
print('Number of unique values:',len(sub['sirna'].value_counts()))


# In[22]:


sub.head()


# In[23]:


print(sample.shape)
# splitting a six-channel image into two three-channel images
rgb1 = Image.fromarray(np.uint8(sample[:3,:,:].transpose(1,2,0))).convert('RGB')
rgb2 = Image.fromarray(np.uint8(sample[3:,:,:].transpose(1,2,0))).convert('RGB')
# rgb1 + rgb2 (interpolation)
#rgb3 = Image.blend(rgb1, rgb2, 0.5).convert('RGB')
rgb3 = Image.blend(rgb1, rgb2, 0.5).convert('L')
# after which their composition for color saturation
img_composit = Image.composite(rgb1, rgb2, rgb3)
img_composit


# In[24]:


# Code from https://github.com/recursionpharma/rxrx1-utils/blob/master/rxrx/io.py
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
FPS = float(5)
RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}

def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[i, :, :] / vmax) /             ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) +             rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im


# In[25]:


# Convert images -> tensors -> list rgb -> video
def tensor_rgb_video(df, img_dir, experiment, video_name):
    
    df = df[df['experiment']==experiment].reset_index(drop=True)

    descript = np.zeros((512,512,3),dtype=np.uint8)
    descript = cv2.putText(descript, f'Experiment: {experiment}',
                           (10,250), cv2.FONT_ITALIC,
                           1.3,(255,255,255),2,cv2.LINE_AA)

    img_list = [descript] * 5
    for i in range(100):#len(df)
        well, plate = df.loc[i,['well', 'plate']]
        img_names = [np.array(Image.open(os.path.join(img_dir,
                                                      experiment,
                                                      f'Plate{plate}',
                                                      f'{well}_s{1}_w{channel}.png')),
                              dtype=np.float32) for channel in range(1,7)]
    
        tensor = np.stack([img_ar for img_ar in img_names],axis=0)
        rgb_img = convert_tensor_to_rgb(tensor)
        img_list.append(rgb_img)
        
    height, width, layers = img_list[1].shape
    video = cv2.VideoWriter(video_name, 0, FPS, (width, height))
    for img in img_list:  
        video.write(img.astype('uint8'))
    video.release()


# In[26]:


# Recursive function call
def all_video_experiments():
    experiments = train_df['experiment'].value_counts().to_string().split()[::2]
    a = len(experiments)
    while a !=0:
        a -= 1
        tensor_rgb_video(train_df,'../input/recursion-cellular-image-classification/train/', experiments[a], f'{experiments[a]}.avi')
        
all_video_experiments()

