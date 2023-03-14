#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from fastai.vision import *


# In[3]:


# deterministic, are we ?
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)


# In[4]:


train_df = pd.read_csv('../input/recursion-cellular-image-classification/train.csv')
train_df.head(10)


# In[5]:


def generate_df(train_df,sample_num=1):
    train_df['path'] = train_df['experiment'].str.cat(train_df['plate'].astype(str).str.cat(train_df['well'],sep='/'),sep='/Plate') + '_s'+str(sample_num) + '_w'
    train_df = train_df.drop(columns=['id_code','experiment','plate','well']).reindex(columns=['path','sirna'])
    return train_df
proc_train_df = generate_df(train_df)


# In[6]:


proc_train_df.head(10)


# In[7]:


import cv2
img = cv2.imread("../input/recursion-cellular-image-classification/train/HEPG2-01/Plate1/B02_s1_w1.png")
plt.imshow(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img)
gray_img.shape


# In[8]:



def open_rcic_image(fn):
    images = []
    for i in range(6):
        file_name = fn+str(i+1)+'.png'
        im = cv2.imread(file_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)
    image = np.dstack(images)
    #print(pil2tensor(image, np.float32).shape)#.div_(255).shape)
    return Image(pil2tensor(image, np.float32).div_(255))
  
class MultiChannelImageList(ImageList):
    def open(self, fn):
        return open_rcic_image(fn)


# In[9]:


il = MultiChannelImageList.from_df(df=proc_train_df, path="../input/recursion-cellular-image-classification/train/")


# In[10]:


def image2np(image:Tensor)->np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1,2,0).numpy()
    if res.shape[2]==1:
        return res[...,0]  
    elif res.shape[2]>3:
        #print(res.shape)
        #print(res[...,:3].shape)
        return res[...,:3]
    else:
        return res

vision.image.image2np = image2np


# In[11]:


il[0]


# In[12]:


# creating a stratified split of data and getting the indices
from sklearn.model_selection import StratifiedKFold
#train_idx, val_idx = next(iter(StratifiedKFold(n_splits=int(1/0.035),random_state=42).split(proc_train_df, proc_train_df.sirna)))
from sklearn.model_selection import train_test_split
train_df,val_df = train_test_split(proc_train_df,test_size=0.035, stratify = proc_train_df.sirna, random_state=42)
_proc_train_df = pd.concat([train_df,val_df])


# In[13]:


# creating databunch
data = (MultiChannelImageList.from_df(df=_proc_train_df,path='../input/recursion-cellular-image-classification/train')
        .split_by_idx(list(range(len(train_df),len(_proc_train_df))))
        .label_from_df()
        .transform(get_transforms(),size=256)
        .databunch(bs=128,num_workers=4)
        .normalize()
       )


# In[14]:


data.show_batch()


# In[15]:


get_ipython().system('pip install efficientnet_pytorch')
#!pip install cv2


# In[18]:


from efficientnet_pytorch import *


# In[17]:


import torchvision
RESNET_MODELS = {
    18: torchvision.models.resnet18,
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    
    152: torchvision.models.resnet152,
}

def resnet_multichannel(depth=50,pretrained=True,num_classes=1108,num_channels=6):
        model = RESNET_MODELS[depth](pretrained=pretrained)
        w = model.conv1.weight
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.conv1.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
        return model

    
DENSENET_MODELS = {
    121: torchvision.models.densenet121,
    161: torchvision.models.densenet161,
    169: torchvision.models.densenet169,
    201: torchvision.models.densenet201,
}

def densenet_multichannel(depth=121,pretrained=True,num_classes=1108,num_channels=6):
        model = DENSENET_MODELS[depth](pretrained=pretrained)
        w = model.features.conv0.weight
        model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.features.conv0.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
        return model
        
        
#EFFICIENTNET_MODELS = {
#    'b0': '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',
#    'b1': '../input/efficientnet-pytorch/efficientnet-b1-dbc7070a.pth',
#    'b2': '../input/efficientnet-pytorch/efficientnet-b2-27687264.pth',
#    'b3': '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth',
#    'b4': '../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth',
#    'b5': '../input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth'
#}


def efficientnet_multichannel(pretrained=True,name='b0',num_classes=1108,num_channels=6,image_size=256):
    model = EfficientNet.from_pretrained('efficientnet-'+name,num_classes=num_classes)
    #model.load_state_dict(torch.load(EFFICIENTNET_MODELS[name]))
    w = model._conv_stem.weight
    #s = model._conv_stem.static_padding
    model._conv_stem = utils.Conv2dStaticSamePadding(num_channels,32,kernel_size=(3, 3), stride=(2, 2), bias=False, image_size = image_size)
    model._conv_stem.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
    return model


# In[20]:


def resnet1(pretrained, num_channels=6):
    return resnet_multichannel(depth=18, pretrained=pretrained, num_channels=num_channels)
def _resnet_split(m):
    return (m[0][6], m[1])

def densenet161(pretrained, num_channels=6):
    return densenet_multichannel(depth=161, pretrained=pretrained, num_channels=num_channels)
def _densenet_split(m:nn.Module):
    return (m[0][0][7], m[1])

def efficientnetb0(pretrained=True, num_channels=6):
    return efficientnet_multichannel(pretrained=pretrained, name='b0', num_channels=num_channels)


# In[21]:


from fastai.metrics import *
learn = Learner(data, efficientnetb0(), metrics=[accuracy]).to_fp16()
learn.path = Path('../')


# In[22]:


# Let us unfreexe and train the entire model
learn.unfreeze()


# In[23]:


learn.fit_one_cycle(5, 1e-3)]

# this error comes ecause the data is not loading correctly
# need to find out more out it


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


learn.save('stage-2')
learn.export()


# In[ ]:


test_df = pd.read_csv("../input/recursion-cellular-image-classification/test")
proc_test_df = generate_df(test_df.copy())


# In[ ]:


data_test = MultiChannelImageList.from_df(df=proc_test_df, path='../input/recursion-cellular-image-classification/test')
learn.data.add_test(data_test)


# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds_ = preds.argmax(dim=-1)


# In[ ]:


test_df.head(10)


# In[ ]:


submission_df = pd.read_csv('../input/recursion-cellular-image-classification/submission.csv')


# In[ ]:


submission_df.sirna = preds_.numpy().astype(int)
submission_df.head(10)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

