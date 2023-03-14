#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali


# In[3]:


import time
import glob
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
class BasicDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob('../input/severstal-steel-defect-detection/train_images/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image


# In[4]:


transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_loader = DataLoader(
                BasicDataset(transform=transform),
                    batch_size=64, shuffle=True, num_workers=2)


# In[5]:


get_ipython().run_cell_magic('time', '', 'start_time = time.time()\nfor image in tqdm(data_loader):\n    image = image.cuda()\n    pass\nbasic_time = time.time() - start_time')


# In[6]:


get_ipython().system('apt-get install libturbojpeg0')
get_ipython().system('pip install jpeg4py')


# In[7]:


import jpeg4py as jpeg
class jpeg4pyDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob('../input/severstal-steel-defect-detection/train_images/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = jpeg.JPEG(self.img_list[idx]).decode()
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image


# In[8]:


transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_loader = DataLoader(
                jpeg4pyDataset(transform=transform),
                    batch_size=64, shuffle=True, num_workers=2)


# In[9]:


get_ipython().run_cell_magic('time', '', 'start_time = time.time()\nfor image in tqdm(data_loader):\n    image = image.cuda()\n    pass\njpeg4py_time = time.time() - start_time')


# In[10]:


import torch


# In[11]:


import albumentations
class jpeg4pyalbDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob('../input/severstal-steel-defect-detection/train_images/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = jpeg.JPEG(self.img_list[idx]).decode()

        if self.transform is not None:
            image = self.transform(**{"image": image})

        return torch.from_numpy(image['image'])


# In[12]:


transform = albumentations.Compose([
                        albumentations.Resize(height=224, width=224, interpolation=1, always_apply=True, p=1),
                        albumentations.Flip(always_apply=False, p=0.5),
                        albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True, p=1.0)])
data_loader = DataLoader(
                jpeg4pyalbDataset(transform=transform),
                    batch_size=64, shuffle=True, num_workers=2)


# In[13]:


get_ipython().run_cell_magic('time', '', 'start_time = time.time()\nfor image in tqdm(data_loader):\n    image = image.cuda()\n    pass\njpeg4pyalb_time = time.time() - start_time')


# In[14]:


import jpeg4py as jpeg
class jpeg4pyDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob('../input/severstal-steel-defect-detection/train_images/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = jpeg.JPEG(self.img_list[idx]).decode()
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image


# In[15]:


from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types


# In[16]:


class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.img_list = glob.glob('../input/severstal-steel-defect-detection/train_images/*.jpg')

        dummy_label = [0]*len(self.img_list)
        df = pd.DataFrame({'data' : self.img_list, 'label' : dummy_label})
        df.to_csv('dali.txt', header=False, index=False, sep=' ')
        
        self.input = ops.FileReader(file_root='.', file_list='dali.txt')
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 resize_x=224., resize_y=224.)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            mirror = 1,
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.decode(images)
        images = self.resize(images)
        output = self.cmn(images)
        return output


# In[17]:


class DALICustomIterator(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False):
        super(DALICustomIterator, self).__init__(pipelines, output_map, size, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __len__(self):
        return int(self._size / self.batch_size) + 1

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        feed = super().__next__()
        data = feed[0]['data']
        return data


# In[18]:


def DALIDataLoader(batch_size):
    num_gpus = 1
    pipes = [DALIPipeline(batch_size=batch_size, num_threads=2, device_id=device_id) for device_id in range(num_gpus)]

    pipes[0].build()
    dali_iter = DALICustomIterator(pipes, ['data'], pipes[0].epoch_size("Reader"), auto_reset=True)
    return dali_iter
data_loader = DALIDataLoader(batch_size=64)


# In[19]:


get_ipython().run_cell_magic('time', '', 'start_time = time.time()\nfor image in tqdm(data_loader):\n    # image is already on GPU\n    image = image\n    pass\ndali_time = time.time() - start_time')


# In[20]:


print('simple data loader         : {}'.format(basic_time))
print('jpeg4py data loader        : {}'.format(jpeg4py_time))
print('jpeg4py + alb data loader  : {}'.format(jpeg4pyalb_time))
print('dali data loader           : {}'.format(dali_time))

