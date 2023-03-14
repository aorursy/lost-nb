#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import shutil
from dataclasses import dataclass
from functools import partial
import pathlib
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import skimage
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from tqdm import tqdm_notebook as tqdm


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm_notebook as tqdm
from time import time
from PIL import Image
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import xml.etree.ElementTree as ET
import random
from torch.nn.utils import spectral_norm
from scipy.stats import truncnorm
import torch as th


# In[3]:


@dataclass
class Example:
    img: np.ndarray
    category: str
    difficult: int
    transform: object = None

    def read_img(self):
        # read and crop
        img = self.img

        if self.transform is not None:
            img = self.transform(image=img)['image']

        # convert to ndarray and transpose HWC => CHW
        img = np.array(img).transpose(2, 0, 1)
        return img

    def show(self):
        img = (self.read_img() + 1) / 2
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(self.category)

class DogDataset(Dataset):
    def __init__(self,
                 img_dir='../input/all-dogs/all-dogs/',
                 anno_dir='../input/annotation/Annotation/',
                 transform=None,
                 examples=None):
        self.img_dir = pathlib.Path(img_dir)
        self.anno_dir = pathlib.Path(anno_dir)
        self.transform = transform
        self.preprocess = albu.Compose([
            albu.SmallestMaxSize(64),
        ])
        
        if examples is None:
            self.examples = self._correct_examples()
        else:
            self.examples = examples

        self.categories = sorted(set([e.category for e in self.examples]))
        self.categ2id = dict(zip(self.categories, range(len(self.categories))))
        
    def _correct_examples(self):
        examples = []
        for anno in tqdm(list(self.anno_dir.glob('*/*'))):
            tree = ET.parse(anno)
            root = tree.getroot()

            img_path = self.img_dir / f'{root.find("filename").text}.jpg'
            if not img_path.exists():
                continue

            objects = root.findall('object')
            for obj in objects:
                examples.append(self._create_example(img_path, obj))
        return examples

    def _create_example(self, img_path, obj):
        # reading bound box
        bbox = obj.find('bndbox')
        # read and preprocess image
        img = skimage.io.imread(img_path)
        xmin=int(bbox.find('xmin').text)
        ymin=int(bbox.find('ymin').text)
        xmax=int(bbox.find('xmax').text)
        ymax=int(bbox.find('ymax').text)
        img = img[ymin:ymax, xmin:xmax]
        img = self.preprocess(image=img)['image']

        # add example
        return Example(
            img=img,
            category=obj.find('name').text,
            difficult=int(obj.find('difficult').text),
            transform=self.transform,
        )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        e = self.examples[i]
        img = e.read_img()

        categ_id = self.categ2id[e.category]
        return img, categ_id

    def show_examples(self, indices=None, n_cols=8):
        if indices is None:
            indices = np.random.randint(0, len(self), n_cols)

        n_rows = (len(indices)-1) // n_cols + 1

        fig = plt.figure(figsize=(n_cols*4, n_rows*4))
        for i, idx in enumerate(indices, 1):
            fig.add_subplot(n_rows, n_cols, i)
            self.examples[idx].show()
        plt.show()


# In[4]:


start = time()


# In[5]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[6]:


import albumentations.pytorch


# In[7]:


get_ipython().run_cell_magic('time', '', '\ntransform = albu.Compose([\n    albu.CenterCrop(64, 64),\n    albu.HorizontalFlip(p=0.5),\n    albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),\n])\n\ndataset = DogDataset(\n    transform=transform,\n)\n\nprint(dataset)\ndataset.show_examples()')


# In[8]:


ls /opt/conda/lib/python3.6/site-packages/cv2/data/haarcascade_*.xml


# In[9]:


def crop_with_facebox(img, pos):
    pos = pos[0]
    left, right = pos[0], pos[0]+pos[2]
    top, bottom = pos[1], pos[1]+pos[3]
    img = img[top:bottom, left:right]
    return img

def create_cropped_dataset(dataset):
    rescale = albu.SmallestMaxSize(64)

    cascade_file_path = '/opt/conda/lib/python3.6/site-packages/cv2/data/haarcascade_{}.xml'
    
    # classifiers are prioritized
    ordered_classifiers = [
        (cv2.CascadeClassifier(cascade_file_path.format('frontalcatface_extended')), crop_with_facebox),
        (cv2.CascadeClassifier(cascade_file_path.format('frontalcatface')), crop_with_facebox),
        (cv2.CascadeClassifier(cascade_file_path.format('frontalface_default')), crop_with_facebox),
        (cv2.CascadeClassifier(cascade_file_path.format('frontalface_alt')), crop_with_facebox),
        (cv2.CascadeClassifier(cascade_file_path.format('frontalface_alt2')), crop_with_facebox),
    ]

    cropped_examples = []
    for i, e in enumerate(tqdm(dataset.examples)):
        grayimg = cv2.cvtColor(e.img, cv2.COLOR_RGB2GRAY)
        for clf, crop_fn in ordered_classifiers:
            pos = clf.detectMultiScale(grayimg)
            if len(pos) != 0:
                break

        if len(pos) == 0:
            continue

        img = crop_fn(e.img, pos)
        if img is None:
            continue

        img = rescale(image=img)['image']
        cropped_examples.append(Example(
            img=img, category=e.category,
            difficult=e.difficult,
            transform=e.transform,
        ))

    return DogDataset(examples=cropped_examples)


# In[10]:


get_ipython().run_cell_magic('time', '', 'cropped_dataset = create_cropped_dataset(dataset)\nprint(len(cropped_dataset))\ncropped_dataset.show_examples()')


# In[11]:


batch_size = 32
train_loader = torch.utils.data.DataLoader(cropped_dataset, shuffle=True,batch_size=batch_size, num_workers = 4)


# In[12]:


# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------
class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


# In[13]:


def show_generated_img_all():
    gen_z = torch.randn(32, nz, 1, 1, device=device)
    gen_images = netG(gen_z).to("cpu").clone().detach()
    gen_images = gen_images.numpy().transpose(0, 2, 3, 1)
    gen_images = (gen_images+1.0)/2.0
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(gen_images):
        ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
    #plt.savefig(filename)  


# In[14]:


### This is to show one sample image for iteration of chosing
def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = ((gen_image+1.0)/2.0)
    plt.imshow(gen_image)
    plt.show()


# In[15]:


class MinibatchStdDev(th.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = th.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size,1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = th.cat([x, y], 1)
        # return the computed values:
        return y


# In[16]:


class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))
        #self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        self.conv2 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))
        #self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8
        
        self.conv3 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))
        #self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16
        
        self.conv4 = spectral_norm(nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))
        #self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = spectral_norm(nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))
        #self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64
        
        self.conv6 = spectral_norm(nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()
    def forward(self, x):
        #x = F.leaky_relu(self.bn1(self.conv1(x)))
        #x = F.leaky_relu(self.bn2(self.conv2(x)))
        #x = F.leaky_relu(self.bn3(self.conv3(x)))
        #x = F.leaky_relu(self.bn4(self.conv4(x)))
        #x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))
        
        return x



class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
        
        self.conv2 = spectral_norm(nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16
        
        self.conv3 = spectral_norm(nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
       
        self.conv4 = spectral_norm(nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False))
        self.bn4 = nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        self.conv5 = spectral_norm(nn.Conv2d(nfeats * 8 +1, 1, 2, 1, 0, bias=False))
        # state size. 1 x 1 x 1
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
       # x = self.pixnorm(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
       # x = self.pixnorm(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
       # x = self.pixnorm(x)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        #x= self.conv5(x)
        return x.view(-1, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0003
lr_d = 0.0001
beta1 = 0.5
#epochs = 900
epochs = 14000
netG = Generator(100, 32, 3).to(device)
netD = Discriminator(3, 48).to(device)

criterion = nn.BCELoss()
#criterion = nn.MSELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_d, betas=(beta1, 0.999))
lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
                                                                     T_0=epochs//200, eta_min=0.00005)
lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,
                                                                     T_0=epochs//200, eta_min=0.00005)

nz = 100
fixed_noise = torch.randn(25, nz, 1, 1, device=device)

real_label = 0.7
fake_label = 0.0
batch_size = train_loader.batch_size



### training here


step = 0
for epoch in range(epochs):
    for ii, (real_images,_) in tqdm(enumerate(train_loader), total=len(train_loader)):
        end = time()
        if (end -start) > 31800:
            break
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device) +  np.random.uniform(-0.1, 0.1)

        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        labels.fill_(fake_label) + np.random.uniform(0, 0.2)
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        if step % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            valid_image = netG(fixed_noise)
        step += 1
        lr_schedulerG.step(epoch)
        lr_schedulerD.step(epoch)

    if epoch % 200 == 0:
        show_generated_img()
        
# torch.save(netG.state_dict(), 'generator.pth')
# torch.save(netD.state_dict(), 'discriminator.pth')

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 100
n_images=10000
for i_batch in range(0, n_images, im_batch_size):
    z = truncated_normal((im_batch_size, 100, 1, 1), threshold=1)
    gen_z = torch.from_numpy(z).float().to(device)    
    #gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
    gen_images = netG(gen_z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image((gen_images[i_image, :, :, :] +1.0)/2.0, os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


import shutil
shutil.make_archive('images', 'zip', '../output_images')


# In[ ]:




