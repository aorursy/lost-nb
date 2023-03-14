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
os.mkdir('../tmp')
os.mkdir('../tmp/images1')
os.mkdir('../tmp/images2')
os.mkdir('../tmp/imagesr')
os.mkdir('../tmp/images1/dogs')
os.mkdir('../tmp/images2/dogs')
os.mkdir('../tmp/imagesr/dogs')

# Any results you write to the current directory are saved as output.





# from https://www.kaggle.com/whizzkid/crop-images-using-bounding-boximport numpy as np # linear algebra
import xml.etree.ElementTree as ET # for parsing XML
import matplotlib.pyplot as plt # to show images
from PIL import Image # to read images
import os
import glob
import cv2
import numpy as np

root_images="../input/all-dogs/all-dogs/"
root_annots="../input/annotation/Annotation/"

path1 = '../tmp/images1/dogs/'
path2 = '../tmp/images2/dogs/'
pathr = '../tmp/imagesr/dogs/'

all_images=os.listdir("../input/all-dogs/all-dogs/")

breeds = glob.glob('../input/annotation/Annotation/*')
annotation=[]
for b in breeds:
    annotation+=glob.glob(b+"/*")

breed_map={}
for annot in annotation:
    breed=annot.split("/")[-2]
    index=breed.split("-")[0]
    breed_map.setdefault(index,breed)
    
def resizeAndCrop(img, size):

    h, w = img.shape[:2]
    sh = size
    sw = size


    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
    elif aspect < 1: # vertical image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
    else: # square image
        new_h, new_w = sh, sw
    # scale and crop
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if aspect > 1: # horizontal image
        middle = np.floor(new_w/2).astype(int)
        crop_img = scaled_img[0:64, middle-32:middle+32]
    elif aspect < 1: # vertical image
        middle = np.floor(new_h/2).astype(int)
        crop_img = scaled_img[middle-32:middle+32, 0:64]
    else:
        crop_img = scaled_img
    return crop_img


def bounding_box(image):
    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])
    tree = ET.parse(bpath)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)      
    return (xmin,ymin,xmax,ymax)

def square_bounding(xmin,ymin,xmax,ymax, w, h):
    xmin2 = xmin
    xmax2 = xmax
    ymin2 = ymin
    ymax2 = ymax
    if(xmax - xmin > ymax - ymin):
        bh = xmax - xmin
        xmin2 = xmin
        xmax2 = xmax
        middle = (ymin + ymax)/2
        ymin2 = middle - (bh/2)
        marginmin = ymin2
        ymax2 = middle + (bh/2)
        marginmax = h - ymax2
        if(marginmin + marginmax < 0):
            ymin2 = 0
            ymax2 = h
            middle2 = (xmin + xmax)/2
            xmin2 = middle2 - (h/2)
            xmax2 = middle2 + (h/2)
        elif(marginmin < 0):
            ymax2 = ymax2 - marginmin
            ymin2 = 0
        elif(marginmax < 0):
            ymin2 = ymin2 + marginmax
            ymax2 = h
    elif(xmax - xmin < ymax - ymin):
        bw = ymax - ymin
        ymin2 = ymin
        ymax2 = ymax
        middle = (xmin + xmax)/2
        xmin2 = middle - (bw/2)
        marginmin = xmin2
        xmax2 = middle + (bw/2)
        marginmax = w - xmax2
        if(marginmin + marginmax < 0):
            xmin2 = 0
            xmax2 = w
            middle2 = (ymin + ymax)/2
            ymin2 = middle2 - (w/2)
            ymax2 = middle2 + (w/2)
        elif(marginmin < 0):
            xmax2 = xmax2 - marginmin
            xmin2 = 0
        elif(marginmax < 0):
            xmin2 = xmin2 + marginmax
            xmax2 = w
            
        
    return (xmin2,ymin2,xmax2,ymax2)

for i,image in enumerate(all_images):
    xmin, ymin, xmax, ymax=bounding_box(image)
    image=Image.open(os.path.join(root_images,image))
    w, h = image.size
    bbox2 = square_bounding(xmin, ymin, xmax, ymax, w, h)
    image=image.crop(bbox2)
    image = cv2.resize(np.uint8(image), (64, 64), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image,(8,8), interpolation = cv2.INTER_LINEAR)
    image1 = cv2.resize(image1,(64,64), interpolation = cv2.INTER_NEAREST)
    image2 = image
    image2 = np.subtract(image2, image1)
    cv2.imwrite(path1 + str(i) + '.png', image1)
    cv2.imwrite(path2 + str(i) + '.png', image2)
    cv2.imwrite(pathr + str(i) + '.png', image)




import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import cv2
import PIL
from PIL import Image
import imageio
import torchvision.transforms.functional as TF
get_ipython().run_line_magic('matplotlib', 'inline')

batch_size = 32

# 64x64 images!
transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data1 = datasets.ImageFolder('../tmp/images1', transform=transform)
train_loader1 = torch.utils.data.DataLoader(train_data1, shuffle=True, batch_size=batch_size)
train_data2 = datasets.ImageFolder('../tmp/imagesr', transform=transform)
train_loader2 = torch.utils.data.DataLoader(train_data2, shuffle=True, batch_size=batch_size)
                                           
imgs1, label1 = next(iter(train_loader1))
imgs1 = imgs1.numpy().transpose(0, 2, 3, 1)
imgs2, label2 = next(iter(train_loader2))
imgs2 = imgs2.numpy().transpose(0, 2, 3, 1)

#in fact it's like https://arxiv.org/abs/1506.05751 ?

class Generator1(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator1, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        self.conv2 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8
        
        self.conv3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16
        
        self.conv4 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64
        
        self.conv6 = nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False)
        # state size. (nchannels) x 64 x 64

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = torch.tanh(self.conv6(x))
        
        return x

class Generator2(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator2, self).__init__()
        
        self.avg1 = nn.AvgPool2d(8);

         # input is Z, going into a convolution
        #self.conv1 = nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)
        #self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        #self.conv2 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8
        
        self.convn2 = nn.ConvTranspose2d(3, nfeats * 8, 5, 1, 2, bias=False)
        self.bnn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*4) x 8 x 8
        
        self.convn3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 5, 1, 2, bias=False)
        self.bnn3 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*4) x 8 x 8
        
        #
        
        self.conv3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16
        
        self.dp1 = nn.Dropout2d(p=0.15)
        
        self.conv4 = nn.ConvTranspose2d(nfeats * 4, nfeats * 4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats) x 64 x 64
        
        self.convn1 = nn.ConvTranspose2d(nfeats * 2, nfeats * 2, 7, 1, 3, bias=False)
        self.bnn1 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats) x 64 x 64
        
        self.convn4 = nn.ConvTranspose2d(nfeats * 2, nfeats, 3, 1, 1, bias=False)
        self.bnn4 = nn.BatchNorm2d(nfeats)
        
        self.conv6 = nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False)
        # state size. (nchannels) x 64 x 64

    def forward(self, x):
        #x = F.leaky_relu(self.bn1(self.conv1(x)))
        #x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.avg1(x))
        #print('A', x.size())
        x = F.leaky_relu(self.bnn2(self.convn2(x)))
        #print('A', x.size())
        x = F.leaky_relu(self.bnn3(self.convn3(x)))
        #print('A', x.size())
        x = F.leaky_relu(self.bn3(self.conv3(self.dp1(x))))
        #print('A', x.size())
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        #print('A', x.size())
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        #print('A', x.size())
        #x = F.leaky_relu(self.bnn1(self.convn1(x)))
        #print('A', x.size())
        x = F.leaky_relu(self.bnn4(self.convn4(x)))
        x = torch.tanh(self.conv6(x))
        
        return x


class Discriminator1(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator1, self).__init__()
        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
        
        self.dp1 = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16
        
        self.conv3 = nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
       
        self.conv4 = nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        
        self.conv5 = nn.Conv2d(nfeats * 8, 1, 4, 1, 0, bias=False)
        # state size. 1 x 1 x 1
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(self.dp1(x))), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        #x = self.conv5(x) # RaLSGAN
        
        return x.view(-1, 1)
    
class Discriminator2(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator2, self).__init__()

       # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
                         
        self.dp1 = nn.Dropout2d(p=0.25)

        
        self.conv2 = nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16
        
        self.conv3 = nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
       
        self.conv4 = nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        
        self.conv5 = nn.Conv2d(nfeats * 8, 1, 4, 1, 0, bias=False)
        # state size. 1 x 1 x 1
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(self.dp1(x))), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        #x = self.conv5(x)
        
        return x.view(-1, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0003
beta1 = 0.5

netG1 = Generator1(128, 32, 3).to(device)
netD1 = Discriminator1(3, 48).to(device)

netG2 = Generator2(3, 48, 3).to(device)
netD2 = Discriminator2(3, 48).to(device)

criterion = nn.BCELoss()

optimizerD1 = optim.Adam(netD1.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG1 = optim.Adam(netG1.parameters(), lr=lr, betas=(beta1, 0.999))

optimizerD2 = optim.Adam(netD2.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG2 = optim.Adam(netG2.parameters(), lr=lr, betas=(beta1, 0.999))

nz = 128
nz2 = 64*64*3
fixed_noise = torch.randn(25, nz, 1, 1, device=device)
fixed_noise2 = torch.randn(25, nz2, 1, 1, device=device)

real_label = 0.9
fake_label = 0
batch_size = train_loader1.batch_size



 
# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y


### training here

epochs = 70

step = 0
for epoch in range(epochs):
    for ii, (real_images, train_labels) in enumerate(train_loader1):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        netD1.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        #labels = noisy_labels(labels, 0.01)
        output = netD1(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()   

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG1(noise)
        labels.fill_(fake_label)
        #labels = noisy_labels(labels, 0.01)
        output = netD1(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD1.step()
        '''
        netD1.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        #labels = noisy_labels(labels, 0.01)
        outputR = netD1(real_images)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG1(noise)
        outputF = netD1(fake.detach())
        errD = (torch.mean((outputR - torch.mean(outputF) - labels) ** 2) + 
                torch.mean((outputF - torch.mean(outputR) + labels) ** 2))/2
        errD.backward(retain_graph=True)
        optimizerD1.step()
        '''

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        netG1.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD1(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG1.step()
        '''
        netG1.zero_grad()
        outputF = netD1(fake)   
        errG = (torch.mean((outputR - torch.mean(outputF) + labels) ** 2) +
                torch.mean((outputF - torch.mean(outputR) - labels) ** 2))/2
        errG.backward()
        optimizerG1.step()
        '''
        if step % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, epochs, ii, len(train_loader1),
                     errD.item(), errG.item()))
            
            #valid_image = netG1(fixed_noise)
        step += 1

        
epochs = 350

step = 0
        
for epoch in range(epochs):
    for ii, (real_images, train_labels) in enumerate(train_loader2):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        netD2.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        #labels = noisy_labels(labels, 0.01)
        output = netD2(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        #noise = torch.randn(batch_size, nz2, 1, 1, device=device)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        intermediate = netG1(noise)
        #print(intermediate.size())
        #intermediaten = intermediate.cpu().detach().numpy()
        #intermediate2 = np.reshape(intermediaten, (32, 64*64*3, 1, 1))
        #fake = netG2(torch.from_numpy(intermediate2).to(device))
        fake = netG2(intermediate.detach())
        #print(fake.size())
        labels.fill_(fake_label)
        #labels = noisy_labels(labels, 0.01)
        output = netD2(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD2.step()
        '''
        netD2.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        #labels = noisy_labels(labels, 0.01)
        outputR = netD2(real_images)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        intermediate = netG1(noise)
        fake = netG2(intermediate.detach())
        outputF = netD2(fake.detach())
        errD = (torch.mean((outputR - torch.mean(outputF) - labels) ** 2) + 
                torch.mean((outputF - torch.mean(outputR) + labels) ** 2))/2
        errD.backward(retain_graph=True)
        optimizerD2.step()
        '''
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        netG2.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD2(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG2.step()
        '''
        netG2.zero_grad()
        outputF = netD2(fake)   
        errG = (torch.mean((outputR - torch.mean(outputF) + labels) ** 2) +
                torch.mean((outputF - torch.mean(outputR) - labels) ** 2))/2
        errG.backward()
        optimizerG2.step()
        '''
        if step % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, epochs, ii, len(train_loader2),
                     errD.item(), errG.item()))
            
            #valid_image = netG2(fixed_noise2)
        step += 1
        
        
# torch.save(netG.state_dict(), 'generator.pth')
# torch.save(netD.state_dict(), 'discriminator.pth')


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 50
n_images=10000
for i_batch in range(0, n_images, im_batch_size):
    gen_z = torch.randn(im_batch_size, 128, 1, 1, device=device)
    #gen_z2 = torch.randn(im_batch_size, 64*64*3, 1, 1, device=device)
    gen_images1 = netG1(gen_z)
    gen_images2 = netG2(gen_images1.detach())
    images1 = gen_images1.to("cpu").clone().detach()
    images1 = images1.numpy().transpose(0, 2, 3, 1)
    images2 = gen_images2.to("cpu").clone().detach()
    images2 = images2.numpy().transpose(0, 2, 3, 1) 
    for i_image in range(gen_images2.size(0)):
        #image = 255.0 * np.add(images1[i_image, :, :, :],images2[i_image, :, :, :])
        #print('A', images2[i_image, :, :, :])
        #print(images2.shape)
        image = 127.5 * (1 + images2[i_image, :, :, :])
        #print(image.shape)
        #print('B', image)
        #save_image(images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
        #imageio.imwrite('../output_images/' + str(i_batch + i_image) + 'X.png', image)
        cv2.imwrite('../output_images/' + str(i_batch + i_image) + '.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        '''
        imageR1 = cv2.imread('../output_images/' + str(i_batch + i_image) + '.png')
        plt.imshow(imageR1)
        plt.show()
        print('IM')
        imageR2 = cv2.imread('../output_images/' + str(i_batch + i_image) + 'X.png')
        plt.imshow(imageR2)
        plt.show()
        '''
        


import shutil
shutil.make_archive('images', 'zip', '../output_images')

