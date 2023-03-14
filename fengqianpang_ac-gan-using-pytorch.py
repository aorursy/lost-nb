#!/usr/bin/env python
# coding: utf-8



from __future__ import print_function, division
import os, time, glob, argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.utils import save_image
from tqdm import tqdm_notebook as tqdm
from PIL import Image

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ion()   # interactive mode
plt.rcParams['image.interpolation'] = 'nearest'
multiGPU = False




rootpath = "../input/dog-breed-identification/"
TRAIN_IMG_PATH = rootpath + "train"
TEST_IMG_PATH = rootpath + "test"
LABELS_CSV_PATH = rootpath + "labels.csv"
SAMPLE_SUB_PATH = rootpath + "sample_submission.csv"




# 重要参数
workers = 4
batch_size = 32
image_size = 128   #  try 64 and 128
nc = 3          # number of channels, RGB image is 3
nz = 128
ngf = 64
ndf = 64
num_epochs = 300 # 循环次数
lr = 0.001      # learning rate
beta1 = 0.5     # momentum
ngpu = 1        # nubmer of gpu
num_show = 6
n_class = 120   # number of classes

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)




class DogsDataset(Dataset):
    """Dog breed identification dataset."""

    def __init__(self, img_dir, dataframe, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.        
            dataframe (pandas.core.frame.DataFrame): Pandas dataframe obtained
                by read_csv().
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.id[idx]) + ".jpg"
        image = Image.open(img_name)
        label = self.labels_frame.target[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label]




dframe = pd.read_csv(LABELS_CSV_PATH)
labelnames = pd.read_csv(SAMPLE_SUB_PATH).keys()[1:]
codes = range(len(labelnames))
breed_to_code = dict(zip(labelnames, codes))
code_to_breed = dict(zip(codes, labelnames))
dframe['target'] =  [breed_to_code[x] for x in dframe.breed]

cut = int(len(dframe)*0.8)
train, test = np.split(dframe, [cut], axis=0)
test = test.reset_index(drop=True)

train_ds = DogsDataset(TRAIN_IMG_PATH, train)
test_ds = DogsDataset(TRAIN_IMG_PATH, test)
idx = 29
plt.imshow(train_ds[idx][0])
print(code_to_breed[train_ds[idx][1]])
print("Shape of the image is: ", train_ds[idx][0].size)




# change random crop to resize+center crop
data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])




train_ds = DogsDataset(TRAIN_IMG_PATH, train, data_transform)
test_ds = DogsDataset(TRAIN_IMG_PATH, test, data_transform)
datasets = {"train": train_ds, "val": test_ds}

idx = 29
print(code_to_breed[train_ds[idx][1]])
print("Shape of the image is: ", train_ds[idx][0].shape)




trainloader = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

testloader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

#dataloaders = {"train": trainloader, "val": testloader}
dataloaders = [trainloader, testloader]




# 网络模型参数的初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




# 生成网络定义
class Generator(nn.Module):

    def __init__(self, ngpu, nz=nz, ngf=ngf, nc=nc, n_class=n_class):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz+n_class, ngf * 16, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 16)

        self.conv2 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 8)

        self.conv3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 4)

        self.conv4 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ngf * 2)
        
        self.conv5 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False)
        self.BatchNorm5 = nn.BatchNorm2d(ngf * 1)

        self.conv6 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)

        self.apply(weights_init)


    def forward(self, input):

        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)
        
        x = self.conv5(x)
        x = self.BatchNorm5(x)
        x = self.ReLU(x)

        x = self.conv6(x)
        output = self.Tanh(x)
        return output
    
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)




# 鉴别网络定义
class Discriminator(nn.Module):

    def __init__(self, ngpu, ndf=ndf, nc=nc, n_class=n_class):

        super(Discriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.BatchNorm5 = nn.BatchNorm2d(ndf * 16)
        self.conv6 = nn.Conv2d(ndf * 16, ndf * 1, 4, 1, 0, bias=False)
        self.disc_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, n_class)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)
        #print(x.shape)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)
        #print(x.shape)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)
        #print(x.shape)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)
        #print(x.shape)
        
        x = self.conv5(x)
        x = self.BatchNorm5(x)
        x = self.LeakyReLU(x)
        #print(x.shape)

        x = self.conv6(x)
        #print(x.shape)
        
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)
        c = self.softmax(c)
        s = self.disc_linear(x)
        s = self.sigmoid(s)
        return s,c

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)




# Setup Adam optimizers for both G and D  优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))




# Loss functions 损失函数
s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()




# 展示生成图像
def show_generated_img(num_show):
    gen_images = []
    for _ in range(num_show):
        noise = torch.randn(1, nz, 1, 1, device=device)
        dog_label = torch.randint(0, n_class, (1, ), device=device)
        gen_image = concat_noise_label(noise, dog_label, device)
        gen_image = netG(gen_image).to("cpu").clone().detach().squeeze(0)
        # gen_image = gen_image.numpy().transpose(0, 2, 3, 1)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        gen_images.append(gen_image)
        
    fig = plt.figure(figsize=(10, 5))
    for i, gen_image in enumerate(gen_images):
        ax = fig.add_subplot(1, num_show, i + 1, xticks=[], yticks=[])
        plt.imshow(gen_image + 1 / 2)
    plt.show()




def onehot_encode(label, device, n_class=n_class):  
    eye = torch.eye(n_class, device=device) 
    return eye[label].view(-1, n_class, 1, 1)   
 
def concat_image_label(image, label, device, n_class=n_class):
    B, C, H, W = image.shape   
    oh_label = onehot_encode(label, device=device)
    oh_label = oh_label.expand(B, n_class, H, W)
    return torch.cat((image, oh_label), dim=1)
 
def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device=device)
    return torch.cat((noise, oh_label), dim=1)




r_label = 0.7
f_label = 0

input = torch.tensor([batch_size, nc, image_size, image_size], device=device)
noise = torch.tensor([batch_size, nz, 1, 1], device=device)

fixed_noise = torch.randn(1, nz, 1, 1, device=device)
fixed_label = torch.randint(0, n_class, (1, ), device=device)
fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device)




# Training Loop

# Lists to Keep track pf progress
G_losses, D_losses = [], []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    # training train and test datasets 
    for dataloader in dataloaders:  
        for i, data in enumerate(dataloader):
            # prepare real image and label
            real_label = data[1].cuda()
            real_image = data[0].cuda()
            b_size = real_label.size(0)      
        
            # prepare fake image and label
            fake_label = torch.randint(n_class, (b_size,), dtype=torch.long, device=device)
            noise = torch.randn(b_size, nz, 1, 1, device=device).squeeze(0)
            noise = concat_noise_label(noise, real_label, device)  
            fake_image = netG(noise)
        
            # target
            real_target = torch.full((b_size,), r_label, device=device)
            fake_target = torch.full((b_size,), f_label, device=device)
            
            #-----------------------
            # Update Discriminator
            #-----------------------
            netD.zero_grad()
        
            # train with real
            s_output, c_output = netD(real_image)
            
            print(s_output.shape, real_target.shape)
            s_errD_real = s_criterion(s_output, real_target)  # realfake
            c_errD_real = c_criterion(c_output, real_label)  # class
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()
            D_x = s_output.data.mean()

            # train with fake
            s_output,c_output = netD(fake_image.detach())
            s_errD_fake = s_criterion(s_output, fake_target)  # realfake
            c_errD_fake = c_criterion(c_output, real_label)  # class
            errD_fake = s_errD_fake + c_errD_fake
            errD_fake.backward()
            D_G_z1 = s_output.data.mean()
        
            errD = s_errD_real + s_errD_fake
            optimizerD.step()        

            #-----------------------
            # Update Generator
            #-----------------------
            netG.zero_grad()
        
            s_output,c_output = netD(fake_image)
            s_output.shape
            real_target.shape
            s_errG = s_criterion(s_output, real_target)  # realfake
            c_errG = c_criterion(c_output, real_label)  # class
            errG = s_errG + c_errG
            errG.backward()
            D_G_z2 = s_output.data.mean()
        
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
            iters += 1
    
    print('[%d/%d][%d/%d]\nLoss_D: %.4f\tLoss_G: %.4f\nD(x): %.4f\tD(G(z)): %.4f / %.4f'
          % (epoch+1, num_epochs, i+1, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))      
    
    if epoch%5 ==0:
        show_generated_img(num_show)

#     # --------- save fake image  ----------
#     fake_image = netG(fixed_noise_label)   
#     vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
#                     normalize=True, nrow=5)
 
#     # ---------  save model  ----------
#     if (epoch + 1) % 10 == 0:  
#         torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(outf, epoch + 1))
#         torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(outf, epoch + 1))




def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def analyse_generated_by_class(n_images):
    good_breeds = []
    for l in range(n_class):
        sample = []
        for _ in range(n_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            dog_label = l
            noise_label = concat_noise_label(noise, dog_label, device)
            gen_image = netG(noise_label).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)

        d = np.round(np.sum([mse(sample[k], sample[k + 1]) for k in range(len(sample) - 1)]) / n_images, 1,)
        if d < 1.0:
            continue  # had mode colapse(discard)
            
        print(f"Generated breed({d}): ", code_to_breed[l])    
        good_breeds.append(l)
    return good_breeds




def show_generated_img_all():
    dog_label = torch.randint(n_class, (64,), dtype=torch.long, device=device)
    noise = torch.randn(64, nz, 1, 1, device=device)
    gen_image = concat_noise_label(noise, dog_label, device)  
    gen_image = netG(gen_image).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(0, 2, 3, 1)
    # gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = (gen_image + 1.0) / 2.0
    
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(gen_image):
        ax = fig.add_subplot(8, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
        
def show_loss(ylim): 
    sns.set_style("white")
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Generator and Discriminator Loss During Training")
    ax.plot(G_losses,label="G",c="b")
    ax.plot(D_losses,label="D",c="r")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    if ylim == True:
        ax.set_ylim(0,4)




def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def create_submit(good_breeds):
    print("Creating submit")
    os.makedirs("../output_images", exist_ok=True)
    im_batch_size = 100
    n_images = 10000

    for i_batch in range(0, n_images, im_batch_size):
        z = truncated_normal((im_batch_size, nz, 1, 1), threshold=1)
        noise = torch.from_numpy(z).float().to(device)
        
        dog_label = np.random.choice(good_breeds, size=im_batch_size, replace=True) 
        dog_label = torch.from_numpy(dog_label).to(device).clone().detach().squeeze(0)
        noise_label = concat_noise_label(noise, dog_label, device)
    
        gen_images = (netG(noise_label) + 1) / 2
        
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))

    import shutil
    shutil.make_archive("images", "zip", "../output_images")




# loss curve

show_loss(ylim=False)




# loss curve

show_loss(ylim=True)




# analysis
good_breeds = analyse_generated_by_class(6)
#create_submit(good_breeds)




show_generated_img_all()




# visualization generate image of all breeds 

fig = plt.figure(figsize=(20,40))
for i in range(n_class):
    ax = fig.add_subplot(20,6,i+1)
    noise = torch.randn(1, nz, 1, 1, device=device)
    dog_label = i
    noise_label = concat_noise_label(noise, dog_label, device)
    gen_image = netG(noise_label).to("cpu").clone().detach().squeeze(0)
    # gen_image = gen_image.numpy().transpose(0, 2, 3, 1)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = (gen_image + 1.0) / 2.0
    ax.axis('off')
    ax.set_title(code_to_breed[i])
    ax.imshow(gen_image, cmap="gray")
plt.tight_layout()
plt.show() 






