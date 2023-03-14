#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('bash', '', 'pip3 install --quiet torchsummary')


# In[2]:


import os, sys
import math
import json, logging, argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms
from torchsummary import summary

# visualisation
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[4]:


data_dir = 'data/Kannada'


# In[5]:


get_ipython().run_cell_magic('bash', '', '# Install Kaggle CLI using Python Pip\npip3 install --quiet kaggle\nmkdir -p ~/.kaggle\n\n# Copy API key file to where Kaggle expects it\n# Make sure to upload kaggle key file next to this notebook\ncp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json')


# In[6]:


get_ipython().run_cell_magic('bash', '', '\n# Download Kannada MNIST from Kaggle\nkaggle competitions download -c Kannada-MNIST\n\n# Create our data directory\nmkdir -p data/Kannada/raw\nmkdir -p data/Kannada/processed\n\n# Unzip to data/Kannada directory\nunzip Kannada-MNIST.zip -d data/Kannada/raw')


# In[7]:


for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        print('data at: ' + os.path.join(dirname, filename))


# In[8]:


from sklearn.model_selection import train_test_split

# # Load Data
train = pd.read_csv(os.path.join(data_dir, 'raw/train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'raw/Dig-MNIST.csv'))
submission_set = pd.read_csv(os.path.join(data_dir, 'raw/test.csv')).iloc[:,1:]

# # Seperate train data and labels
train_data = train.drop('label',axis=1)
train_targets = train['label']

# # Seperate test data and labels
test_images=test.drop('label',axis=1)
test_labels=test['label']


# In[9]:


# Train Test Split for validation
train_images, val_images, train_labels, val_labels = train_test_split(train_data, 
                                                                     train_targets, 
                                                                     test_size=0.15)


# In[10]:


# Reset Index
train_images.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)

val_images.reset_index(drop=True, inplace=True)
val_labels.reset_index(drop=True, inplace=True)

test_images.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)


# In[11]:


train_images.iloc[20000:20005, 200:320]


# In[12]:


print("Train Set: \n" + '-'*20)
print(train_images.shape)
print(train_labels.shape)


# In[13]:


val_images.iloc[8000:8005, 200:320]


# In[14]:


print("\nValidation Set: \n"  + '-'*20)
print(val_images.shape)
print(val_labels.shape)


# In[15]:


test_images.iloc[5000:5005, 200:320]


# In[16]:


print("\nTest Set: \n"  + '-'*20)
print(test_images.shape)
print(test_labels.shape)

print("\nSubmission: ")
print(submission_set.shape)


# In[17]:


train_dist = train_labels.value_counts(normalize = True)
test_dist = test_labels.value_counts(normalize = True)
submission_dist = train_labels.value_counts(normalize = True)

# display table for visualising dataset distribution
pd.DataFrame({
    'Trainset Distribution': train_dist,
    'Testset Distribution': test_dist,
    'Submissionset Distribution': submission_dist
})


# In[18]:


class KannadaDataSet(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i,:]
        data = np.array(data).astype(np.uint8).reshape(IMGSIZE,IMGSIZE,1)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            # for train set, val set, and test set
            return (data, self.y[i])
        else:
            # for kaggle submission
            # since submission set will not have labels
            return data


# In[19]:


IMGSIZE = 28

# Transformations for the train
train_trans = transforms.Compose(([
    transforms.ToPILImage(),
    transforms.RandomCrop(IMGSIZE),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ToTensor(), # automatically divide pixels by 255
]))

# Transformations for the validation & test sets
val_trans = transforms.Compose(([
    transforms.ToPILImage(),
    transforms.ToTensor(), # automatically divide pixels by 255
]))


# In[20]:


batch_size = 64

# Initialise dataset object for each set
train_data = KannadaDataSet(train_images, train_labels, train_trans)
val_data   = KannadaDataSet(val_images, val_labels, val_trans)
test_data  = KannadaDataSet(test_images, test_labels, val_trans)
submission_data = KannadaDataSet(submission_set, None, val_trans)

# Define Dataloader for each set
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(val_data, 
                        batch_size=batch_size, # batch_size=1000
                        shuffle=False)

test_loader = DataLoader(test_data,
                         batch_size=batch_size, # batch_size=1000
                         shuffle=False)

# for kaggle submission
submission_loader = DataLoader(submission_data,
                               batch_size=batch_size,
                               shuffle=False)


# In[21]:


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 6))
for idx in np.arange(16):
    ax = fig.add_subplot(2, 16/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title('digit ' + str(labels[idx].item()), fontsize=16)  # .item() gets single value in scalar tensor


# In[22]:


img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5

for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')

ax.set_title('Kannada Digit in detail: label %d' % labels[1].item());


# In[23]:


# Save data to local folder first
train_images.to_csv(os.path.join(data_dir, 'processed/train.csv'), index=False, header=False)
train_labels.to_csv(os.path.join(data_dir, 'processed/train_labels.csv'), index=False, header=False)

val_images.to_csv(os.path.join(data_dir, 'processed/validation.csv'), index=False, header=False)
val_labels.to_csv(os.path.join(data_dir, 'processed/validation_labels.csv'), index=False, header=False)

test_images.to_csv(os.path.join(data_dir, 'processed/test.csv'), index=False, header=False)
test_labels.to_csv(os.path.join(data_dir, 'processed/test_labels.csv'), index=False, header=False)


# In[24]:


class KannadaCNN(nn.Module):
    """ Convolutional Neural Network
    """
    def __init__(self, drop_p=0.4):
        super().__init__()
        
        # First hidden layer
        self.conv2d_0 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.convbn_0 = nn.BatchNorm2d(num_features=64)

        self.conv2d_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.convbn_1 = nn.BatchNorm2d(num_features=64)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_1 = nn.Dropout2d(p=drop_p)

        # Second hidden layer
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.convbn_2 = nn.BatchNorm2d(num_features=128)

        self.conv2d_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.convbn_3 = nn.BatchNorm2d(num_features=128)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_2 = nn.Dropout2d(p=drop_p)

        # Third hidden layer
        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.convbn_4 = nn.BatchNorm2d(num_features=256)
        
        self.conv2d_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convbn_5 = nn.BatchNorm2d(num_features=256)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_3 = nn.Dropout(p=drop_p)

        # Dense fully connected layer
        self.dense_linear_1 = nn.Linear(256*3*3, 512)
        self.drop_4 = nn.Dropout(p=drop_p)

        self.dense_linear_2 = nn.Linear(512, 256)
        self.drop_5 = nn.Dropout(p=drop_p)

        self.dense_linear_3 = nn.Linear(256, 128)
        self.out_layer = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.convbn_0(x)
        x = F.leaky_relu(x)
        
        x = self.conv2d_1(x)
        x = self.convbn_1(x)
        x = F.leaky_relu(x)

        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv2d_2(x)
        x = self.convbn_2(x)
        x = F.leaky_relu(x)

        x = self.conv2d_3(x)
        x = self.convbn_3(x)
        x = F.leaky_relu(x)

        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv2d_4(x)
        x = self.convbn_4(x)
        x = F.leaky_relu(x)
        
        x = self.conv2d_5(x)
        x = self.convbn_5(x)
        x = F.leaky_relu(x)
        
        x = self.pool_3(x)
        x = self.drop_3(x)

        x = x.view(-1, 256*3*3)
        x = self.dense_linear_1(x)
        x = F.relu(x)
        x = self.drop_4(x)
        
        x = self.dense_linear_2(x)
        x = F.relu(x)
        x = self.drop_5(x)
        
        x = self.dense_linear_3(x)
        x = F.relu(x)

        out = self.out_layer(x)
        return out


# In[25]:


# Constructing our CNN module
model = KannadaCNN().to(device)
# initialise network
net = KannadaCNN().to(device)

# optimiser
optimiser = optim.Adam(net.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

# display model summary
summary(model, input_size=(1,IMGSIZE,IMGSIZE))  # IMGSIZE = 28


# In[26]:


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[27]:


def train_helper(train_loader, model, optimizer, criterion,
                  epoch, device='cpu', log_interval=25):
    # set to training mode
    model.train()

    # training result to record
    train_loss = 0.0
    train_top1 = 0.0
    train_top5 = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        # convert tensor for current runtime device
        data, target = data.to(device), target.to(device)

        # reset optimiser gradient to zero
        optimizer.zero_grad()

        # feed forward
        out = model(data)
        
        # calculate loss and optimise network params
        loss = criterion(out, target)
        loss.backward()
        
        # optimize weight to account for loss/gradient
        optimizer.step()

        # calculate training accuracy for top1 and top5
        top1, top5 = accuracy(out, target, topk=(1,5))

        # update result records
        train_top1 += top1.item()
        train_top5 += top5.item()
        train_loss += loss.item()

        # logging loss output to stdout
        if batch_idx % log_interval == 0:
            print('Train Epoch: {:03d} [{:05d}/{:05d} ({:2.0f}%)] | '
                  'Top1 g Acc: {:4.1f} \t| Top5 Acc: {:4.1f} \t| Loss: {:.4f}'
                  .format(epoch, batch_idx * len(data), len(train_loader.sampler),
                      100 * batch_idx / len(train_loader),
                      top1, top5, loss.item()))

    # display training result
    train_loss /= len(train_loader.dataset)
    train_top1 /= len(train_loader) # average loss over mini-batches
    train_top5 /= len(train_loader) # average loss over mini-batches

    print('Training Summary Epoch: {:03d} | '
          'Average Top1 Acc: {:.2f}  | Average Top5 Acc: {:.2f} | Loss: {:.4f}'
          .format(epoch, train_top1, train_top5, train_loss))
    
    return train_loss, train_top1, train_top5 


# In[28]:


def test_helper(test_loader, model, criterion, 
                 epoch, device='cpu'):
    # set to validation mode
    model.eval()
    
    test_loss = 0.0  # record testing loss
    test_top1 = 0.0
    test_top5 = 0.0
    for batch_idx, (data, target) in enumerate(test_loader, start=1):

        # convert tensor for current runtime device
        data, target = data.to(device), target.to(device)

        # generate image x
        out = model(data)

        # calculate loss and optimise network params
        loss = criterion(out, target)
        
        # calculate testing accuracy for top1 and top5
        top1, top5 = accuracy(out, target, topk=(1,5))

        # update test loss
        test_top1 += top1.item()
        test_top5 += top5.item()
        test_loss += loss.item()

    # display validation/testing result
    test_loss /= len(test_loader.dataset)  # average loss over all images
    test_top1 /= len(test_loader)
    test_top5 /= len(test_loader)

    print('Val/Test Summary Epoch: {:03d} | '
          'Average Top1 Acc: {:.2f}  | Average Top5 Acc: {:.2f} | Loss: {:.4f}'
          .format(epoch, test_top1, test_top5, test_loss))
    
    return test_loss, test_top1, test_top5


# In[29]:


# ----------------------------
# TRAINING SESSION
# ----------------------------

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

for epoch in range(1, 5 + 1):
    print('\n' + '-' * 100)
    # run session on training set
    train_loss, train_acc, _  = train_helper(train_loader, net, optimiser, criterion,
                                              epoch=epoch, device=device, log_interval=100)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # run session on validation set
    val_loss, val_acc, _ = test_helper(val_loader, net, criterion, epoch, device=device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

# finally, run session testing set
print('\n' + 'Final Test Set Result:\n'+ '*' * 80)
test_helper(test_loader, net, criterion, epoch, device=device);


# In[30]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# Plot Error training vs validation
axes[0].plot(train_losses);
axes[0].plot(val_losses);

axes[0].set_ylabel('Error');
axes[0].set_xlabel('Epochs');

axes[0].set_ylim(0, 0.005);
axes[0].legend(labels=['train error', 'validation error']);

# Plot Accuracy training vs validation
axes[1].plot(train_accuracies);
axes[1].plot(val_accuracies);

axes[1].set_ylabel('Accuracy (%)');
axes[1].set_xlabel('Epochs');

axes[1].set_ylim(80, 100);
axes[1].legend(labels=['train acc', 'validation acc']);


# In[31]:


# Time to get the network's predictions on the test set
# Put the test set in a DataLoader

net.eval() # Safety first
predictions = torch.LongTensor().to(device) # Tensor for all predictions

# Go through the test set, saving the predictions in... 'predictions'
for images in submission_loader:
    images = images.to(device)
    preds = net(images)
    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)


# In[32]:


submission_pred_df = pd.DataFrame(predictions.cpu().detach().numpy())


# In[33]:


submission_pred_df.to_csv(os.path.join(data_dir, 'kannada_sub_baseline.csv'), 
                          index=True, index_label='id', header=['label'])


# In[ ]:




