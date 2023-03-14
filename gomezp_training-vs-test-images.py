#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load necessary modules
import os, cv2
from scipy import stats
from glob import glob 
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook,trange
import matplotlib.pyplot as plt


# In[2]:


def load_data(N,df):
    """ This functions loads N images using the data df
    """
    # allocate a numpy array for the images (N, 96x96px, 3 channels, values 0 - 255)
    X = np.zeros([N,96,96,3],dtype=np.uint8) 
    #if we have labels for this data, also get them
    if 'label' in df.columns:
        y = np.squeeze(df.as_matrix(columns=['label']))[0:N]
    else:
        y = None
    #read images one by one, tdqm notebook displays a progress bar
    for i, row in tqdm_notebook(df.iterrows(), total=N):
        if i == N:
            break
        X[i] = cv2.imread(row['path'])
          
    return X,y


# In[3]:


#set paths to training and test data
path = "../input/" #adapt this path, when running locally
train_path = path + 'train/'
test_path = path + 'test/'

df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))}) # load the filenames
df_test = pd.DataFrame({'path': glob(os.path.join(test_path,'*.tif'))}) # load the test set filenames
df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0]) # keep only the file names in 'id'
labels = pd.read_csv(path+"train_labels.csv") # read the provided labels
df = df.merge(labels, on = "id") # merge labels and filepaths
df.head(3) # print the first three entrys


# In[4]:


#shuffle the dataframes to a representative sample
df = df.sample(frac=1,random_state = 42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state = 4242).reset_index(drop=True)

# Load N images from the training set
N = 50000
print("Loading training data samples...")
X,y = load_data(N=N,df=df) 
print("Loading test data samples...")
X_test,_ = load_data(N=N,df=df_test) 
print("Done.")


# In[5]:


nr_of_bins = 256
fig,axs = plt.subplots(1,2,sharey=True, sharex = False, figsize=(8,2),dpi=150)
training_brightness = np.mean(X,axis=(1,2,3))
test_brightness = np.mean(X_test,axis=(1,2,3))
axs[0].hist(training_brightness, bins=nr_of_bins, density=True)
axs[1].hist(test_brightness, bins=nr_of_bins, density=True)
axs[0].legend(("Mean={:2.2f}|Std={:2.2f}".format(np.mean(training_brightness),np.std(training_brightness)),),loc=2,prop={'size': 6})
axs[1].legend(("Mean={:2.2f}|Std={:2.2f}".format(np.mean(test_brightness),np.std(test_brightness)),),loc=2,prop={'size': 6})
axs[0].set_title("Mean brightness, training samples")
axs[1].set_title("Mean brightness, test samples")
axs[0].set_xlabel("Image mean brightness")
axs[1].set_xlabel("Image mean brightness")
axs[0].set_ylabel("Relative frequency")
axs[1].set_ylabel("Relative frequency");


# In[6]:


fig = plt.figure(figsize=(6,3),dpi=150)
plt.hist(training_brightness, bins=nr_of_bins, density=True,cumulative=True, alpha = 0.5)
plt.hist(test_brightness, bins=nr_of_bins, density=True,cumulative=True, alpha = 0.5);
plt.legend(("Training","Test"),loc=2,prop={'size': 6})
# axs[1].legend(("Mean={:2.2f}|Std={:2.2f}".format(np.mean(test_brightness),np.std(test_brightness)),),loc=2,prop={'size': 6})
plt.title("Cumulative mean brightness, training vs test")
plt.xlabel("Image mean brightness")
plt.ylabel("Cumulative frequency")
plt.show()


# In[7]:


nr_of_bins = 256 #each possible pixel value will get a bin in the following histograms
N_hist = 10000 #we will limit the nr of images we look at, because otherwise this tends to kill the kernel...
fig,axs = plt.subplots(4,2,sharey=True,figsize=(8,8),dpi=150)

#RGB channels
axs[0,0].hist(X[0:N_hist,:,:,0].flatten(),bins=nr_of_bins,density=True)
axs[0,1].hist(X_test[0:N_hist,:,:,0].flatten(),bins=nr_of_bins,density=True)
axs[1,0].hist(X[0:N_hist,:,:,1].flatten(),bins=nr_of_bins,density=True)
axs[1,1].hist(X_test[0:N_hist,:,:,1].flatten(),bins=nr_of_bins,density=True)
axs[2,0].hist(X[0:N_hist,:,:,2].flatten(),bins=nr_of_bins,density=True)
axs[2,1].hist(X_test[0:N_hist,:,:,2].flatten(),bins=nr_of_bins,density=True)

#All channels
axs[3,0].hist(X[0:N_hist].flatten(),bins=nr_of_bins,density=True)
axs[3,1].hist(X_test[0:N_hist].flatten(),bins=nr_of_bins,density=True)

#Set image labels
axs[0,0].set_title("Training samples (N =" + str(X.shape[0]) + ")");
axs[0,1].set_title("Test samples (N =" + str(X_test.shape[0]) + ")");

axs[0,1].set_ylabel("Red",rotation='horizontal',labelpad=35,fontsize=12)
axs[1,1].set_ylabel("Green",rotation='horizontal',labelpad=35,fontsize=12)
axs[2,1].set_ylabel("Blue",rotation='horizontal',labelpad=35,fontsize=12)
axs[3,1].set_ylabel("RGB",rotation='horizontal',labelpad=35,fontsize=12)
for i in range(4):
    axs[i,0].set_ylabel("Relative frequency")
axs[3,0].set_xlabel("Pixel value")
axs[3,1].set_xlabel("Pixel value")
fig.tight_layout()


# In[8]:


#first count pixels with value 255 in train and test data per image
bright_pixels_train = (X == 255).sum(axis=(1,2,3))
bright_pixels_test = (X_test == 255).sum(axis=(1,2,3))


# In[9]:


N_bright_train,N_bright_test,N_bright_positive_labels = [],[],[]
xtics = range(0,5000,100)
for threshold in xtics:
    #count images with more than threshold 255 pixels
    N_bright_train.append((bright_pixels_train > threshold).sum() / N) 
    N_bright_test.append((bright_pixels_test > threshold).sum() / N) 
    #count positive samples
    N_bright_positive_labels.append(y[bright_pixels_train > threshold].sum() / (bright_pixels_train > threshold).sum())
    
fig = plt.figure(figsize=(6,3),dpi=150)
plt.plot(xtics,N_bright_train)
plt.plot(xtics,N_bright_test)
plt.plot(xtics,N_bright_positive_labels)
plt.legend(("Training images over threshold","Test images over threshold","Positive samples portion (Training images)"),loc=1,prop={'size': 6})
plt.title("Frequency of bright pixels in training and test with positive rate for training")
plt.xlabel("Pixel Number Threshold (how many pixels have to be 255)")
plt.ylabel("Relative frequency")
plt.show()


# In[10]:


#let's take those images where between 1475 and 1525 pixels have values of 255
bright_train_imgs = X[np.logical_and(bright_pixels_train > 1475,bright_pixels_train < 1525)] 
bright_test_imgs = X_test[np.logical_and(bright_pixels_test > 1475,bright_pixels_test < 1525)]

#train
fig = plt.figure(figsize=(8, 5), dpi=100)
np.random.seed(100) #we can use the seed to get a different set of random images
fig.suptitle("Images with 1475 < B < 1500 \n Training samples (N =" + str(bright_train_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_train_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_train_imgs[idx]) #plot image
    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image

#test
fig = plt.figure(figsize=(8, 4), dpi=100)
fig.suptitle("Test samples (N =" + str(bright_test_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_test_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_test_imgs[idx]) #plot image


# In[11]:


#let's take those images where between 2400 and 2600 pixels have values of 255
bright_train_imgs = X[np.logical_and(bright_pixels_train > 2400,bright_pixels_train < 2600)] 
bright_test_imgs = X_test[np.logical_and(bright_pixels_test > 2400,bright_pixels_test < 2600)]

#train
fig = plt.figure(figsize=(8, 5), dpi=100)
np.random.seed(42) #we can use the seed to get a different set of random images
fig.suptitle("Images with 2400 < B < 2600 \n Training samples (N =" + str(bright_train_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_train_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_train_imgs[idx]) #plot image
    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image
    
#test
fig = plt.figure(figsize=(8, 4), dpi=100)
fig.suptitle("Test samples (N =" + str(bright_test_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_test_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_test_imgs[idx]) #plot image


# In[12]:


#calculate relative and absolute incidence of 255 pixels
nr_train = (X == 255).sum() / N
nr_test = (X_test == 255).sum() / N
freq_train = nr_train*100 / (96*96*3)
freq_test = nr_test*100 / (96*96*3)

print("Nr of pixels with value 255 per image \nTraining data - {:.4f}% ; avg. Nr = {:.0f}; maxval = {} \nTest - {:.4f}% ; avg. Nr = {:.0f}; maxval = {}".format(freq_train,nr_train,np.max(bright_pixels_train),freq_test,nr_test,np.max(bright_pixels_test)))


# In[13]:


#let's take those images with high mean values (> 220)
bright_train_imgs = X[np.mean(X,axis=(1,2,3)) > 220] 
bright_test_imgs = X_test[np.mean(X_test,axis=(1,2,3)) > 220]

#train
fig = plt.figure(figsize=(8, 5), dpi=100)
np.random.seed(100) #we can use the seed to get a different set of random images
fig.suptitle("Images with mean brightness over 220 \n Training samples (N =" + str(bright_train_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_train_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_train_imgs[idx]) #plot image
    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image
    
#test
fig = plt.figure(figsize=(8, 4), dpi=100)
fig.suptitle("Test samples (N =" + str(bright_test_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_test_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_test_imgs[idx]) #plot image


# In[14]:


training_brightness = np.mean(X,axis=(1,2,3))
test_brightness = np.mean(X_test,axis=(1,2,3))

N_bright_train,N_bright_test,N_bright_positive_labels = [],[],[]
xtics = range(100,255,1)
for threshold in xtics:
    #count images with more than threshold 255 pixels
    N_bright_train.append((training_brightness > threshold).sum() / N) 
    N_bright_test.append((test_brightness > threshold).sum() / N) 
    #count positive samples
    N_bright_positive_labels.append(y[bright_pixels_train > threshold].sum() / (bright_pixels_train > threshold).sum())
    
fig = plt.figure(figsize=(6,3),dpi=150)
plt.plot(xtics,N_bright_train)
plt.plot(xtics,N_bright_test)
plt.plot(xtics,N_bright_positive_labels)
plt.legend(("Training images over threshold","Test images over threshold","Positive samples portion (Training images)"),loc=1,prop={'size': 6})
plt.title("Frequency of bright pixels in training and test with positive rate for training")
plt.xlabel("Pixel Number Threshold (how many pixels have to be 255)")
plt.ylabel("Relative frequency")
plt.show()

