#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Wavelet decomposition using pywavelets
To install: pip install pywavelets
"""
import pywt
import random
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import spectrogram
from scipy.signal import resample
import matplotlib.pyplot as plt


# In[2]:


preictal_tst = '../input/seizure-prediction/Patient_1/Patient_1/Patient_1_preictal_segment_0001.mat'
preictal_data = scipy.io.loadmat(preictal_tst)


# In[3]:


preictal_data


# In[4]:


preictal_array = preictal_data['preictal_segment_1'][0][0][0]
#EXTRA
print(type(preictal_data['preictal_segment_1']) , preictal_data['preictal_segment_1'][0][0][0].shape)


# In[5]:


preictal_array


# In[6]:


# five decomposition coefficients
cA,cD4,cD3,cD2,cD1 = pywt.wavedec(preictal_array, pywt.Wavelet('db4'), level = 4)


# In[7]:


tot_data = [cA, cD4, cD3, cD2, cD1]


# In[8]:


list(set(tot_data[1][1]))


# In[9]:


import math

# renyi
def renyi_entropy(d1):
    """
    d1 shape: (Sample count, Sample length)
    """
    d1=np.rint(d1)
    rend1=[]
    alpha=2    
    for i in range(d1.shape[0]):
        X=d1[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend1.append(Renyi_En)
    return(rend1)


# In[ ]:


for i in range(len(tot_data)):
    renyi_ent[i] = renyi_entropy(tot_data[i])


# In[ ]:


# permutation entropy
from pyentrp import entropy
def permu(d1):
    pd1=[]
    for i in range(d1.shape[0]):
        X=d1[i]
        pd1.append(entropy.permutation_entropy(X,3,1))
    return(pd1)


# In[ ]:


for i in range(len(tot_data)):
    perm_ent[i] = permu(tot_data[i])


# In[ ]:


from scipy.special import gamma,psi
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors

def kraskov_entropy(d1):
    k=4
    def nearest_distances(X, k):
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        d, _ = knn.kneighbors(X)
        return d[:, -1]
    def entropy(X, k):
        r = nearest_distances(X, k)
        n, d = X.shape
        volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
        return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))+ np.log(volume_unit_ball) + psi(n) - psi(k))
    kd1=[]
    for i in range(d1.shape[0]):
        x=d1[i]
        x=np.array(x).reshape(-1,1)
        kd1.append(entropy(x, k))
    return(kd1)


# In[ ]:


for i in range(len(tot_data)):
    krak_ent[i] = kraskov_entropy(tot_data[i])


# In[ ]:


sample entropy
def sampl(d1):
   sa1=[]
   for i in range(d1.shape[0]):
       X=d1[i]
       std_X = np.std(X)
       ee=entropy.sample_entropy(X,2,0.2*std_X)
       sa1.append(ee[0])
   return(sa1)


# In[ ]:


for i in range(len(tot_data)):
    sample_ent[i] = sampl(tot_data[i])


# In[ ]:


# shannon entropy
def shan(d1):
    sh1=[]
    d1=np.rint(d1)
    for i in range(d1.shape[0]):
        X=d1[i]
        sh1.append(entropy.shannon_entropy(X))
    return(sh1)


# In[ ]:


for i in range(len(tot_data)):
    shan_ent[i] = shan(tot_data[i])


# In[ ]:


# stack to have shape (sample count, number of features)
my_data = np.vstack((np.array(shan_ent), np.array(sample_ent), 
                     np.array(krak_ent), np.array(perm_ent), np.array(renyi_ent)))

